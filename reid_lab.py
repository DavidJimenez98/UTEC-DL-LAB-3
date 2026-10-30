"""MLCOR + CEM for the Laboratory 3 unsupervised person Re-ID task.

The implementation intentionally never accesses identity labels while training.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models, transforms
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def all_h5_keys(group: h5py.Group, prefix: str = "") -> list[str]:
    result = []
    for name, item in group.items():
        key = f"{prefix}/{name}" if prefix else name
        result.append(key)
        if isinstance(item, h5py.Group):
            result.extend(all_h5_keys(item, key))
    return result


def key(h5: h5py.File, *candidates: str) -> str:
    keys = set(all_h5_keys(h5))
    for candidate in candidates:
        if candidate in keys:
            return candidate
    raise KeyError(f"No encontre ninguna clave de {candidates}. Claves: {sorted(keys)}")


class H5Images(Dataset):
    def __init__(self, h5_path: str, split: str = "train", transform=None):
        self.path, self.split, self.transform = h5_path, split, transform
        with h5py.File(h5_path, "r") as h5:
            base = "" if split == "train" else split
            image_name = key(h5, f"{base}/images".strip("/"), "images")
            self.image_key = image_name
            self.camera_key = next((x for x in (f"{base}/camera_ids", "camera_ids") if x in all_h5_keys(h5)), None)
            self.id_key = next((x for x in (f"{base}/image_ids", "image_ids") if x in all_h5_keys(h5)), None)
            self.n = len(h5[image_name])
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")

    def __len__(self): return self.n

    def __getitem__(self, i):
        self._open()
        image = self._h5[self.image_key][i]
        # HDF5 commonly stores HWC uint8; also support CHW.
        image = torch.from_numpy(np.asarray(image)).permute(2, 0, 1) if image.ndim == 3 and image.shape[-1] in (1, 3) else torch.from_numpy(np.asarray(image))
        image = image.float().div(255.0) if image.max() > 1 else image.float()
        if self.transform: image = self.transform(image)
        camera = int(self._h5[self.camera_key][i]) if self.camera_key else -1
        # Market-1501 image_ids are file-name-like byte strings; they are metadata,
        # not a numeric training target. Keep the dataset index as a stable ID.
        image_id = i
        return image, camera, image_id, i


def train_transform():
    return transforms.Compose([transforms.Resize((256, 128), antialias=True), transforms.RandomHorizontalFlip(), transforms.Pad(10), transforms.RandomCrop((256, 128)), transforms.RandomErasing(p=0.5), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])


def test_transform():
    return transforms.Compose([transforms.Resize((256, 128), antialias=True), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])


class GeM(nn.Module):
    def __init__(self, p=3.0): super().__init__(); self.p = nn.Parameter(torch.ones(1) * p)
    def forward(self, x):
        p = self.p.clamp_min(1e-3)
        return F.avg_pool2d(x.clamp_min(1e-6).pow(p), x.shape[-2:]).pow(1 / p).flatten(1)


class ReIDNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.gem, self.bn = GeM(), nn.BatchNorm1d(2048)

    def forward(self, x):
        fmap = self.backbone(x)
        glob = self.bn(self.gem(fmap))
        upper = self.gem(fmap[:, :, : fmap.shape[2] // 2])
        lower = self.gem(fmap[:, :, fmap.shape[2] // 2 :])
        return tuple(F.normalize(z, dim=1) for z in (glob, upper, lower))


@torch.no_grad()
def features(model, dataset, batch=128, workers=0):
    model.eval(); loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=DEVICE.type == "cuda")
    parts = [[], [], []]
    for images, *_ in tqdm(loader, desc="features"):
        out = model(images.to(DEVICE, non_blocking=True))
        for j in range(3): parts[j].append(out[j].cpu())
    return [torch.cat(x) for x in parts]


@torch.no_grad()
def spherical_kmeans(fused: np.ndarray, groups: int, iterations: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Small-memory cosine K-means on the available GPU (or CPU)."""
    data = torch.from_numpy(fused).to(DEVICE)
    rng = torch.Generator().manual_seed(42)
    seeds = torch.randperm(len(data), generator=rng)[:groups].to(DEVICE)
    centers = data[seeds].clone()
    assignments = torch.empty(len(data), dtype=torch.long, device=DEVICE)
    confidence = torch.empty(len(data), device=DEVICE)
    for _ in range(iterations):
        for start in range(0, len(data), 1024):
            score, group = (data[start:start + 1024] @ centers.T).max(dim=1)
            assignments[start:start + 1024] = group
            confidence[start:start + 1024] = score
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, data)
        counts = torch.bincount(assignments, minlength=groups)
        updated = F.normalize(sums, dim=1)
        empty = counts == 0
        if empty.any():
            updated[empty] = data[seeds[:int(empty.sum())]]
        centers = updated
    for start in range(0, len(data), 1024):
        score, group = (data[start:start + 1024] @ centers.T).max(dim=1)
        assignments[start:start + 1024] = group
        confidence[start:start + 1024] = score
    return assignments.cpu().numpy(), confidence.cpu().numpy()


def fused_features(feats: list[Tensor]) -> Tensor:
    # Cosine-distance adaptation of the paper's 0.70/0.15/0.15 fusion.
    weights = (0.70 ** 0.5, 0.15 ** 0.5, 0.15 ** 0.5)
    return F.normalize(torch.cat([weight * branch for weight, branch in zip(weights, feats)], dim=1), dim=1)


def cluster(feats: list[Tensor], eps: float) -> tuple[np.ndarray, float | None]:
    """Reject degenerate DBSCAN clusters; bootstrap with unsupervised K-means."""
    # Global/upper/lower information is fused before DBSCAN, as in the paper.
    fused = fused_features(feats).numpy()
    candidates = []
    for candidate in (eps, 0.25, 0.20):
        if candidate not in candidates:
            candidates.append(candidate)
    minimum_assigned = max(32, int(0.10 * len(fused)))
    for candidate in candidates:
        labels = DBSCAN(eps=candidate, min_samples=4, metric="cosine", n_jobs=-1).fit_predict(fused)
        sizes = np.bincount(labels[labels >= 0])
        clusters = len(sizes)
        assigned = int((labels >= 0).sum())
        largest = int(sizes.max()) if len(sizes) else 0
        print(f"DBSCAN eps={candidate:.3f}: {clusters} clusters, {len(labels) - assigned} outliers, largest={largest}")
        if clusters >= 20 and assigned >= minimum_assigned and largest <= 0.20 * assigned:
            return labels, candidate
    # ImageNet features sometimes connect thousands of people into one DBSCAN
    # component. A deterministic K-means partition prevents the resulting
    # single-class supervision from collapsing the embedding. Low-confidence
    # assignments remain outliers, preserving a meaningful MLCOR comparison.
    groups = min(300, max(20, len(fused) // 30))
    labels, confidence = spherical_kmeans(fused, groups)
    for group in range(groups):
        members = np.where(labels == group)[0]
        if len(members) >= 5:
            threshold = np.quantile(confidence[members], 0.15)
            labels[members[confidence[members] < threshold]] = -1
    print(f"K-means fallback: {groups} clusters, {(labels < 0).sum()} outliers")
    return labels, None


@torch.no_grad()
def mlcor(feats: list[Tensor], labels: np.ndarray, neighbors=5, budget=40,
          sigma=0.5, consensus=0.6) -> np.ndarray:
    """Reciprocal-neighbor MLCOR with confidence tiers and a per-round budget."""
    outliers = np.flatnonzero(labels < 0)
    if not len(outliers) or not np.any(labels >= 0):
        return labels
    fused = fused_features(feats).to(DEVICE)
    k = min(neighbors, len(fused) - 1)
    indices = torch.empty(len(fused), k, dtype=torch.long)
    similarities = torch.empty(len(fused), k)
    for start in range(0, len(fused), 256):
        end = min(start + 256, len(fused))
        scores = fused[start:end] @ fused.T
        rows = torch.arange(end - start, device=DEVICE)
        scores[rows, torch.arange(start, end, device=DEVICE)] = -2
        values, nearest = scores.topk(k, dim=1)
        indices[start:end] = nearest.cpu()
        similarities[start:end] = values.cpu()
    neighbors_np = indices.numpy()
    similarities_np = similarities.numpy()
    candidates = {"high": [], "medium": [], "low": []}
    for image_index in outliers:
        nearby = neighbors_np[image_index]
        reciprocal = np.array([image_index in neighbors_np[j] for j in nearby])
        usable = reciprocal & (labels[nearby] >= 0)
        if not usable.any():
            continue
        valid_neighbors = nearby[usable]
        distances = 1.0 - similarities_np[image_index, usable]
        confidence = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        best = int(np.argmax(confidence))
        reliability = float(confidence[best])
        if reliability >= 0.60:
            tier = "high" if reliability >= 0.80 else "medium"
            label = int(labels[valid_neighbors[best]])
        elif reliability >= 0.40:
            scores = np.bincount(labels[valid_neighbors], weights=confidence,
                                 minlength=int(labels.max()) + 1)
            label = int(np.argmax(scores))
            if scores[label] / scores.sum() < consensus:
                continue
            tier = "low"
        else:
            continue
        candidates[tier].append((reliability, int(image_index), label))
    for tier in candidates:
        candidates[tier].sort(reverse=True)
    selected = candidates["high"][:budget]
    if len(selected) < budget // 2:
        selected.extend((candidates["medium"] + candidates["low"])[:budget - len(selected)])
    result = labels.copy()
    for _, image_index, label in selected:
        result[image_index] = label
    print(f"MLCOR: recovered {len(selected)}/{len(outliers)} outliers "
          f"(high={len(candidates['high'])}, medium={len(candidates['medium'])}, low={len(candidates['low'])})")
    return result


def remap(labels: np.ndarray) -> np.ndarray:
    good = sorted(set(labels[labels >= 0])); lookup = {old: new for new, old in enumerate(good)}
    return np.array([lookup.get(x, -1) for x in labels], dtype=np.int64)


class PseudoDataset(Dataset):
    def __init__(self, source, labels): self.source, self.labels = source, labels; self.ids = np.where(labels >= 0)[0]
    def __len__(self): return len(self.ids)
    def __getitem__(self, j):
        x, *_ = self.source[int(self.ids[j])]
        return x, int(self.labels[self.ids[j]]), int(self.ids[j])


class PKBatchSampler(Sampler[list[int]]):
    """Sample P pseudo-identities and K images per identity, as in the paper."""
    def __init__(self, labels: np.ndarray, batch_size: int, images_per_id: int = 4, seed: int = 42):
        if batch_size % images_per_id:
            raise ValueError("Batch size must be divisible by images_per_id")
        self.groups = {int(label): np.flatnonzero(labels == label) for label in np.unique(labels)}
        self.people = batch_size // images_per_id
        if len(self.groups) < self.people:
            raise ValueError(f"Need at least {self.people} pseudo-identities for PK sampling")
        self.images_per_id = images_per_id
        self.steps = max(1, len(labels) // batch_size)
        self.seed = seed
        self.epoch = 0

    def __len__(self): return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        people = np.array(list(self.groups))
        for _ in range(self.steps):
            selected = rng.choice(people, size=self.people, replace=False)
            batch = []
            for label in selected:
                members = self.groups[int(label)]
                batch.extend(rng.choice(members, size=self.images_per_id,
                                        replace=len(members) < self.images_per_id).tolist())
            rng.shuffle(batch)
            yield batch


def global_embedding_loss(global_features, labels, memory, memory_labels, classes, temperature=0.07):
    """Paper GEL: aggregate instance-memory evidence by pseudo-identity."""
    logits = global_features.float() @ memory.T / temperature
    weights = torch.exp(logits - logits.max(dim=1, keepdim=True).values.detach())
    scores = torch.zeros(len(labels), classes, device=weights.device)
    scores.scatter_add_(1, memory_labels[None, :].expand(len(labels), -1), weights)
    positive = scores.gather(1, labels[:, None]).squeeze(1)
    return (scores.sum(dim=1).clamp_min(1e-12).log() - positive.clamp_min(1e-12).log()).mean()


def cluster_driven_triplet_loss(outputs, labels, centroids, beta=0.2):
    """Reliability-weighted, batch-hard C-DTL with stop-gradient soft targets."""
    same = labels[:, None] == labels[None, :]
    same.fill_diagonal_(False)
    different = labels[:, None] != labels[None, :]
    valid = same.any(dim=1) & different.any(dim=1)
    if not valid.any():
        return outputs[0].sum() * 0.0
    branch_losses = []
    for branch, z in enumerate(outputs):
        similarity = z.float() @ z.float().T
        hardest_positive = similarity.masked_fill(~same, 2.0).argmin(dim=1)
        hardest_negative = similarity.masked_fill(same | torch.eye(len(z), device=z.device, dtype=torch.bool), -2.0).argmax(dim=1)
        positive_distance = (z.float() - z[hardest_positive].float()).norm(dim=1)
        negative_distance = (z.float() - z[hardest_negative].float()).norm(dim=1)
        with torch.no_grad():
            reference = torch.softmax(torch.stack((-positive_distance.detach(), -negative_distance.detach()), dim=1), dim=1)[:, 1]
            nearest = (z.float() @ centroids[branch].T).max(dim=1).values
            reliability = torch.exp(-0.5 * (2 - 2 * nearest.clamp(max=1)).clamp_min(0).sqrt())
        log_negative = F.log_softmax(torch.stack((-positive_distance, -negative_distance), dim=1), dim=1)[:, 1]
        # Eq. 20 already defines a log probability; do not take log(log Q).
        branch_losses.append(-(reliability[valid] * reference[valid] * log_negative[valid]).mean())
    return branch_losses[0] + beta * (branch_losses[1] + branch_losses[2])


def cem_losses(outputs, labels, centroids, memory, memory_labels, classes):
    gel = global_embedding_loss(outputs[0], labels, memory, memory_labels, classes)
    cdtl = cluster_driven_triplet_loss(outputs, labels, centroids)
    return cdtl + 0.6 * gel


@torch.no_grad()
def update_memories(outputs, labels, positions, memory, centers, momentum=0.1):
    """Update instance and cluster dictionaries; average repeated PK indices."""
    for target, indices, values in [(memory, positions, outputs[0])]:
        unique, inverse = torch.unique(indices, return_inverse=True)
        sums = torch.zeros(len(unique), values.shape[1], device=values.device)
        sums.index_add_(0, inverse, values.detach().float())
        means = sums / torch.bincount(inverse)[:, None]
        target[unique] = F.normalize(momentum * target[unique] + (1 - momentum) * means, dim=1)
    unique, inverse = torch.unique(labels, return_inverse=True)
    counts = torch.bincount(inverse)[:, None]
    for target, values in zip(centers, outputs):
        sums = torch.zeros(len(unique), values.shape[1], device=values.device)
        sums.index_add_(0, inverse, values.detach().float())
        target[unique] = F.normalize(momentum * target[unique] + (1 - momentum) * sums / counts, dim=1)


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    source = H5Images(args.train, transform=train_transform())
    extraction = H5Images(args.train, transform=test_transform())
    model = ReIDNet().to(DEVICE)
    start_round = 0
    current_eps = args.eps
    reviewed = args.protocol == "paper_adapted_v3"
    last_checkpoint = output / "last.pt"
    if reviewed and args.resume:
        numbered = sorted(output.glob("round_[0-9][0-9].pt"))
        if numbered:
            # The numbered checkpoint is committed before last.pt. It remains
            # authoritative if the process stops between those two replacements.
            last_checkpoint = numbered[-1]
    if args.resume and last_checkpoint.exists():
        previous = torch.load(last_checkpoint, map_location=DEVICE, weights_only=False)
        if previous.get("args", {}).get("protocol", "legacy_v2") != args.protocol:
            raise RuntimeError("Checkpoint protocol differs. Use a new output directory.")
        for setting in ("method", "epochs_per_round", "batch", "seed", "lr"):
            if reviewed and previous["args"].get(setting) != getattr(args, setting):
                raise RuntimeError(f"Cannot change {setting} while resuming this experiment")
        model.load_state_dict(previous["model"])
        start_round = int(previous["round"]) + 1
        current_eps = previous.get("current_eps", args.eps)
        print(f"Resuming {args.method} at round {start_round}/{args.rounds}")
    if start_round >= args.rounds:
        print(f"{args.method} already completed {args.rounds} rounds")
        return
    for rnd in range(start_round, args.rounds):
        if reviewed:
            # Identical round seeds for continuous runs and separate processes.
            random.seed(args.seed + rnd); np.random.seed(args.seed + rnd); torch.manual_seed(args.seed + rnd)
        round_started = time.monotonic()
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        feats = features(model, extraction, args.extract_batch, args.workers)
        labels, effective_eps = cluster(feats, current_eps)
        if effective_eps is not None:
            current_eps = effective_eps
        initial_outliers = int((labels < 0).sum())
        if args.method == "full": labels = mlcor(feats, labels)
        labels = remap(labels); classes = labels.max() + 1
        if classes < 2: raise RuntimeError("DBSCAN no genero clusters: aumente --eps.")
        data = PseudoDataset(source, labels)
        sampler = PKBatchSampler(labels[data.ids], args.batch, seed=args.seed + rnd)
        loader = DataLoader(data, batch_sampler=sampler, num_workers=args.workers,
                            pin_memory=DEVICE.type == "cuda", persistent_workers=reviewed and args.workers > 0)
        classifier = nn.ModuleList([nn.Linear(2048, classes, bias=False).to(DEVICE) for _ in range(3)])
        opt = torch.optim.Adam([
            {"params": model.parameters(), "lr": args.lr * 0.1},
            {"params": classifier.parameters(), "lr": args.lr},
        ], weight_decay=5e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
        centers = []
        for branch in feats:
            centers.append(F.normalize(torch.stack([branch[labels == c].mean(0) for c in range(classes)]), dim=1).to(DEVICE))
        memory = F.normalize(feats[0][data.ids].to(DEVICE), dim=1)
        memory_labels = torch.from_numpy(labels[data.ids]).long().to(DEVICE)
        memory_positions = np.full(len(labels), -1, dtype=np.int64)
        memory_positions[data.ids] = np.arange(len(data.ids))
        for epoch in range(args.epochs_per_round):
            global_epoch = rnd * args.epochs_per_round + epoch
            decay = 0.1 ** (global_epoch // 20) if reviewed else 1.0
            opt.param_groups[0]["lr"] = args.lr * 0.1 * decay
            opt.param_groups[1]["lr"] = args.lr * decay
            losses = []
            model.train()
            # Keep the ImageNet BatchNorm statistics stable for small,
            # pseudo-labelled batches; only learn their affine parameters.
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            for x, y, source_index in tqdm(loader, desc=f"{args.method} round {rnd + 1} epoch {epoch + 1}"):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    out = model(x)
                # Keep memory aggregation and distances in float32 under AMP.
                out = tuple(z.float() for z in out)
                ce = [F.cross_entropy(F.linear(z, F.normalize(head.weight, dim=1)) * 16.0, y)
                      for head, z in zip(classifier, out)]
                loss = (0.8 if reviewed else 0.6) * ce[0] + 0.2 * (ce[1] + ce[2])
                if args.method != "baseline":
                    loss = loss + args.cem_weight * cem_losses(out, y, centers, memory, memory_labels, classes)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; checkpoint not overwritten")
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                losses.append(float(loss.detach()))
                if args.method != "baseline":
                    with torch.no_grad():
                        positions = torch.from_numpy(memory_positions[source_index.numpy()]).to(DEVICE)
                        if reviewed:
                            update_memories(out, y, positions, memory, centers)
                        else:
                            memory[positions] = F.normalize(0.1 * memory[positions] + 0.9 * out[0].detach().float(), dim=1)
            print({"epoch": global_epoch + 1, "loss": float(np.mean(losses)),
                   "backbone_lr": opt.param_groups[0]["lr"]}, flush=True)
        state = {"model": model.state_dict(), "args": {k: v for k, v in vars(args).items() if k != "fn"}, "round": rnd, "current_eps": current_eps}
        # Atomic replacements prevent an interrupted Slurm job from leaving a
        # partially written checkpoint that cannot be resumed.
        for checkpoint_path in (output / f"round_{rnd:02d}.pt", output / "last.pt"):
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoint_path)
        summary = {"round": rnd, "clusters": int(classes), "outliers": int((labels < 0).sum()),
                   "outliers_before_mlcor": initial_outliers,
                   "recovered_outliers": initial_outliers - int((labels < 0).sum()),
                   "eps": effective_eps, "clustering": "dbscan" if effective_eps is not None else "spherical_kmeans",
                   "method": args.method, "protocol": args.protocol,
                   "epochs_completed": (rnd + 1) * args.epochs_per_round,
                   "optimizer_steps_this_round": len(loader) * args.epochs_per_round,
                   "pk_people": args.batch // 4, "pk_images_per_person": 4,
                   "round_seconds": time.monotonic() - round_started,
                   "last_epoch_loss": float(np.mean(losses)),
                   "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else 0}
        (output / "training_summary.json").write_text(json.dumps(summary, indent=2))
        (output / f"round_{rnd:02d}_training_summary.json").write_text(json.dumps(summary, indent=2))
        print(summary)


def _image_hwc(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    return image


def save_retrieval_examples(data_path, records, output_path):
    """Save five top-1 successes and five failures for the required analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    successes = [x for x in records if x["top1_correct"]][:5]
    failures = [x for x in records if not x["top1_correct"]][:5]
    selected = successes + failures
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 11, figsize=(16, 1.8 * len(selected)), squeeze=False)
    with h5py.File(data_path, "r") as h5:
        queries, gallery = h5["query/images"], h5["gallery/images"]
        for row, record in enumerate(selected):
            axes[row, 0].imshow(_image_hwc(queries[record["query_index"]]))
            outcome = "Success" if record["top1_correct"] else "Failure"
            axes[row, 0].set_title(f"Query\n{outcome}", fontsize=8)
            axes[row, 0].axis("off")
            for col, (gallery_index, correct) in enumerate(zip(record["gallery_indices"], record["matches"]), 1):
                ax = axes[row, col]
                ax.imshow(_image_hwc(gallery[gallery_index]))
                ax.axis("off")
                border = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                       edgecolor="green" if correct else "red", linewidth=2.5)
                ax.add_patch(border)
                ax.set_title(str(col), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = ReIDNet(pretrained=False).to(DEVICE); model.load_state_dict(checkpoint["model"]); model.eval()
    def get(split, low=False):
        ds = H5Images(args.data, split=split, transform=test_transform())
        if low:
            with h5py.File(args.data, "r") as h5:
                ds.image_key = key(h5, f"{split}/images_low_resolution")
                ds.n = len(h5[ds.image_key])
        z = features(model, ds, args.batch, args.workers)
        with h5py.File(args.data, "r") as h5:
            base = split
            pid = np.asarray(h5[key(h5, f"{base}/person_ids")]); cam = np.asarray(h5[key(h5, f"{base}/camera_ids")])
        return fused_features(z).numpy(), pid, cam
    gallery, gpid, gcam = get("gallery")
    results = {}
    example_records = []
    for name, low in (("original", False), ("low_resolution", True)):
        query, qpid, qcam = get("query", low)
        ranks, aps = [], []
        # Batched matrix multiplication is substantially faster than one BLAS
        # call per query and keeps peak RAM low on the Khipu node.
        for start in range(0, len(query), 128):
            score_batch = query[start:start + 128] @ gallery.T
            for offset, scores in enumerate(score_batch):
                query_index = start + offset
                pid, cam = qpid[query_index], qcam[query_index]
                if pid == -1:
                    continue
                order = np.argsort(-scores)
                # Market-1501 excludes junk gallery entries and same-person,
                # same-camera images from ranking.
                keep = (gpid[order] != -1) & ~((gpid[order] == pid) & (gcam[order] == cam))
                match = (gpid[order][keep] == pid)
                if not match.any():
                    continue
                ranks.append([float(match[:k].any()) for k in (1, 5, 10)])
                hits = np.cumsum(match)
                aps.append(float((hits / (np.arange(len(match)) + 1) * match).sum() / hits[-1]))
                if name == "original":
                    filtered_order = order[keep][:10]
                    example_records.append({
                        "query_index": query_index,
                        "query_person_id": int(pid),
                        "query_camera_id": int(cam),
                        "top1_correct": bool(match[0]),
                        "first_correct_rank": int(np.flatnonzero(match)[0] + 1),
                        "top1_gallery_person_id": int(gpid[filtered_order[0]]),
                        "top1_gallery_camera_id": int(gcam[filtered_order[0]]),
                        "gallery_indices": filtered_order.tolist(),
                        "matches": match[:10].tolist(),
                    })
        if not aps:
            raise RuntimeError(f"No valid queries were found for {name}")
        results[name] = {"mAP": float(np.mean(aps)), "Rank-1": float(np.mean(ranks, 0)[0]), "Rank-5": float(np.mean(ranks, 0)[1]), "Rank-10": float(np.mean(ranks, 0)[2]), "valid_queries": len(aps)}
    results["mAP_drop"] = results["original"]["mAP"] - results["low_resolution"]["mAP"]
    metrics_path = Path(args.metrics_output) if args.metrics_output else Path(args.checkpoint).with_name("metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))
    examples_json = metrics_path.with_name(metrics_path.stem + "_retrieval_examples.json")
    examples_json.write_text(json.dumps(example_records, indent=2))
    save_retrieval_examples(args.data, example_records, metrics_path.with_name(metrics_path.stem + "_retrieval_examples.png"))
    print(json.dumps(results, indent=2))


def inspect(args):
    with h5py.File(args.data, "r") as h5:
        for name in all_h5_keys(h5):
            if isinstance(h5[name], h5py.Dataset): print(f"{name}: shape={h5[name].shape}, dtype={h5[name].dtype}")


def preflight(args):
    """Fail fast on the login node before consuming a scheduled GPU job."""
    import sys
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10+ required, found {sys.version}")
    expected = {
        args.train: ("images", "camera_ids", "image_ids"),
        args.validation: (
            "query/images", "query/images_low_resolution", "query/person_ids", "query/camera_ids",
            "gallery/images", "gallery/person_ids", "gallery/camera_ids",
        ),
        args.testing: (
            "query/images", "query/images_low_resolution", "query/person_ids", "query/camera_ids",
            "gallery/images", "gallery/person_ids", "gallery/camera_ids",
        ),
    }
    dataset_shapes = {}
    for path, required in expected.items():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as h5:
            available = set(all_h5_keys(h5))
            missing = set(required) - available
            if missing:
                raise KeyError(f"{path}: missing HDF5 keys {sorted(missing)}")
            dataset_shapes[Path(path).name] = {name: list(h5[name].shape) for name in required}
    # Downloads/caches ImageNet weights now, when network access is available.
    model = ReIDNet(pretrained=True).to(DEVICE).eval()
    sample = H5Images(args.train, transform=test_transform())[0][0].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embeddings = model(sample)
    if any(tuple(x.shape) != (1, 2048) for x in embeddings):
        raise RuntimeError("Unexpected model output shape")
    # Exercise CEM forward/backward independently of the dataset.
    labels = torch.tensor([0, 1, 2, 0], device=DEVICE)
    outputs = tuple(F.normalize(torch.randn(4, 16, device=DEVICE), dim=1).requires_grad_() for _ in range(3))
    centers = [F.normalize(torch.randn(3, 16, device=DEVICE), dim=1) for _ in range(3)]
    memory = F.normalize(torch.randn(8, 16, device=DEVICE), dim=1)
    memory_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], device=DEVICE)
    cem_losses(outputs, labels, centers, memory, memory_labels, 3).backward()
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available_on_login": torch.cuda.is_available(),
        "datasets": dataset_shapes,
        "model_outputs": [list(x.shape) for x in embeddings],
        "status": "ok",
    }
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(required=True)
    p = sub.add_parser("inspect"); p.add_argument("--data", required=True); p.set_defaults(fn=inspect)
    p = sub.add_parser("preflight"); p.add_argument("--train", required=True); p.add_argument("--validation", required=True); p.add_argument("--testing", required=True); p.set_defaults(fn=preflight)
    p = sub.add_parser("train"); p.add_argument("--train", required=True); p.add_argument("--output", required=True); p.add_argument("--method", choices=["baseline", "cem_only", "full"], default="full"); p.add_argument("--rounds", type=int, default=8); p.add_argument("--epochs-per-round", type=int, default=8); p.add_argument("--batch", type=int, default=32); p.add_argument("--extract-batch", type=int, default=128); p.add_argument("--workers", type=int, default=0); p.add_argument("--eps", type=float, default=0.30); p.add_argument("--lr", type=float, default=3.5e-4); p.add_argument("--cem-weight", type=float, default=1.0); p.add_argument("--seed", type=int, default=42); p.add_argument("--resume", action="store_true"); p.set_defaults(fn=train)
    p.add_argument("--protocol", choices=["legacy_v2", "paper_adapted_v3"], default="legacy_v2")
    p = sub.add_parser("evaluate"); p.add_argument("--data", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--metrics-output"); p.add_argument("--batch", type=int, default=128); p.add_argument("--workers", type=int, default=0); p.set_defaults(fn=evaluate)
    args = parser.parse_args(); args.fn(args)

if __name__ == "__main__": main()
