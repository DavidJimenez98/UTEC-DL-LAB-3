"""Checks for the paper losses, fair selection and actual process deadline."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid_lab import global_embedding_loss, cluster_driven_triplet_loss, update_memories
from khipu.select_and_test import select_experiments
from workstation_v3 import run_command, write_manifest
import workstation_v3


class ReviewChecks(unittest.TestCase):
    def test_gel_matches_explicit_class_aggregation(self):
        x = F.normalize(torch.tensor([[1., 2., 1.], [2., -1., 1.]]), dim=1).requires_grad_()
        bank = F.normalize(torch.tensor([[1., 1., 0.], [0., 1., 1.], [-1., 0., 1.]]), dim=1)
        memory_labels = torch.tensor([0, 0, 1])
        labels = torch.tensor([0, 1])
        responses = (x @ bank.T / 0.07).exp()
        scores = torch.stack([responses[:, :2].sum(1), responses[:, 2]], dim=1)
        expected = -(scores[torch.arange(2), labels] / scores.sum(1)).log().mean()
        actual = global_embedding_loss(x, labels, bank, memory_labels, 2)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_cdtl_singleton_anchors_and_gradients(self):
        torch.manual_seed(42)
        outputs = tuple(F.normalize(torch.randn(6, 8), dim=1).requires_grad_() for _ in range(3))
        centers = [F.normalize(torch.randn(3, 8), dim=1) for _ in range(3)]
        loss = cluster_driven_triplet_loss(outputs, torch.tensor([0, 0, 1, 1, 2, 3]), centers)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(z.grad is not None and torch.isfinite(z.grad).all() for z in outputs))
        zero = cluster_driven_triplet_loss(outputs, torch.zeros(6, dtype=torch.long), centers)
        self.assertEqual(zero.item(), 0.)

    def test_duplicate_indices_update_both_memories(self):
        bank = torch.tensor([[1., 0.], [0., 1.]])
        centers = [bank.clone() for _ in range(3)]
        values = torch.tensor([[0., 1.], [1., 0.], [-1., 0.]])
        outputs = (values, values, values)
        update_memories(outputs, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1]), bank, centers)
        expected = F.normalize(torch.tensor([[0.55, 0.45], [-0.9, 0.1]]), dim=1)
        torch.testing.assert_close(bank, expected)
        for center in centers:
            torch.testing.assert_close(center, expected)

    def test_selection_excludes_unmatched_later_rounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            for method, score in (("baseline", .2), ("cem_only", .3), ("full", .25)):
                target = runs / method
                target.mkdir()
                (target / "round_00.pt").touch()
                (target / "round_00_validation_metrics.json").write_text(json.dumps({"original": {"mAP": score}}))
            (runs / "baseline/round_01.pt").touch()
            (runs / "baseline/round_01_validation_metrics.json").write_text(json.dumps({"original": {"mAP": .9}}))
            best, rounds = select_experiments(runs, common_rounds=True)
            self.assertEqual(rounds, ["round_00"])
            self.assertEqual(max(best, key=lambda key: best[key]["score"]), "cem_only")
            self.assertEqual(best["baseline"]["score"], .2)

    def test_deadline_stops_a_real_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                run_command([sys.executable, "-c", "import time; time.sleep(60)"],
                            Path(temporary), Path(temporary) / "log.txt", start + .5)
            self.assertLess(time.monotonic() - start, 15)

    def test_deadline_also_stops_spawned_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = "import time; from pathlib import Path; time.sleep(2); Path('orphan.txt').write_text('unexpected')"
            parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"
            with self.assertRaises(TimeoutError):
                run_command([sys.executable, "-c", parent], root, root / "log.txt", time.monotonic() + .5)
            time.sleep(2.1)
            self.assertFalse((root / "orphan.txt").exists())

    def test_orchestrator_round_order_and_validation_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "results"
            order = []
            def fake_command(command, project, log_path, deadline):
                command = [str(x) for x in command]
                if "train" in command:
                    method = command[command.index("--method") + 1]
                    rnd = int(command[command.index("--rounds") + 1]) - 1
                    output = Path(command[command.index("--output") + 1])
                    output.mkdir(exist_ok=True)
                    (output / f"round_{rnd:02d}.pt").touch()
                    order.append((rnd, method))
                elif "evaluate" in command:
                    values = {"mAP": .2, "Rank-1": .3, "Rank-5": .4, "Rank-10": .5}
                    Path(command[command.index("--metrics-output") + 1]).write_text(json.dumps(
                        {"original": values, "low_resolution": values, "mAP_drop": 0}))
                elif "khipu/select_and_test.py" in command:
                    run_command(command, project, log_path, deadline)
            argv = ["workstation_v3.py", "--output", str(runs), "--rounds", "2", "--epochs-per-round", "1", "--validation-only"]
            with patch.object(sys, "argv", argv), patch.object(workstation_v3, "run_command", fake_command):
                workstation_v3.main()
            self.assertEqual(order, [(r, m) for r in range(2) for m in ("baseline", "cem_only", "full")])
            self.assertTrue((runs / "validation_report.md").exists())
            self.assertFalse((runs / "test_metrics.json").exists())
            self.assertEqual(json.loads((runs / "execution_status.json").read_text())["status"], "validation_complete")

    def test_manifest_prevents_mixing_configurations(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            project = Path(__file__).resolve().parents[1]
            args = argparse.Namespace(rounds=8, epochs_per_round=8, batch=64, extract_batch=256,
                                      workers=8, seed=42, data=project / "Datos preparados/Market-1501")
            write_manifest(project, runs, args)
            write_manifest(project, runs, args)
            args.batch = 32
            with self.assertRaises(RuntimeError):
                write_manifest(project, runs, args)


if __name__ == "__main__":
    unittest.main()
