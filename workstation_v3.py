"""Bounded, resumable workstation experiment; select only on validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

METHODS = ("baseline", "cem_only", "full")


class WindowsProcessJob:
    """Own only the worker process and its descendants, using Windows handles."""
    def __init__(self, process):
        import ctypes
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            error = ctypes.get_last_error()
            self.kernel.CloseHandle(self.handle)
            raise ctypes.WinError(error)

    def stop(self):
        import ctypes
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        self.kernel.CloseHandle(self.handle)


def stop_process_tree(process):
    job = getattr(process, "_lab_job", None)
    if job is not None:
        job.stop()
        process.wait(timeout=10)
        return
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.kill()
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=15)


def run_command(command, project, log_path, deadline):
    """Enforce the same wall-clock deadline even inside long training rounds."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Experiment time budget exhausted")
    print("Running: " + subprocess.list2cmdline([str(x) for x in command]), flush=True)
    environment = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONUNBUFFERED="1")
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen([str(x) for x in command], cwd=project, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", **options)
    if os.name == "nt":
        try:
            process._lab_job = WindowsProcessJob(process)
        except OSError:
            process.kill()
            process.wait(timeout=10)
            raise RuntimeError("Windows could not create the process group needed to enforce the time budget") from None
    with log_path.open("a", encoding="utf-8") as log:
        def forward_output():
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except UnicodeEncodeError:
                    sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
        reader = threading.Thread(target=forward_output, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            stop_process_tree(process)
            raise TimeoutError("Stopped process tree at experiment deadline") from None
        except BaseException:
            stop_process_tree(process)
            raise
        finally:
            if os.name == "nt":
                process._lab_job.stop()
                process._lab_job.close()
            reader.join(timeout=5)
            process.stdout.close()
    if code:
        raise RuntimeError(f"Command failed with exit code {code}; see {log_path}")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(project, runs, args):
    configuration = {name: getattr(args, name) for name in
                     ("rounds", "epochs_per_round", "batch", "extract_batch", "workers", "seed")}
    configuration.update(protocol="paper_adapted_v3", data=str(args.data.resolve()),
                         round_order="round_then_method", lr=3.5e-4, backbone_lr_multiplier=0.1,
                         lr_decay_every_epochs=20, lr_decay_factor=0.1,
                         ce_weights=[0.8, 0.2, 0.2], cdtl_weight=1.0, gel_weight=0.6,
                         source_sha256={str(path): file_sha256(project / path) for path in
                                        (Path("reid_lab.py"), Path("workstation_v3.py"), Path("khipu/select_and_test.py"))})
    target = runs / "experiment_config.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old != configuration:
            raise RuntimeError("Configuration or code changed since this run. Choose a new --output directory.")
    else:
        if any(runs.glob("*/round_*.pt")) or any(runs.glob("*/last.pt")):
            raise RuntimeError("Old/untracked v3 checkpoints exist. Use -Output runs_workstation_v3_reviewed.")
        target.write_text(json.dumps(configuration, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--epochs-per-round", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--extract-batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs_workstation_v3"))
    parser.add_argument("--data", type=Path, default=Path("Datos preparados/Market-1501"))
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.epochs_per_round, args.batch, args.extract_batch) < 1 or args.batch % 4 or args.workers < 0:
        parser.error("Positive rounds/epochs/batches required; batch divisible by 4; workers >= 0")
    if not 0 < args.max_hours <= 5:
        parser.error("--max-hours must be greater than zero and at most 5")
    started = time.monotonic()
    deadline = started + args.max_hours * 3600 - 30  # allow process cleanup/status writes
    reserve = min(15 * 60, args.max_hours * 3600 * 0.2)
    training_deadline = deadline - reserve
    project = Path(__file__).resolve().parent
    os.chdir(project)
    args.data = args.data.resolve()
    runs = args.output.resolve()
    runs.mkdir(parents=True, exist_ok=True)
    write_manifest(project, runs, args)
    log = runs / "execution.log"
    status = {"status": "running", "time_budget_hours_this_invocation": args.max_hours,
              "protocol": "paper_adapted_v3", "training_budget_reached": False}

    def save_status():
        status["elapsed_seconds_this_invocation"] = time.monotonic() - started
        (runs / "execution_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    def command(parts, end=deadline):
        run_command([sys.executable, *parts], project, log, end)

    save_status()
    try:
        test_record_path = runs / "test_evaluation_record.json"
        if test_record_path.exists() and json.loads(test_record_path.read_text())["status"] == "complete":
            print("Final test already recorded: experiment is frozen; reusing saved results.", flush=True)
            command(["khipu/select_and_test.py", "--runs", runs, "--common-rounds", "--once",
                     "--comparison-runs", project / "runs_workstation_v2",
                     "--testing", args.data / "testing.h5",
                     "--batch", str(args.extract_batch), "--workers", str(args.workers)])
            status["status"] = "complete"
            return
        # Opening testing here checks schema only, never reads labels or metrics.
        command(["-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__); print(torch.cuda.get_device_name(0))"], training_deadline)
        command(["reid_lab.py", "preflight", "--train", args.data / "training.h5",
                 "--validation", args.data / "validation.h5", "--testing", args.data / "testing.h5"], training_deadline)
        try:
            # Compare equally completed rounds if the budget ends mid-cycle.
            for rnd in range(args.rounds):
                for method in METHODS:
                    output = runs / method
                    checkpoint = output / f"round_{rnd:02d}.pt"
                    metrics = output / f"round_{rnd:02d}_validation_metrics.json"
                    if not checkpoint.exists():
                        command(["reid_lab.py", "train", "--train", args.data / "training.h5",
                                 "--method", method, "--protocol", "paper_adapted_v3", "--rounds", str(rnd + 1),
                                 "--epochs-per-round", str(args.epochs_per_round), "--batch", str(args.batch),
                                 "--extract-batch", str(args.extract_batch), "--workers", str(args.workers),
                                 "--seed", str(args.seed), "--output", output, "--resume"], training_deadline)
                    if not metrics.exists():
                        command(["reid_lab.py", "evaluate", "--data", args.data / "validation.h5",
                                 "--checkpoint", checkpoint, "--metrics-output", metrics,
                                 "--batch", str(args.extract_batch), "--workers", str(args.workers)], training_deadline)
                    print(f"Validated {method} round {rnd}; elapsed {(time.monotonic() - started) / 60:.1f} min", flush=True)
        except TimeoutError as error:
            status["training_budget_reached"] = True
            print(str(error) + "; using jointly completed validation rounds.", flush=True)

        from khipu.select_and_test import select_experiments
        try:
            _, common = select_experiments(runs, common_rounds=True)
        except RuntimeError as error:
            status.update(status="partial_no_comparable_round", detail=str(error))
            print("Saved progress; at least one completed round per method is needed for comparison.", flush=True)
            return
        status["eligible_common_rounds"] = common
        selection_args = ["khipu/select_and_test.py", "--runs", runs, "--common-rounds",
                          "--comparison-runs", project / "runs_workstation_v2",
                          "--batch", str(args.extract_batch), "--workers", str(args.workers)]
        command([*selection_args, "--validation-only"])
        if not args.validation_only:
            command([*selection_args, "--testing", args.data / "testing.h5", "--once"])
        status["status"] = "validation_complete" if args.validation_only else "complete"
        print(f"Results: {runs}. Review validation_report.md and execution_status.json.", flush=True)
    except TimeoutError as error:
        status.update(status="time_limit", detail=str(error))
        print("Five-hour budget enforced; completed checkpoints and metrics were retained.", flush=True)
    except KeyboardInterrupt:
        status["status"] = "interrupted"
        print("Interrupted; completed rounds can be resumed.", flush=True)
    except Exception as error:
        status.update(status="failed", detail=str(error))
        raise
    finally:
        save_status()


if __name__ == "__main__":
    main()
