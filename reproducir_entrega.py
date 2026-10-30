"""Revisión y reproducción portable de la entrega académica V3."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("accion", choices=("reporte", "preflight", "evaluar", "entrenar"), nargs="?", default="reporte")
    p.add_argument("--data", type=Path, default=ROOT / "Datos preparados" / "Market-1501")
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args()
    if a.accion == "reporte":
        metrics = ROOT / "resultados" / "test_metrics.json"
        if not metrics.exists():
            metrics = ROOT / "runs_workstation_v3" / "test_metrics.json"
        print(json.dumps(json.loads(metrics.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
        return
    data = a.data.resolve()
    if a.accion == "preflight":
        args = ["reid_lab.py", "preflight", "--train", str(data / "training.h5"), "--validation", str(data / "validation.h5"), "--testing", str(data / "testing.h5")]
    elif a.accion == "evaluar":
        checkpoint = ROOT / "modelo" / "cem_only_round_04.pt"
        if not checkpoint.is_file():
            checkpoint = ROOT / "runs_workstation_v3" / "cem_only" / "round_04.pt"
        out = (a.output or ROOT / "reproduccion_test" / "test_metrics.json").resolve()
        if out.exists():
            p.error("El archivo de salida ya existe. Use --output con otra ruta para preservar los resultados.")
        args = ["reid_lab.py", "evaluate", "--data", str(data / "testing.h5"), "--checkpoint", str(checkpoint), "--metrics-output", str(out), "--batch", "256", "--workers", str(a.workers)]
    else:
        out = (a.output or ROOT / "runs_reproduccion").resolve()
        args = ["workstation_v3.py", "--data", str(data), "--output", str(out), "--workers", str(a.workers), "--validation-only"]
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
