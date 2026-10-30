"""Select the best experiment using validation mAP and evaluate it once on test."""
import argparse
import json
import subprocess
import sys
import math
import hashlib
from pathlib import Path


def render_report(runs, best_by_method, selected_method):
    test = json.loads((runs / "test_metrics.json").read_text())
    examples = json.loads((runs / "test_metrics_retrieval_examples.json").read_text())
    metrics_by_method = {method: candidate["metrics"] for method, candidate in best_by_method.items()}
    lines = [
        "# Laboratorio 3: resultados de reidentificacion no supervisada",
        "",
        "## Metodologia y adaptacion",
        "",
        "ResNet-50 preentrenada en ImageNet extrae vectores globales, superiores e inferiores. "
        "Las pseudoetiquetas se generan sin identidades reales; si DBSCAN crea un grupo dominante, "
        "se usa K-means esferico y se reservan asignaciones inciertas como outliers. "
        "MLCOR recupera hasta 40 outliers por ronda mediante vecinos reciprocos y confianza. "
        "CEM combina una memoria de instancias para GEL con restricciones globales/locales C-DTL.",
        "",
        "Adaptaciones respecto al paper: fusion de distancias coseno en vez de Jaccard, "
        "respaldo K-means ante colapso de DBSCAN y presupuesto de entrenamiento propio. "
        "El conjunto de prueba no participa en entrenamiento ni seleccion.",
        "",
        "## Comparacion y ablacion en validacion",
        "",
        "| Metodo | Ronda | mAP original | Rank-1 | Rank-5 | Rank-10 | mAP baja resolucion | Rank-1 baja | Rank-5 baja | Rank-10 baja | Caida mAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    config_path = runs / "experiment_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        lines[2:2] = ["## Configuracion registrada", "",
                      f"Protocolo {config['protocol']}; hasta {config['rounds']} rondas de "
                      f"{config['epochs_per_round']} epocas por metodo, batch {config['batch']}. "
                      "Las rondas se numeran desde cero. El limite de tiempo puede reducir las rondas completas; "
                      "se comparan solo las rondas terminadas y validadas por los tres metodos. "
                      "La configuracion exacta y huellas del codigo estan en experiment_config.json.", ""]
    for method in ("baseline", "cem_only", "full"):
        m = metrics_by_method[method]
        orig, low = m["original"], m["low_resolution"]
        lines.append(f"| {method} | {best_by_method[method]['round']} | {orig['mAP']:.4f} | {orig['Rank-1']:.4f} | "
                     f"{orig['Rank-5']:.4f} | {orig['Rank-10']:.4f} | "
                     f"{low['mAP']:.4f} | {low['Rank-1']:.4f} | {low['Rank-5']:.4f} | "
                     f"{low['Rank-10']:.4f} | {m['mAP_drop']:.4f} |")
    baseline = metrics_by_method["baseline"]["original"]["mAP"]
    cem = metrics_by_method["cem_only"]["original"]["mAP"]
    full = metrics_by_method["full"]["original"]["mAP"]
    lines += [
        "",
        f"Efecto de CEM frente a baseline: {cem - baseline:+.4f} mAP. "
        f"Efecto adicional de MLCOR frente a CEM: {full - cem:+.4f} mAP. "
        "Son diferencias descriptivas de una semilla y de 100 consultas de validacion; "
        "no prueban significancia estadistica ni identifican la causa de una caida. "
        "Son resultados de esta adaptacion, no cifras publicadas del paper.",
        "",
        f"Se selecciono **{selected_method}** por mayor mAP original en validacion.",
        "",
        "## Evaluacion final en testing",
        "",
        "| Condicion | mAP | Rank-1 | Rank-5 | Rank-10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in ("original", "low_resolution"):
        m = test[condition]
        lines.append(f"| {condition} | {m['mAP']:.4f} | {m['Rank-1']:.4f} | "
                     f"{m['Rank-5']:.4f} | {m['Rank-10']:.4f} |")
    lines += ["", f"Caida de mAP: **{test['mAP_drop']:.4f}**.", "",
              "## Cinco aciertos y cinco fallos", "",
              "![Consultas y top 10 recuperados](test_metrics_retrieval_examples.png)", ""]
    for title, correct in (("Aciertos", True), ("Fallos", False)):
        lines += [f"### {title}", ""]
        selected = [item for item in examples if item["top1_correct"] == correct][:5]
        for item in selected:
            query = item["query_index"]
            if correct:
                explanation = (f"la primera imagen recuperada tiene la misma identidad "
                               f"({item['query_person_id']}) desde la camara "
                               f"{item['top1_gallery_camera_id']}.")
            else:
                explanation = (f"el primer resultado es otra identidad "
                               f"({item['top1_gallery_person_id']}); la identidad correcta "
                               f"aparece por primera vez en el rango {item['first_correct_rank']}.")
            lines.append(f"- Consulta {query}, camara {item['query_camera_id']}: {explanation}")
        if len(selected) < 5:
            lines.append(f"- Solo se encontraron {len(selected)} casos de este tipo; no se inventaron ejemplos.")
        lines.append("")
    lines += [
        "## Limitaciones",
        "",
        "Las pseudoetiquetas pueden mezclar identidades visualmente parecidas. "
        "La particion K-means es un respaldo cuando DBSCAN se vuelve degenerado, "
        "y puede crear grupos artificiales. La baja resolucion elimina detalles utiles "
        "para distinguir ropa y siluetas; la magnitud observada del efecto se muestra arriba. "
        "La figura debe inspeccionarse visualmente para describir causas como ropa, oclusion o postura "
        "en la presentacion oral; esas causas no se infieren de los IDs.",
        "",
    ]
    (runs / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def select_experiments(runs, common_rounds=False):
    candidates = {}
    for method in ("baseline", "cem_only", "full"):
        method_dir = runs / method
        metric_paths = sorted(method_dir.glob("round_*_validation_metrics.json"))
        if not metric_paths:
            metric_paths = [method_dir / "validation_metrics.json"]
        candidates[method] = {}
        for metrics_path in metric_paths:
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text())
            round_name = metrics_path.name.removesuffix("_validation_metrics.json")
            checkpoint = method_dir / (round_name + ".pt") if round_name.startswith("round_") else method_dir / "last.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            score = metrics["original"]["mAP"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid validation mAP in {metrics_path}")
            candidates[method][round_name] = {"method": method, "round": round_name, "metrics": metrics,
                                             "checkpoint": checkpoint, "score": score}
    if any(not entries for entries in candidates.values()):
        raise RuntimeError("Missing validation metrics for one or more methods.")
    eligible = set.intersection(*(set(entries) for entries in candidates.values())) if common_rounds else None
    if common_rounds and not eligible:
        raise RuntimeError("No jointly completed validation round for baseline, cem_only and full.")
    best_by_method = {
        method: max((item for name, item in entries.items() if eligible is None or name in eligible),
                    key=lambda item: item["score"])
        for method, entries in candidates.items()
    }
    return best_by_method, sorted(eligible) if eligible is not None else None


def render_validation_report(runs, best_by_method, common, comparison_runs=None):
    lines = ["# Validacion del experimento", "",
             "Seleccion exclusivamente por mAP original de validacion. Rondas elegibles: " + str(common), "",
             "| Metodo | Ronda (indice desde 0) | mAP | Rank-1 | Rank-5 | Rank-10 | mAP baja | Rank-1 baja | Rank-5 baja | Rank-10 baja | Caida mAP |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, item in best_by_method.items():
        orig, low = item["metrics"]["original"], item["metrics"]["low_resolution"]
        vals = [orig[k] for k in ("mAP", "Rank-1", "Rank-5", "Rank-10")]
        vals += [low[k] for k in ("mAP", "Rank-1", "Rank-5", "Rank-10")]
        vals += [item["metrics"]["mAP_drop"]]
        lines.append(f"| {method} | {item['round']} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    config_path = runs / "experiment_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        lines += ["", f"Configuracion: hasta {config['rounds']} rondas x {config['epochs_per_round']} epocas, "
                  f"batch {config['batch']}, protocolo {config['protocol']}. "
                  "Los resumenes por ronda contienen pasos de optimizacion, perdida, tiempo y memoria GPU."]
    if comparison_runs is not None and comparison_runs.exists():
        old, _ = select_experiments(comparison_runs)
        lines += ["", "## Comparacion exploratoria con v2", "",
                  "Cambian batch, duracion y detalles de optimizacion; las diferencias no se atribuyen solo a mas epocas.", "",
                  "| Metodo | mAP val v2 | mAP val actual | Diferencia |", "|---|---:|---:|---:|"]
        for method, item in best_by_method.items():
            before, after = old[method]["score"], item["score"]
            lines.append(f"| {method} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    lines += ["", "Una semilla y 100 consultas: diferencias descriptivas, sin prueba de significancia.",
              "No se ha usado testing para elegir configuracion. La revision visual de errores se realiza tras la evaluacion final.", ""]
    (runs / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--testing")
    parser.add_argument("--batch", default="128")
    parser.add_argument("--workers", default="4")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--common-rounds", action="store_true")
    parser.add_argument("--comparison-runs", type=Path)
    parser.add_argument("--once", action="store_true", help="Protect a completed final test from being repeated or replaced")
    args = parser.parse_args()
    runs = Path(args.runs)
    project = Path(__file__).resolve().parents[1]
    best_by_method, eligible = select_experiments(runs, args.common_rounds)
    selected_record = max(best_by_method.values(), key=lambda item: item["score"])
    method = selected_record["method"]
    selected = selected_record["checkpoint"]
    record = {"selected_method": method, "selected_round": selected_record["round"],
              "validation_original_mAP": selected_record["score"], "validation": selected_record["metrics"],
              "eligible_common_rounds": eligible, "checkpoint": str(selected)}
    (runs / "selection.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    render_validation_report(runs, best_by_method, eligible, args.comparison_runs)
    if args.validation_only:
        return
    if not args.testing:
        parser.error("--testing is required for final evaluation")
    test_record_path = runs / "test_evaluation_record.json"
    if args.once:
        digest = hashlib.sha256()
        with selected.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        identity = {"method": method, "round": selected_record["round"], "checkpoint_sha256": digest.hexdigest()}
        if test_record_path.exists():
            previous = json.loads(test_record_path.read_text())
            if previous["identity"] != identity:
                raise RuntimeError("A different checkpoint was already tested in this experiment. Do not tune on testing.")
            if previous["status"] == "complete":
                render_report(runs, best_by_method, method)
                print("Final test already complete; reusing recorded results.")
                return
        elif (runs / "test_metrics.json").exists():
            raise RuntimeError("Untracked test results already exist; refusing to overwrite them.")
        test_record_path.write_text(json.dumps({"identity": identity, "status": "started"}, indent=2))
    subprocess.run([
        sys.executable, str(project / "reid_lab.py"), "evaluate", "--data", args.testing,
        "--checkpoint", str(selected), "--batch", args.batch, "--workers", args.workers,
        "--metrics-output", str(runs / "test_metrics.json"),
    ], check=True)
    if args.once:
        test_record_path.write_text(json.dumps({"identity": identity, "status": "complete"}, indent=2))
    render_report(runs, best_by_method, method)


if __name__ == "__main__":
    main()
