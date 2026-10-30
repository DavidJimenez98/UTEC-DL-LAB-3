"""Exercise revised full training and resumption on 80 real unlabeled images."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np

project = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="lab3_v3_smoke_") as temporary:
    temporary = Path(temporary)
    data = temporary / "unlabeled.h5"
    with h5py.File(project / "Datos preparados/Market-1501/training.h5", "r") as original, h5py.File(data, "w") as target:
        target.create_dataset("images", data=original["images"][:80])
        target.create_dataset("camera_ids", data=original["camera_ids"][:80])
        target.create_dataset("image_ids", data=np.arange(80))
    output = temporary / "full"
    for rounds in (1, 2):
        subprocess.run([sys.executable, str(project / "reid_lab.py"), "train", "--train", str(data),
                        "--method", "full", "--protocol", "paper_adapted_v3", "--rounds", str(rounds),
                        "--epochs-per-round", "1", "--batch", "8", "--extract-batch", "16",
                        "--workers", "2", "--output", str(output), "--resume"], cwd=project, check=True, timeout=180)
        summary = json.loads((output / "training_summary.json").read_text())
        assert summary["round"] == rounds - 1
        assert summary["protocol"] == "paper_adapted_v3"
        assert np.isfinite(summary["last_epoch_loss"])
        assert summary["clusters"] >= 2
    print("REAL_DATA_FULL_TRAIN_AND_RESUME_OK")
