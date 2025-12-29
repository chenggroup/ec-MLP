# SPDX-License-Identifier: LGPL-3.0-or-later
import os
from pathlib import Path


def clean_training_files(work_dir: str):
    root_dir = Path(__file__).parent
    os.chdir(work_dir)
    for f in os.listdir("."):
        try:
            if f.startswith("model.ckpt"):
                os.remove(f)
            elif f.endswith(".pb"):
                os.remove(f)
            elif f in [
                "lcurve.out",
                "checkpoint",
                "input_v2_compat.json",
                "out.json",
            ]:
                os.remove(f)
        except OSError:
            # Ignore failures during cleanup to allow remaining files to be processed
            pass

    os.chdir(root_dir)
