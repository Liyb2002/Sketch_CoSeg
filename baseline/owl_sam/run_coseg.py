#!/usr/bin/env python3
# run_coseg.py — outside entrypoint for rank-only, binary grouping.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coseg.train import run  # rank-only version (no entropy_w / mass_w)

def main():
    OUTPUTS_DIR    = "outputs"
    SAVE_DIR       = "outputs"
    LABEL          = "wing"     # <-- your label
    K              = 2          # binary: belongs vs not
    D              = 256
    EPOCHS         = 400
    LR             = 3e-4
    DEVICE         = "auto"     # 'auto'|'cuda'|'cpu'
    DROPOUT        = 0.10
    LAMBDA_BETWEEN = 0.20       # separation term in rank loss

    print(f"[run_coseg] starting: label='{LABEL}', K={K}, epochs={EPOCHS}, lr={LR}")
    run(
        outputs_dir=OUTPUTS_DIR,
        label=LABEL,
        K=K,
        d=D,
        epochs=EPOCHS,
        lr=LR,
        device=DEVICE,
        save_dir=SAVE_DIR,
        dropout=DROPOUT,
        lambda_between=LAMBDA_BETWEEN,
    )
    print("[run_coseg] done.")

if __name__ == "__main__":
    main()
