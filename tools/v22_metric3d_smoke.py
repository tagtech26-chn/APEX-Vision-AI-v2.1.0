from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
METRIC3D_ROOT = ROOT / "third_party" / "Metric3D"
HUBCONF = METRIC3D_ROOT / "hubconf.py"


def main() -> None:
    print(f"Metric3D checkout: {METRIC3D_ROOT}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if not HUBCONF.exists():
        raise SystemExit(f"Missing Metric3D hubconf.py: {HUBCONF}")

    spec = importlib.util.spec_from_file_location("metric3d_hubconf", HUBCONF)
    if spec is None or spec.loader is None:
        raise SystemExit("Unable to load Metric3D hubconf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("Metric3D hubconf imported")
    print("ViT-Small factory:", hasattr(module, "metric3d_vit_small"))


if __name__ == "__main__":
    main()
