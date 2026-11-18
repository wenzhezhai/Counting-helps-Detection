#!/usr/bin/env python3
"""Export official DINOv3 + Plain-DETR native candidates without evaluation labels."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from counting_help_detection.dino import main
if __name__ == "__main__":
    main()
