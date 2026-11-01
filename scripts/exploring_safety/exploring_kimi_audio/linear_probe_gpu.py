#!/usr/bin/env python3
import sys
from pathlib import Path
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from linear_probe_gpu_common import run
run("kimi_audio")
