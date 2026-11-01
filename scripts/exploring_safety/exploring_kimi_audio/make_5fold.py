#!/usr/bin/env python3
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from make_5fold_common import run

run("kimi_audio")
