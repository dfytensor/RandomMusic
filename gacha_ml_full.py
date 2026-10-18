# -*- coding: utf-8 -*-
"""完整输出版"""
import json, sys, warnings, re
from pathlib import Path
import numpy as np
from collections import Counter
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# 重跑完整逻辑（同 gacha_ml_v2 但带输出）
exec(open(Path(__file__).parent / "gacha_ml_v2.py").read().replace(
    "if __name__", "if False and __name__"
))
