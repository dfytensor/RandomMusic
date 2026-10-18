# -*- coding: utf-8 -*-
import json
from pathlib import Path

txt = Path(r"F:\yinyue\eval\full_validation_matrix.md").read_text(encoding="utf-8")
print(txt[:4200])
