# -*- coding: utf-8 -*-
"""music_map.md 章节重排：11 矩阵块移到 10.6 之后"""
from pathlib import Path

p = Path(r"F:\yinyue\music_map.md")
lines = p.read_text(encoding="utf-8").splitlines()

i_11 = next(i for i, l in enumerate(lines) if l.startswith("## 11."))
i_105 = next(i for i, l in enumerate(lines) if l.startswith("### 10.5 "))
i_12 = next(i for i, l in enumerate(lines) if l.startswith("## 12."))

block = lines[i_11:i_105]
rest = lines[:i_11] + lines[i_105:]
j_12 = next(i for i, l in enumerate(rest) if l.startswith("## 12."))
out = rest[:j_12] + block + [""] + rest[j_12:]
p.write_text("\n".join(out) + "\n", encoding="utf-8")
print("reordered OK")
