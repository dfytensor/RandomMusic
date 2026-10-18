# -*- coding: utf-8 -*-
import os
from pathlib import Path

d = Path(r"F:\yinyue\novel_songs\final")
flacs = sorted(f for f in os.listdir(d) if f.endswith(".flac") and "_hum" not in f)
print(f"最终交付 {len(flacs)} 首:")
for f in flacs:
    mb = os.path.getsize(d / f) / 1048576
    print(f"  {f}  ({mb:.0f}MB)")
