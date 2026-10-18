# -*- coding: utf-8 -*-
"""盘点 final 目录：正式版 + 可哼版 + 状态"""
import os
from pathlib import Path

final = Path(r"F:\yinyue\novel_songs\final")
straight = sorted([f for f in os.listdir(final) if f.endswith(".flac") and "_hum" not in f])
hum = sorted([f for f in os.listdir(final) if f.endswith(".flac") and "_hum" in f])

print("=== 正式版（12 首）===")
for f in straight:
    print(f"  {f}")
print(f"\n=== 可哼 A/B 版（{len(hum)} 首）===")
for f in hum:
    print(f"  {f}")
print(f"\n可哼版缺失: ", end="")
all_no = [f"{i:02d}" for i in range(1, 13)]
hum_nos = [f[:2] for f in hum]
missing = [n for n in all_no if n not in hum_nos]
print(missing if missing else "无")
