# -*- coding: utf-8 -*-
"""全专辑抽卡驱动：02-12 每首 3 新种子，断点续跑（final/<no>_gacha_report.json 存在即跳过）"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
FINAL = ROOT / "novel_songs" / "final"

songs = [s["no"] for s in []] or [f"{i:02d}" for i in range(2, 13)]
todo = []
for no in songs:
    rep = FINAL / f"{no}_gacha_report.json"
    if not rep.exists():
        todo.append(no)
print("todo:", todo, flush=True)

for no in todo:
    base = 50000 + int(no) * 13
    seeds = ",".join(str(base + k) for k in (1, 2, 3))
    print(f"\n=== gacha {no} seeds={seeds} ===", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "gacha_select.py"), no, "--seeds", seeds],
                       capture_output=True, text=True)
    ok = "WINNER" in r.stdout
    print(f"{'OK' if ok else 'PARTIAL/FAIL'}: {no}", flush=True)
    if not ok:
        err = (r.stderr or "")[-300:]
        print("  stderr tail:", err, flush=True)
print("ALL DONE", flush=True)
