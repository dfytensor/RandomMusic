# -*- coding: utf-8 -*-
import json
import subprocess
import sys
from pathlib import Path

FINAL = Path(r"F:\yinyue\novel_songs\final")
ROOT = Path(r"F:\yinyue")
lines = ["# 全专辑抽卡选优结果", "",
         "| # | 曲目 | 现役AI味 | 新种最优AI味 | 胜者 | 替换? |", "|---|---|---|---|---|---|"]
replaced = improved = 0
for rep in sorted(FINAL.glob("*_gacha_report.json")):
    no = rep.name[:2]
    cands = json.loads(rep.read_text(encoding="utf-8"))
    inc = next((c for c in cands if c["tag"] == "incumbent"), None)
    news = [c for c in cands if c["tag"] != "incumbent" and c["ai"] is not None]
    if not inc or not news:
        continue
    best = min(news, key=lambda c: c["ai"])
    chosen_tag = "incumbent" if inc["ai"] <= best["ai"] else best["tag"]
    is_new_win = chosen_tag != "incumbent"
    replaced += is_new_win
    improved += (best["ai"] < inc["ai"])
    lines.append(f"| {no} | {rep.name[3:-18]} | {inc['ai']} | {best['ai']} (seed{best.get('seed')}) "
                 f"| {chosen_tag} | {'✅替换' if is_new_win else '保留'} |")

lines += ["", f"替换 {replaced} 首 · 有更优种子的 {improved} 首"]
# 技术终检
import subprocess
files = [str(p) for p in FINAL.glob("*.flac")]
subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe") if False else sys.executable,
                str(ROOT / "reference_free_check.py"), *files, "--out", str(FINAL / "tech_report.md")],
               check=True, capture_output=True)
txt = (FINAL / "tech_report.md").read_text(encoding="utf-8")
data = json.loads(txt.split("json")[1].split("```")[0])
ok = sum(1 for r in data if r["pass"] == r["total"])
lines.append(f"技术终检: {ok}/{len(data)} 全过")
print("\n".join(lines))
Path(FINAL / "gacha_summary.md").write_text("\n".join(lines), encoding="utf-8")
