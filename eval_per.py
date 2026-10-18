# -*- coding: utf-8 -*-
"""PER(字错率) 评估 (run in .venv-asr): Qwen3-ASR 转写副歌片段 vs 意向歌词。
输出 eval/per_report.md
"""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

ROOT = Path(__file__).parent

model = Qwen3ASRModel.from_pretrained(
    str(ROOT / "models" / "Qwen3-ASR-1.7B"),
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
print("ASR loaded", flush=True)

refs = json.loads((ROOT / "eval" / "clips" / "refs.json").read_text(encoding="utf-8"))


def norm(t):
    t = re.sub(r"[^\w\u4e00-\u9fff]", "", t).lower()
    return t


rows = []
for name, ref in refs.items():
    clip = ROOT / "eval" / "clips" / f"chorus_{name}.wav"
    if not clip.exists() or not ref.strip():
        continue
    results = model.transcribe(audio=str(clip), language="Chinese")
    text = results[0].text
    r, h = norm(ref), norm(text)
    ratio = SequenceMatcher(None, r, h).ratio()
    per = round((1 - ratio) * 100, 1)
    rows.append(dict(name=name, per=per, ref_len=len(r), hyp=text.strip(), ref=ref))
    print(f"{name}: PER={per}% hyp={text.strip()[:40]}", flush=True)

lines = ["# 副歌咬字评估（Qwen3-ASR，PER=1-相似度）", "",
         "| 歌曲 | 副歌 PER | ASR 转写 |", "|---|---|---|"]
for r in rows:
    grade = "✅" if r["per"] <= 25 else ("🟡" if r["per"] <= 40 else "❌")
    lines.append(f"| {r['name']} | {r['per']}% {grade} | {r['hyp'][:60]} |")
lines += ["", "> 参考基准：YuE2 官方 benchmark 好歌 PER ≈ 7-10%（4 次 ASR 取最小），本简化口径（单次+合唱混响）阈值放宽。"]
report = "\n".join(lines)
(ROOT / "eval" / "per_report.md").write_text(report, encoding="utf-8")
print(report)
