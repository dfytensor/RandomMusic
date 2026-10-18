# -*- coding: utf-8 -*-
"""全库指标数据覆盖率审计：哪些歌缺哪些指标、为什么缺"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent

def load_json(p):
    p = Path(p)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}

fp = load_json(ROOT/"eval"/"lib_metric_cache.json")
wv = load_json(ROOT/"eval"/"wave_metric_cache.json")
rs = load_json(ROOT/"eval"/"resonance_cache.json")
ms = load_json(ROOT/"eval"/"melody_theory_cache.json")

print(f"fp_cache: {len(fp)} 首")
print(f"wv_cache: {len(wv)} 首")
print(f"rs: {len(rs 首)} 首")
print(f"ms: {len(ms)} 首")

# fp_cache 缺失原因分析
skip_count = sum(1 for v in fp.values() if v.get("skip"))
no_voiced = sum(1 for v in fp.values() if not v.get("skip") and v.get("voiced_ratio") is None)
ok_count = sum(1 for v in fp.values() if not v.get("skip") and v.get("voiced_ratio") is not None)
print(f"\nfp_cache 分析: ok={ok_count} skip={skip_count} no_voiced={no_voiced}")

# 找出所有 skip 的歌及原因
for k, v in fp.items():
    if v.get("skip"):
        print(f"  SKIP: {k[:30]}")

# 检查 library_wavs 和 library_transcripts 的对应
lib_wav = ROOT / "eval" / "library_wavs"
lib_tr = ROOT / "eval" / "library_transcripts"
wavs = {p.stem for p in lib_wav.glob("*.wav")}
trs = {p.stem for p in lib_tr.glob("*/score.abc")}
print(f"\nlibrary_wavs: {len(wavs)}")
print(f"library_transcripts: {len(trs 首)}")
print(f"wav 无 transcript: {len(wavs - trs 首)}")
print(f"transcript 无 wav: {len(trs - wavs)}")
missing = wavs - trs
if missing:
    for m in sorted(missing)[:5]:
        print(f"  missing: {m}")