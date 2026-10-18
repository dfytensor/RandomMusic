# -*- coding: utf-8 -*-
"""两阶段重扫 stage2: 每池 v3 top-N ∪ 现任冠军 → Demucs 分离 → v3.5 11维终评
口径: v3.5 单窗(副歌段), 与 stage1 rescan_v3 单窗可比; 新维全曲 stem 歌级恒定
断点续跑: render_stems/<pool>/<seed>/vocals.wav+beats.json (分离)
          + eval/render_art_cache.json (新维) + eval/rescan_v35_cache.json (分)
用法: python rescan_v35_stage2.py [N_TOP=5] [LIMIT=999]
"""
import json, re, sys, warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
N_TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 999
STAGE1_P = ROOT/"eval/rescan_v3_cache.json"
ART_CACHE_P = ROOT/"eval/render_art_cache.json"
S35_P = ROOT/"eval/rescan_v35_cache.json"
STEMS = ROOT/"render_stems"

CHAMPS = {  # 歌号 -> 池目录
    "01": "01_v2", "02": "02_wide", "03": "03_v2", "04": "04", "05": "05_wide",
    "06": "06_wide", "07": "07_adj", "08": "11", "09": "09", "10": "10",
    "11": "11", "12": "12",
}
CHAMP_FILE = {
    "01": "seed58888", "02": "seed7778", "03": "seed61001", "04": "seed90406",
    "05": "seed1111", "06": "seed3690", "07": "seed1357_r1", "08": "seed91104",
    "09": "seed90904", "10": "seed91007", "11": "seed91103", "12": "seed91202",
}

stage1 = json.loads(STAGE1_P.read_text(encoding="utf-8"))
pools = defaultdict(list)  # pool -> [(v3_score, rel)]
for rel, v in stage1.items():
    if v is None or not rel.startswith("novel_songs/batch_gacha/"):
        continue
    parts = rel.split("/")
    pool, fname = parts[2], parts[3]
    if not re.match(r"seed\d+.*\.flac$", fname):
        continue
    pools[pool].append((v["score"], rel))

cand = {}  # pool -> [rel]
for pool, lst in pools.items():
    lst.sort(key=lambda x: -x[0])
    picks = [rel for _, rel in lst[:N_TOP]]
    for no, pdir in CHAMPS.items():
        if pdir == pool:
            crel = f"novel_songs/batch_gacha/{pool}/{CHAMP_FILE[no]}.flac"
            if crel not in picks and (ROOT/crel).exists():
                picks.append(crel)
    if pool == "11":  # 08/11 双现任
        for s in ["seed91104", "seed91103"]:
            crel = f"novel_songs/batch_gacha/11/{s}.flac"
            if crel not in picks and (ROOT/crel).exists():
                picks.append(crel)
    cand[pool] = picks
total = sum(len(v) for v in cand.values())
print(f"pools={len(cand)} candidates={total} (N_TOP={N_TOP})", flush=True)

s35 = json.loads(S35_P.read_text(encoding="utf-8")) if S35_P.exists() else {}
art_cache = json.loads(ART_CACHE_P.read_text(encoding="utf-8")) if ART_CACHE_P.exists() else {}
pending = [(pool, rel) for pool in sorted(cand) for rel in cand[pool] if rel not in s35]
print(f"pending={min(len(pending), LIMIT)}/{len(pending)}", flush=True)

if pending:
    sys.path.insert(0, str(ROOT))
    from art_core import art_dims
    from score_v3 import measure, score_of
    from demucs.api import Separator
    import torch, soundfile as sf, librosa, numpy as np

    sep = Separator(model="htdemucs", device="cuda")
    todo = pending[:LIMIT]
    for i, (pool, rel) in enumerate(todo, 1):
        p = ROOT/rel
        sd = STEMS/pool/p.stem
        vpath, bpath = sd/"vocals.wav", sd/"beats.json"
        print(f"[{i}/{len(todo)}] {rel}", flush=True)
        try:
            if not (vpath.exists() and bpath.exists()):
                sd.mkdir(parents=True, exist_ok=True)
                origin, stems = sep.separate_audio_file(str(p))
                vocals = stems["vocals"]; accomp = origin - vocals
                sf.write(str(vpath), vocals.cpu().numpy().T, sep.samplerate)
                acc_np = accomp.cpu().numpy().mean(axis=0)
                ya = librosa.resample(acc_np, orig_sr=sep.samplerate, target_sr=22050)
                tempo, beats = librosa.beat.beat_track(y=ya, sr=22050, hop_length=512)
                bt = librosa.frames_to_time(beats, sr=22050, hop_length=512)
                bpath.write_text(json.dumps({
                    "tempo": float(np.atleast_1d(tempo)[0]),
                    "beat_times": [round(float(x), 4) for x in bt]}), encoding="utf-8")
            if rel not in art_cache or art_cache[rel] is None:
                art_cache[rel] = art_dims(vpath, bpath)
                ART_CACHE_P.write_text(json.dumps(art_cache, ensure_ascii=False), encoding="utf-8")
            m = measure(p, vocals_path=vpath, beats_path=bpath)
            s35[rel] = score_of(m)
        except Exception as e:
            print(f"  ERR {type(e).__name__}: {e}", flush=True)
            s35[rel] = None
        if i % 5 == 0:
            S35_P.write_text(json.dumps(s35, ensure_ascii=False), encoding="utf-8")
    S35_P.write_text(json.dumps(s35, ensure_ascii=False), encoding="utf-8")
    print(f"RUN_DONE {len(todo)} processed, cache={len(s35)}", flush=True)

# ===== 报告 =====
missing = [(pool, rel) for pool in sorted(cand) for rel in cand[pool] if rel not in s35]
if missing:
    print(f"\n未完成 {len(missing)} 个, 先跑报告会缺行; 补齐后重跑本脚本出全表", flush=True)

lines = ["# 全库存量 v3.5 两阶段重扫: 每池终评重选冠军 (2026-10-16)", "",
         "阶段1: v3 6维单窗筛 top-N (rescan_v3_cache.json) | 阶段2: Demucs 人声 → v3.5 11维单窗终评",
         "口径: 单窗(副歌段) 与 stage1 可比; 新维=全曲 stem 歌级恒定; n<11 维时自动跳过缺失维", "",
         f"| 池 | 现任 | 现任v3 | 现任v3.5 | v3.5重选 | Δ | 新冠军(top-N内) |",
         "|---|---|---|---|---|---|---|"]
print(f"\n{'池':10s} {'现任':>10s} {'v3':>6s} {'v3.5现任':>8s} {'v3.5重选':>8s} {'Δ':>7s}  新冠军", flush=True)
rows = []
for pool in sorted(cand):
    scored = [(s35[rel]["score"], stage1[rel]["score"], rel)
              for rel in cand[pool] if s35.get(rel) and stage1.get(rel)]
    if not scored:
        continue
    scored.sort(key=lambda x: -x[0])
    best35, _, best_rel = scored[0]
    # 现任(该池; 池11 取两位现任中 v3.5 较高者代表)
    champ_rels = [f"novel_songs/batch_gacha/{pool}/{CHAMP_FILE[no]}.flac"
                  for no, pd in CHAMPS.items() if pd == pool]
    if pool == "11":
        champ_rels = [f"novel_songs/batch_gacha/11/{s}.flac" for s in ["seed91104", "seed91103"]]
    champ35 = [(s35[r]["score"], r) for r in champ_rels if s35.get(r)]
    if not champ35:
        continue
    cur35, cur_rel = max(champ35)
    cur31 = (stage1.get(cur_rel) or {}).get("score")
    delta = best35 - cur35
    best_seed = Path(best_rel).stem
    cur_seed = Path(cur_rel).stem
    mark = "★" if best_seed != cur_seed else "="
    print(f"{pool:10s} {cur_seed:>10s} {cur31 if cur31 else '?':>6} {cur35:>8.1f} {best35:>8.1f} {delta:>+7.1f}  {mark} {best_seed}", flush=True)
    detail = ", ".join(f"{Path(r).stem.replace('seed','')}={s35[r]['score']:.0f}" for s, _, r in scored)
    lines.append(f"| {pool} | {cur_seed} | {cur31 if cur31 else 'NA'} | {cur35:.1f} | {best35:.1f} | {delta:+.1f} | "
                 f"{'**'+best_seed+'**' if mark=='★' else '不变'} |")
    rows.append((pool, cur_seed, cur35, best35, best_seed, delta))
    print(f"           top: {detail}", flush=True)

if rows:
    cur = [r[2] for r in rows]; new = [r[3] for r in rows]
    msg = (f"可比对 {len(rows)} 池: 现任 v3.5 均值 {np.mean(cur):.1f} → 重选 {np.mean(new):.1f} "
           f"(Δ {np.mean(new)-np.mean(cur):+.1f})")
    print(f"\n{msg}", flush=True)
    lines += ["", msg, "",
              "注: 分数为 v3.5 单窗口径, 与 v3_final 冠军 5 窗中位分不可直接比; 排位只在同口径内有效。", ""]
(ROOT/"eval/v35_rescan_report.md").write_text("\n".join(lines), encoding="utf-8")
print("SAVED eval/v35_rescan_report.md", flush=True)
