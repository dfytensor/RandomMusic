# -*- coding: utf-8 -*-
"""03 池涌现一波: 8 新 seed (4 随机 + 4 邻域), v3 粗筛 → Demucs → v3.5 11维终评
目标: 测 v3.5 单窗口径能否 >80 (现任 seed61000=80.0, rescan_v35_cache.json)
随机: 61010/61013/61016/61019 | 邻域: 61003-61006 (seed 局部性爬坡)
断点续跑: novel_songs/batch_gacha/03_v35/ + eval/gacha03_v35_cache.json
          + render_stems/03_v35/ + eval/render_art_cache.json
用法: python gacha_v35_03.py [PHASE] phase=render|score|report (默认全跑)
"""
import json, subprocess, sys, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
PHASE = sys.argv[1] if len(sys.argv) > 1 else "all"

SEEDS_NEW = [61010, 61013, 61016, 61019, 61003, 61004, 61005, 61006]
LOTTERY = {61010, 61013, 61016, 61019}
OUT = ROOT / "novel_songs" / "batch_gacha" / "03_v35"
OUT.mkdir(exist_ok=True)
V3C = ROOT / "eval" / "gacha03_v35_cache.json"
S35C = ROOT / "eval" / "gacha03_v35_stage2.json"
STEMS = ROOT / "render_stems" / "03_v35"

def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
def save(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

# ===== 阶段A: 渲染 =====
if PHASE in ("all", "render"):
    todo = [s for s in SEEDS_NEW
            if not (OUT / f"seed{s}.flac").exists() or (OUT / f"seed{s}.flac").stat().st_size < 10_000_000]
    print(f"render todo={len(todo)}: {todo}", flush=True)
    def render(seed: int):
        dst = OUT / f"seed{seed}.flac"
        t0 = time.time()
        import os
        env = dict(os.environ, PYTHONUTF8="1")
        # 子进程 stdout/stderr 重定向到文件, 避免管道缓冲死锁
        logp = ROOT / "eval" / f"render_03_{seed}.log"
        with open(logp, "w", encoding="utf-8", errors="replace") as lf:
            r = subprocess.run([sys.executable, "render_one.py", "03", str(seed), str(dst)],
                               cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 10_000_000
        err = "" if ok else logp.read_text(encoding="utf-8", errors="replace")[-300:]
        return seed, ok, time.time() - t0, err
    # 串行渲染: 并行 2 路在本机卡死(原因未明), 串行稳定
    for s in todo:
        print(f"  rendering seed{s} ...", flush=True)
        seed, ok, dt, err = render(s)
        print(f"  seed{s}: {'OK' if ok else 'FAIL'} {dt/60:.1f}min {err[:150]}", flush=True)
    # 失败重试单路
    for s in todo:
        p = OUT / f"seed{s}.flac"
        if not p.exists() or p.stat().st_size < 10_000_000:
            print(f"  retry seed{s} ...", flush=True)
            _, ok, dt, err = render(s)
            print(f"  seed{s}: {'OK' if ok else 'FAIL still'} {err[:150]}", flush=True)

# ===== 阶段B: v3 粗筛 + v3.5 终评 =====
if PHASE in ("all", "score"):
    from art_core import art_dims
    from score_v3 import measure, score_of
    v3c, s35c = load(V3C), load(S35C)
    for s in SEEDS_NEW:
        p = OUT / f"seed{s}.flac"
        key = f"seed{s}"
        if not p.exists():
            print(f"  seed{s}: 缺渲染", flush=True); continue
        if key not in v3c or v3c[key] is None:
            m = measure(p)
            v3c[key] = score_of(m) if m else None
            save(V3C, v3c)
            print(f"  seed{s}: v3={v3c[key]['score'] if v3c[key] else None:.1f} [{v3c[key]['cat'] if v3c[key] else '?'}]", flush=True)
        if key in s35c and s35c[key] is not None:
            continue
        # stage2: Demucs → v3.5
        try:
            from demucs.api import Separator
            import torch, soundfile as sf, librosa, numpy as np
            sd = STEMS / key
            vpath, bpath = sd / "vocals.wav", sd / "beats.json"
            if not (vpath.exists() and bpath.exists()):
                sd.mkdir(parents=True, exist_ok=True)
                sep = Separator(model="htdemucs", device="cuda")
                origin, stems = sep.separate_audio_file(str(p))
                vocals = stems["vocals"]; accomp = origin - vocals
                sf.write(str(vpath), vocals.cpu().numpy().T, sep.samplerate)
                acc_np = accomp.cpu().numpy().mean(axis=0)
                ya = librosa.resample(acc_np, orig_sr=sep.samplerate, target_sr=22050)
                tempo, beats = librosa.beat.beat_track(y=ya, sr=22050, hop_length=512)
                bt = librosa.frames_to_time(beats, sr=22050, hop_length=512)
                bpath.write_text(json.dumps({"tempo": float(np.atleast_1d(tempo)[0]),
                    "beat_times": [round(float(x), 4) for x in bt]}), encoding="utf-8")
                del sep; torch.cuda.empty_cache()
            m = measure(p, vocals_path=vpath, beats_path=bpath)
            s35c[key] = score_of(m)
            save(S35C, s35c)
            print(f"  seed{s}: v3.5={s35c[key]['score']:.1f} [{s35c[key]['cat']}]", flush=True)
        except Exception as e:
            print(f"  seed{s}: stage2 ERR {type(e).__name__}: {e}", flush=True)

# ===== 阶段C: 报告 =====
if PHASE in ("all", "report"):
    v3c, s35c = load(V3C), load(S35C)
    ref = load(ROOT / "eval" / "rescan_v35_cache.json")  # 现任 seed61000=80.0
    ref_key = "novel_songs/batch_gacha/03_v2/seed61000.flac"
    ref35 = ref.get(ref_key, {}).get("score") if ref.get(ref_key) else None
    print(f"\n=== 03 池涌现 8 seed (v3.5 单窗, 对标 seed61000={ref35:.1f}) ===", flush=True)
    rows = []
    for s in SEEDS_NEW:
        v3 = v3c.get(f"seed{s}")
        v35 = s35c.get(f"seed{s}")
        if v35:
            rows.append((v35["score"], v3["score"] if v3 else None, s, "邻域" if s not in LOTTERY else "随机"))
    rows.sort(key=lambda x: -x[0])
    print(f"{'kind':4s} {'seed':>8s} {'v3':>6s} {'v3.5':>6s}", flush=True)
    lines = ["# 03 池涌现 8 seed: v3.5 终评 (2026-10-16)", "",
             "4 随机 (61010/13/16/19) + 4 邻域 (61003-06); 两阶段 v3.5 单窗",
             f"对标: 现任 seed61000 = {ref35:.1f}", "",
             "| kind | seed | v3 | v3.5 |", "|---|---|---|---|"]
    for v35, v3, s, kind in rows:
        print(f"{kind:4s} {s:>8d} {v3 if v3 else 0:>6.1f} {v35:>6.1f}", flush=True)
        lines.append(f"| {kind} | {s} | {v3:.1f} | {v35:.1f} |")
    if rows:
        best = rows[0]
        print(f"\n最优: seed{best[2]} = {best[0]:.1f} ({best[3]}) vs 现任 80.0 → {'★超过' if best[0] > ref35 else '未超过'}", flush=True)
        lines += ["", f"最优: seed{best[2]} = {best[0]:.1f} ({best[3]}) vs 现任 80.0 → {'★超过' if best[0] > ref35 else '未超过'}"]
    (ROOT/"eval"/"gacha03_v35_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("SAVED eval/gacha03_v35_report.md", flush=True)
