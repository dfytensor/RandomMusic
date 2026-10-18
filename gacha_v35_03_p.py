# -*- coding: utf-8 -*-
"""03 prompt 突破实验: express 后缀 (vibrato/legato/rubato) vs 默认 restrained 后缀
假设: 默认后缀 "minimal vocal ornamentation, clean straight tones" 压制颤音 →
      换表达性后缀可提升 v3.5 articulation 维, 试图破 80
对照: 现任 seed61000=80.0 (默认后缀) | 默认后缀 8seed 最优 75.3
用法: python gacha_v35_03_p.py [phase] phase=render|score|report (默认全跑)
"""
import json, subprocess, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
PHASE = sys.argv[1] if len(sys.argv) > 1 else "all"

SEEDS = [61020, 61021, 61022, 61023, 61024, 61025]
OUT = ROOT / "novel_songs" / "batch_gacha" / "03_p35"
OUT.mkdir(exist_ok=True)
V3C = ROOT / "eval" / "gacha03_p35_cache.json"
S35C = ROOT / "eval" / "gacha03_p35_stage2.json"
STEMS = ROOT / "render_stems" / "03_p35"

# express 后缀: 保留编曲/亲密控制, 替换反颤音条款
EXPRESS_SUFFIX = ("expressive but tasteful singing, gentle natural vibrato on sustained notes, "
                  "legato consonant transitions, subtle rubato micro-timing, "
                  "warm intimate tone, sparse piano with silences, breathing space between phrases")
# 基础 style (来自 novel_songs_manifest 03)
BASE_STYLE = ("Chinese style ballad, guzheng arpeggios, dizi flute echoes, warm strings, sparse piano, "
              "intimate breathy male vocal growing strong, lonely yet determined, D minor, 76 BPM")

def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
def save(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

def render_one(seed: int):
    """独立渲染进程: 内联 style (不走 render_one.py 的默认后缀)"""
    dst = OUT / f"seed{seed}.flac"
    if dst.exists() and dst.stat().st_size > 10_000_000:
        return seed, True, 0.0, "cached"
    t0 = time.time()
    # 写临时渲染脚本 (避免引号地狱 + 复用 yue2 环境)
    tmp = ROOT / "eval" / f"_render_p35_{seed}.py"
    tmp.write_text(f'''# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r"F:\\yinyue")
from novel_songs_manifest import SONGS
no = "03"
s = next(x for x in SONGS if x["no"] == no)
v2 = Path(r"F:\\yinyue\\novel_songs\\v2\\03\\lyrics_v2.txt")
lyrics = v2.read_text(encoding="utf-8")
style = {BASE_STYLE!r} + ", " + {EXPRESS_SUFFIX!r}
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(Path(r"F:\\yinyue")/"models"/"YuE2-3B"),
                                     vae=str(Path(r"F:\\yinyue")/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    song = pipe(style=style, lyrics=lyrics, cot="full", seed={seed})
    dst = Path(r"{str(dst)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    song.save(str(dst))
    print("OK 03 seed{seed} express")
finally:
    pipe.close()
''', encoding="utf-8")
    logp = ROOT / "eval" / f"render_p35_{seed}.log"
    import os
    env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1")
    with open(logp, "w", encoding="utf-8", errors="replace") as lf:
        r = subprocess.run([sys.executable, str(tmp)], cwd=str(ROOT), env=env,
                           stdout=lf, stderr=subprocess.STDOUT)
    tmp.unlink(missing_ok=True)
    ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 10_000_000
    err = "" if ok else logp.read_text(encoding="utf-8", errors="replace")[-300:]
    return seed, ok, time.time() - t0, err

# ===== 阶段A: 渲染 (串行) =====
if PHASE in ("all", "render"):
    todo = [s for s in SEEDS
            if not (OUT / f"seed{s}.flac").exists() or (OUT / f"seed{s}.flac").stat().st_size < 10_000_000]
    print(f"render express todo={len(todo)}: {todo}", flush=True)
    for s in todo:
        print(f"  rendering seed{s} (express) ...", flush=True)
        seed, ok, dt, err = render_one(s)
        print(f"  seed{s}: {'OK' if ok else 'FAIL'} {dt/60:.1f}min {err[:150]}", flush=True)

# ===== 阶段B: 两阶段评分 =====
if PHASE in ("all", "score"):
    from art_core import art_dims
    from score_v3 import measure, score_of
    v3c, s35c = load(V3C), load(S35C)
    for s in SEEDS:
        p = OUT / f"seed{s}.flac"
        key = f"seed{s}"
        if not p.exists():
            print(f"  seed{s}: 缺渲染", flush=True); continue
        if key not in v3c or v3c[key] is None:
            m = measure(p)
            v3c[key] = score_of(m) if m else None
            save(V3C, v3c)
            print(f"  seed{s}: v3={v3c[key]['score'] if v3c[key] else 0:.1f} [{v3c[key]['cat'] if v3c[key] else '?'}]", flush=True)
        if key in s35c and s35c[key] is not None:
            continue
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
    def_art = load(ROOT / "eval" / "gacha03_v35_stage2.json")  # 默认后缀 8seed
    ref = load(ROOT / "eval" / "rescan_v35_cache.json")
    ref35 = (ref.get("novel_songs/batch_gacha/03_v2/seed61000.flac") or {}).get("score")
    def_scores = [v["score"] for v in def_art.values() if v]
    def_max = max(def_scores) if def_scores else None
    def_mean = sum(def_scores)/len(def_scores) if def_scores else None
    print(f"\n=== 03 express 后缀 6 seed vs 对照 ===", flush=True)
    print(f"现任 seed61000 (默认后缀) = {ref35:.1f} | 默认后缀 8seed: max={def_max:.1f} mean={def_mean:.1f}", flush=True)
    rows = []
    for s in SEEDS:
        v3 = v3c.get(f"seed{s}"); v35 = s35c.get(f"seed{s}")
        if v35:
            rows.append((v35["score"], v3["score"] if v3 else None, s))
    rows.sort(key=lambda x: -x[0])
    print(f"{'seed':>8s} {'v3':>6s} {'v3.5':>6s}", flush=True)
    lines = ["# 03 express 后缀实验 (2026-10-16)", "",
             "后缀替换: minimal ornamentation/clean straight tones → expressive vibrato/legato/rubato",
             f"对照: 现任 61000={ref35:.1f}; 默认后缀 8seed max={def_max:.1f} mean={def_mean:.1f}", "",
             "| seed | v3 | v3.5 |", "|---|---|---|"]
    for v35, v3, s in rows:
        print(f"{s:>8d} {v3 if v3 else 0:>6.1f} {v35:>6.1f}", flush=True)
        lines.append(f"| {s} | {v3:.1f} | {v35:.1f} |")
    if rows:
        e_mean = sum(r[0] for r in rows) / len(rows)
        e_max = rows[0][0]
        verdict = "★破80" if e_max > (ref35 or 80) else "未破80"
        print(f"\nexpress: max={e_max:.1f} mean={e_mean:.1f} vs 默认 max={def_max:.1f} mean={def_mean:.1f} → {verdict}", flush=True)
        lines += ["", f"express: max={e_max:.1f} mean={e_mean:.1f} vs 默认 max={def_max:.1f} mean={def_mean:.1f} → {verdict}", ""]
    (ROOT/"eval"/"gacha03_p35_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("SAVED eval/gacha03_p35_report.md", flush=True)
