# -*- coding: utf-8 -*-
"""v3 韵律新词抽卡: 每首 6 新 seed, express 后缀, v3 粗筛 → Demucs → v3.5 终评, 对照现任冠军
歌词: novel_songs/v3_lyrics/<no>.txt (全部 ≥80 古诗级韵律)
seed 方案: 每首 N*10000+600-605 (不与历史批次冲突)
断点续跑: novel_songs/batch_gacha/<no>_v3x/ + eval/gacha_v3x_v3_{cache,stage2}.json
          + render_stems/<no>_v3x/
用法: python gacha_v3_lyrics.py [MAX_SONGS=1]
"""
import json, subprocess, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
MAX_SONGS = int(sys.argv[1]) if len(sys.argv) > 1 else 1

SONG_NOS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
V3C = ROOT / "eval" / "gacha_v3x_v3_cache.json"
S35C = ROOT / "eval" / "gacha_v3x_v3_stage2.json"
REF = ROOT / "eval" / "rescan_v35_cache.json"  # 现任冠军 v3.5 分
REPORTS = {}

CHAMPS = {"01": "01_v2", "02": "02_wide", "03": "03_v2", "04": "04", "05": "05_wide",
          "06": "06_wide", "07": "07_adj", "08": "11", "09": "09", "10": "10",
          "11": "11", "12": "12"}

EXPRESS_SUFFIX = ("expressive but tasteful singing, gentle natural vibrato on sustained notes, "
                  "legato consonant transitions, subtle rubato micro-timing, "
                  "warm intimate tone, sparse piano with silences, breathing space between phrases")

def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
def save(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

def seeds_for(no):
    n = int(no)
    return [n * 10000 + 600 + i for i in range(6)]  # 01→10600-10605 ... 12→120600-120605

def lyrics_for(no):
    p = ROOT / "novel_songs" / "v3_lyrics" / f"{no}.txt"
    return p.read_text(encoding="utf-8")

def ref_score(no):
    """池内现任最佳 v3.5 分 (rescan_v35_cache 池内 max)"""
    pool = CHAMPS[no]
    ref = load(REF)
    best, bseed, bcat = None, None, None
    for k, v in ref.items():
        if not v or not k.startswith(f"novel_songs/batch_gacha/{pool}/"):
            continue
        if best is None or v["score"] > best:
            best, bseed, bcat = v["score"], Path(k).stem, v["cat"]
    return best, bcat, bseed

def render_one(no, seed, dst):
    if dst.exists() and dst.stat().st_size > 10_000_000:
        return True, 0.0, "cached"
    t0 = time.time()
    tmp = ROOT / "eval" / f"_render_v3x_{no}_{seed}.py"
    lyrics = lyrics_for(no)
    tmp.write_text(f'''# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r"F:\\yinyue")
from novel_songs_manifest import SONGS
s = next(x for x in SONGS if x["no"] == "{no}")
lyrics = {lyrics!r}
style = s["style"] + ", " + {EXPRESS_SUFFIX!r}
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(Path(r"F:\\yinyue")/"models"/"YuE2-3B"),
                                     vae=str(Path(r"F:\\yinyue")/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    song = pipe(style=style, lyrics=lyrics, cot="full", seed={seed})
    dst = Path(r"{str(dst)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    song.save(str(dst))
    print("OK {no} seed{seed} v3x")
finally:
    pipe.close()
''', encoding="utf-8")
    logp = ROOT / "eval" / f"render_v3x_{no}_{seed}.log"
    import os
    env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1",
               PYTORCH_ALLOC_CONF="expandable_segments:True",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with open(logp, "w", encoding="utf-8", errors="replace") as lf:
        r = subprocess.run([sys.executable, str(tmp)], cwd=str(ROOT), env=env,
                           stdout=lf, stderr=subprocess.STDOUT)
    tmp.unlink(missing_ok=True)
    ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 10_000_000
    err = "" if ok else logp.read_text(encoding="utf-8", errors="replace")[-300:]
    return ok, time.time() - t0, err

def score_one(no, p, key, v3c, s35c):
    if key not in v3c or v3c[key] is None:
        from score_v3 import measure, score_of
        m = measure(p)
        v3c[key] = score_of(m) if m else None
        save(V3C, v3c)
    if key in s35c and s35c[key] is not None:
        return s35c[key]
    from score_v3 import measure, score_of
    import torch, soundfile as sf, librosa, numpy as np
    sd = ROOT / "render_stems" / f"{no}_v3x" / key
    vpath, bpath = sd / "vocals.wav", sd / "beats.json"
    if not (vpath.exists() and bpath.exists()):
        sd.mkdir(parents=True, exist_ok=True)
        from demucs.api import Separator
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
    return s35c[key]

def song_done(no, seeds):
    v3c, s35c = load(V3C), load(S35C)
    for s in seeds:
        k = f"{no}_s{s}"
        if s35c.get(k) is None:
            return False
    return True

def process_song(no):
    seeds = seeds_for(no)
    OUT = ROOT / "novel_songs" / "batch_gacha" / f"{no}_v3x"
    OUT.mkdir(exist_ok=True)
    v3c, s35c = load(V3C), load(S35C)
    print(f"\n===== {no} v3_lyrics seeds={seeds} =====", flush=True)
    for s in seeds:
        dst = OUT / f"seed{s}.flac"
        for attempt in range(2):  # OOM/碎片重试
            if dst.exists() and dst.stat().st_size > 10_000_000:
                break
            print(f"  rendering seed{s} (try{attempt+1}) ...", flush=True)
            ok, dt, err = render_one(no, s, dst)
            print(f"  seed{s}: {'OK' if ok else 'FAIL'} {dt/60:.1f}min {err[:120]}", flush=True)
            if ok:
                break
        else:
            print(f"  {no}: seed{s} 两次失败, 跳过该 seed", flush=True)
    for s in seeds:
        key = f"{no}_s{s}"
        try:
            r = score_one(no, OUT / f"seed{s}.flac", key, v3c, s35c)
            v3 = v3c.get(key)
            print(f"  seed{s}: v3={v3['score'] if v3 else 0:.1f} v3.5={r['score']:.1f} [{r['cat']}]", flush=True)
        except Exception as e:
            print(f"  seed{s}: score ERR {type(e).__name__}: {e}", flush=True)
    ref35, refcat, refseed = ref_score(no)
    rows = [(s35c[f"{no}_s{s}"]["score"], v3c.get(f"{no}_s{s}", {}).get("score"), s)
            for s in seeds if s35c.get(f"{no}_s{s}") and s35c[f"{no}_s{s}"]["score"] > -100]
    rows.sort(key=lambda x: -x[0])
    if rows:
        best = rows[0]
        verdict = "★超现任" if ref35 and best[0] > ref35 + 0.5 else ("平" if ref35 and abs(best[0]-ref35) <= 0.5 else "未超")
        print(f"  {no}: v3x best={best[0]:.1f} (seed{best[2]}) vs 池内最佳 {ref35:.1f} ({refseed} [{refcat}]) → {verdict}", flush=True)
        REPORTS[no] = {"ref35": ref35, "ref_cat": refcat, "ref_seed": refseed,
                       "best": best[0], "best_seed": best[2], "verdict": verdict,
                       "rows": rows, "mean": sum(r[0] for r in rows)/len(rows)}

def write_report():
    lines = ["# v3 韵律新词抽卡: express 后缀 + 两阶段 v3.5 (2026-10-17)", "",
             "歌词 v3 全部 ≥80 (古诗级韵律), 每首 6 新 seed (N*10000+600-605)", "",
             "| 歌 | 现任 | 现任v3.5 | v3x best | seed | mean | verdict |",
             "|---|---|---|---|---|---|---|"]
    n_over = 0
    for no in SONG_NOS:
        d = REPORTS.get(no)
        if not d:
            lines.append(f"| {no} | - | - | - | - | - | 待跑 |")
            continue
        if d["verdict"].startswith("★"):
            n_over += 1
        lines.append(f"| {no} | {d['ref_seed']} | {d['ref35']:.1f} | {d['best']:.1f} | {d['best_seed']} | "
                     f"{d['mean']:.1f} | {d['verdict']} |")
    lines += ["", f"超越现任: {n_over}/12", ""]
    (ROOT / "eval" / "gacha_v3x_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSAVED eval/gacha_v3x_report.md (done {len(REPORTS)}/12)", flush=True)

def rebuild_report(no):
    seeds = seeds_for(no)
    v3c, s35c = load(V3C), load(S35C)
    rows = [(s35c[f"{no}_s{s}"]["score"], v3c.get(f"{no}_s{s}", {}).get("score"), s)
            for s in seeds if s35c.get(f"{no}_s{s}") and s35c[f"{no}_s{s}"]["score"] > -100]
    if not rows:
        return
    rows.sort(key=lambda x: -x[0])
    ref35, refcat, refseed = ref_score(no)
    best = rows[0]
    verdict = "★超现任" if ref35 and best[0] > ref35 + 0.5 else ("平" if ref35 and abs(best[0]-ref35) <= 0.5 else "未超")
    REPORTS[no] = {"ref35": ref35, "ref_cat": refcat, "ref_seed": refseed,
                   "best": best[0], "best_seed": best[2], "verdict": verdict,
                   "rows": rows, "mean": sum(r[0] for r in rows)/len(rows)}

todo = [no for no in SONG_NOS if not song_done(no, seeds_for(no))]
done = [no for no in SONG_NOS if no not in todo]
print(f"songs pending={todo[:MAX_SONGS]} ({len(todo)} total, done={done})", flush=True)
for no in todo[:MAX_SONGS]:
    process_song(no)
for no in done:
    if no not in REPORTS:
        rebuild_report(no)
write_report()
print("BATCH_DONE", flush=True)
