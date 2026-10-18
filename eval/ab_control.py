# -*- coding: utf-8 -*-
"""Step A: 同 seed 严格对照 — 每首现任冠军 seed, 同后缀, 只换词 (旧词已有分, 渲染新词版)
对照分: rescan_v35_cache (默认后缀冠军) / gacha_ex_v3_stage2 (express 冠军)
产物: novel_songs/batch_gacha/<no>_v3ab/seed<seed>.flac + eval/ab_v3x_v35.json
用法: python ab_control.py [MAX_SONGS=12]
"""
import json, subprocess, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
MAX_SONGS = int(sys.argv[1]) if len(sys.argv) > 1 else 12

NOS = [f"{i:02d}" for i in range(1, 13)]
# 现任冠军: (pool, seed, 后缀类型, 现任分从对应缓存取)
REF_DEFAULT = ROOT / "eval" / "rescan_v35_cache.json"   # 默认后缀冠军 (seed90xxx 等)
REF_EXPRESS = ROOT / "eval" / "gacha_ex_v3_stage2.json" # express 冠军 (seed5N02x)
OUT_C = ROOT / "eval" / "ab_v3x_v35.json"

# 每首对照 seed: 动态取 express 批 (gacha_ex_v3_stage2) 该歌最佳 seed — 同 seed 只换词
def champ_express_seed(no):
    ex = load(REF_EXPRESS)
    rows = [(v["score"], int(k.split("_s")[1])) for k, v in ex.items()
            if v and k.startswith(f"{no}_s") and v["score"] > -100]
    if not rows:
        return None
    rows.sort(key=lambda x: -x[0])
    return rows[0]  # (score, seed)

EXPRESS_SUFFIX = ("expressive but tasteful singing, gentle natural vibrato on sustained notes, "
                  "legato consonant transitions, subtle rubato micro-timing, "
                  "warm intimate tone, sparse piano with silences, breathing space between phrases")
DEFAULT_SUFFIX = ("restrained singing, minimal vocal ornamentation, clean straight tones, "
                  "natural irregular breathing, warm dark tone, sparse piano with silences, "
                  "breathing space between phrases, intimate, less is more")

def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
def save(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

def ref_express_score(no, seed):
    ex = load(REF_EXPRESS)
    v = ex.get(f"{no}_s{seed}")
    return v["score"] if v else None

def render_new_words(no, seed, suffix, dst):
    if dst.exists() and dst.stat().st_size > 10_000_000:
        return True, 0.0, ""
    t0 = time.time()
    tmp = ROOT / "eval" / f"_render_ab_{no}_{seed}.py"
    lyrics = (ROOT / "novel_songs" / "v3_lyrics" / f"{no}.txt").read_text(encoding="utf-8")
    tmp.write_text(f'''# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r"F:\\yinyue")
from novel_songs_manifest import SONGS
s = next(x for x in SONGS if x["no"] == "{no}")
lyrics = {lyrics!r}
style = s["style"] + ", " + {suffix!r}
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(Path(r"F:\\yinyue")/"models"/"YuE2-3B"),
                                     vae=str(Path(r"F:\\yinyue")/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    song = pipe(style=style, lyrics=lyrics, cot="full", seed={seed})
    dst = Path(r"{str(dst)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    song.save(str(dst))
    print("OK {no} seed{seed} ab")
finally:
    pipe.close()
''', encoding="utf-8")
    logp = ROOT / "eval" / f"render_ab_{no}_{seed}.log"
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

def score_one(no, p, key):
    from score_v3 import measure, score_of
    import torch, soundfile as sf, librosa, numpy as np
    sd = ROOT / "render_stems" / f"{no}_v3ab" / key
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
    return score_of(m)

def main():
    cache = load(OUT_C)
    todo = [no for no in NOS[:MAX_SONGS] if cache.get(f"{no}_new") is None]
    print(f"AB control pending: {todo}", flush=True)
    for no in todo:
        champ = champ_express_seed(no)
        if champ is None:
            print(f"{no}: express 缓存无数据, 跳过", flush=True)
            continue
        ref, seed = champ
        kind = "express"
        dst = ROOT / "novel_songs" / "batch_gacha" / f"{no}_v3ab" / f"seed{seed}.flac"
        suffix = EXPRESS_SUFFIX
        for attempt in range(2):
            print(f"  {no} rendering new-words seed{seed} (try{attempt+1}) [ref={ref:.1f}] ...", flush=True)
            ok, dt, err = render_new_words(no, seed, suffix, dst)
            print(f"  {no} seed{seed}: {'OK' if ok else 'FAIL'} {dt/60:.1f}min {err[:120]}", flush=True)
            if ok:
                break
        if not ok:
            print(f"  {no}: 渲染两次失败, 跳过", flush=True)
            continue
        try:
            r = score_one(no, dst, f"{no}_seed{seed}")
            cache[f"{no}_new"] = r["score"]
            cache[f"{no}_ref"] = ref
            save(OUT_C, cache)
            d = r["score"] - ref
            verdict = "★新词更优" if d > 0.5 else ("≈持平" if d >= -0.5 else "✗新词受损")
            print(f"  {no}: 新词={r['score']:.1f} vs 旧词={ref:.1f} → {d:+.1f} {verdict}", flush=True)
        except Exception as e:
            print(f"  {no}: score ERR {type(e).__name__}: {e}", flush=True)
    # 汇总
    rows = [(no, cache.get(f"{no}_ref"), cache.get(f"{no}_new")) for no in NOS
            if cache.get(f"{no}_new") is not None]
    if rows:
        print("\n===== AB 同seed对照汇总 =====")
        n_win = n_lose = 0
        for no, ref, new in rows:
            d = new - ref
            if d > 0.5: n_win += 1
            elif d < -0.5: n_lose += 1
            print(f"{no}: 旧={ref:.1f} 新={new:.1f} ({d:+.1f})")
        dm = sum(n - r for _, r, n in rows) / len(rows)
        print(f"\n新词胜 {n_win} / 平 {len(rows)-n_win-n_lose} / 负 {n_lose} | 平均差 {dm:+.1f}")
    print("AB_DONE", flush=True)

if __name__ == "__main__":
    main()
