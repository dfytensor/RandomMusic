# -*- coding: utf-8 -*-
"""v4.2 词分布验证: 05 渲染 seeds 50500-50502,50503 (50505 已有), 对比旧词 express 分布"""
import json, subprocess, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(r"F:\yinyue")
sys.path.insert(0, str(ROOT))

NO = "04"
SEEDS = [40500, 40501, 40502, 40503, 40505]
EXPRESS_SUFFIX = ("expressive but tasteful singing, gentle natural vibrato on sustained notes, "
                  "legato consonant transitions, subtle rubato micro-timing, "
                  "warm intimate tone, sparse piano with silences, breathing space between phrases")
CACHE = ROOT / "eval" / "ab_v42_v35.json"

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else {}
def save(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

def render(no, seed, dst):
    if dst.exists() and dst.stat().st_size > 10_000_000:
        return True
    tmp = ROOT / "eval" / f"_render_v42_{no}_{seed}.py"
    t = (ROOT / "novel_songs" / "v42_lyrics" / f"{no}.txt").read_text(encoding="utf-8")
    tmp.write_text(f'''# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r"F:\\\\yinyue")
from novel_songs_manifest import SONGS
s = next(x for x in SONGS if x["no"] == "{no}")
lyrics = {t!r}
style = s["style"] + ", " + {EXPRESS_SUFFIX!r}
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(Path(r"F:\\\\yinyue")/"models"/"YuE2-3B"),
                                     vae=str(Path(r"F:\\\\yinyue")/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    song = pipe(style=style, lyrics=lyrics, cot="full", seed={seed})
    song.save(str(Path(r"{str(dst)}")))
    print("OK")
finally:
    pipe.close()
''', encoding="utf-8")
    import os
    env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1",
               PYTORCH_ALLOC_CONF="expandable_segments:True",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    logp = ROOT / "eval" / f"render_v42_{no}_{seed}.log"
    with open(logp, "w", encoding="utf-8", errors="replace") as lf:
        r = subprocess.run([sys.executable, str(tmp)], cwd=str(ROOT), env=env,
                           stdout=lf, stderr=subprocess.STDOUT)
    tmp.unlink(missing_ok=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 10_000_000

def score(no, p, key):
    from score_v3 import measure, score_of
    import torch, soundfile as sf, librosa, numpy as np
    sd = ROOT / "render_stems" / f"{no}_v42ab" / key
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
    cache = load(CACHE)
    for seed in SEEDS:
        key = f"{NO}_seed{seed}"
        if cache.get(key) is not None:
            print(f"{key}: cached {cache[key]['score']:.1f}", flush=True)
            continue
        dst = ROOT / "novel_songs" / "batch_gacha" / f"{NO}_v42ab" / f"seed{seed}.flac"
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            print(f"render {key} (try{attempt+1}) ...", flush=True)
            ok = render(NO, seed, dst)
            if ok:
                break
        if not ok:
            print(f"{key}: FAIL", flush=True)
            continue
        r = score(NO, dst, key)
        cache[key] = r
        save(CACHE, cache)
        print(f"{key}: v3.5={r['score']:.1f} [{r['cat']}]", flush=True)
    rows = [cache[k]["score"] for k in (f"{NO}_seed{s}" for s in SEEDS) if cache.get(k)]
    if rows:
        import statistics
        print(f"\n04 v4.2 分布: {sorted(rows, reverse=True)}")
        print(f"mean={statistics.mean(rows):.1f} top={max(rows):.1f} n={len(rows)}")
        print(f"04 旧词 express: mean=52.5 top=63.6 n=6")
        print(f"05 v3 韵文 v3x: 仅 10600 批是 01 池的… (05 未跑过 v3x)")

if __name__ == "__main__":
    main()

