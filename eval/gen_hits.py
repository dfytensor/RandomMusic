# -*- coding: utf-8 -*-
"""3 首自由爆款: 撒网抽卡管线
概念: 充电(C1慢板) / 小卖部(C2怀旧) / 疯是年假(C4快板)
每首 6 seed 跨量级撒网 → v3 粗筛 → top2 Demucs → v3.5 终评
缓存: eval/new_hits_v3_cache.json / new_hits_v35_cache.json
"""
import json, subprocess, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(r"F:\yinyue")
sys.path.insert(0, str(ROOT))

HITS = {
    "发疯年假": {
        "style": ("energetic pop rock, punchy drums, cathartic dynamic chorus, raw vocal energy, "
                  "expressive singing, gentle natural vibrato, legato consonant transitions, "
                  "anthemic but playful"),
        "seeds": [1024, 27182, 58888, 65535, 42, 777],
    },
    "充电": {  # 补评 s27182 (真C1慢板卡) 的 v3.5
        "style": ("melancholic late-night ballad, sparse piano with silences, intimate, "
                  "breathing space between phrases, warm dark tone, restrained singing, "
                  "natural irregular breathing, soft dynamics"),
        "seeds": [27182],
    },
}
V3C = ROOT / "eval" / "new_hits_v3_cache.json"
V35C = ROOT / "eval" / "new_hits_v35_cache.json"

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else {}
def save(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

def render(name, style, seed, dst):
    if dst.exists() and dst.stat().st_size > 10_000_000:
        return True
    tmp = ROOT / "eval" / f"_render_hit_{abs(hash(name))%10000}_{seed}.py"
    t = (ROOT / "novel_songs" / "new_hits" / f"{name}.txt").read_text(encoding="utf-8")
    tmp.write_text(f'''# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r"F:\\\\yinyue")
lyrics = {t!r}
style = {style!r}
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(r"C:\\yue2_model\\models\\m-a-p--YuE2-3B\\snapshots\\master",
                                     vae=str(Path(r"F:\\yinyue")/"models"/"YuE2-Vae"),
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
    logp = ROOT / "eval" / f"render_hit_{seed}_{abs(hash(name))%10000}.log"
    with open(logp, "w", encoding="utf-8", errors="replace") as lf:
        r = subprocess.run([sys.executable, str(tmp)], cwd=str(ROOT), env=env,
                           stdout=lf, stderr=subprocess.STDOUT)
    tmp.unlink(missing_ok=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 10_000_000

def score_v3(p, key, cache):
    if cache.get(key) is not None:
        return cache[key]
    from score_v3 import measure, score_of
    m = measure(p)
    cache[key] = score_of(m) if m else None
    save(V3C, cache)
    return cache[key]

def score_v35(no, p, key):
    from score_v3 import measure, score_of
    import torch, soundfile as sf, librosa, numpy as np
    sd = ROOT / "render_stems" / "new_hits" / key
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
    v3c, v35c = load(V3C), load(V35C)
    for name, cfg in HITS.items():
        print(f"\n===== {name} 撒网 seeds={cfg['seeds']} =====", flush=True)
        OUT = ROOT / "novel_songs" / "new_hits" / "gacha"
        OUT.mkdir(parents=True, exist_ok=True)
        for seed in cfg["seeds"]:
            key = f"{name}_s{seed}"
            if v3c.get(key) is not None:
                print(f"  {key}: cached v3={v3c[key]['score'] if v3c[key] else 'None'}", flush=True)
                continue
            dst = OUT / f"{name}_seed{seed}.flac"
            ok = False
            for attempt in range(2):
                print(f"  render {key} (try{attempt+1}) ...", flush=True)
                t0 = time.time()
                ok = render(name, cfg["style"], seed, dst)
                print(f"  {key}: {'OK' if ok else 'FAIL'} {(time.time()-t0)/60:.1f}min", flush=True)
                if ok:
                    break
            if not ok:
                v3c[key] = None
                save(V3C, v3c)
                continue
            try:
                r = score_v3(dst, key, v3c)
                print(f"  {key}: v3={r['score']:.1f} [{r['cat']}]", flush=True)
            except Exception as e:
                print(f"  {key}: v3 ERR {type(e).__name__}: {e}", flush=True)
        # 阶段2: top2 Demucs → v3.5
        rows = sorted([(v["score"], k) for k, v in v3c.items()
                       if v and k.startswith(f"{name}_s")], key=lambda x: -x[0])[:2]
        print(f"  {name} 阶段2候选: {rows}", flush=True)
        for sc, key in rows:
            if v35c.get(key) is not None:
                print(f"  {key}: v3.5 cached {v35c[key]['score']:.1f}", flush=True)
                continue
            dst = OUT / f"{name}_seed{key.split('_s')[1]}.flac"
            try:
                r = score_v35(name, dst, key)
                v35c[key] = r
                save(V35C, v35c)
                print(f"  {key}: v3.5={r['score']:.1f} [{r['cat']}]", flush=True)
            except Exception as e:
                print(f"  {key}: v3.5 ERR {type(e).__name__}: {e}", flush=True)
    print("\nHITS_DONE", flush=True)

if __name__ == "__main__":
    main()

