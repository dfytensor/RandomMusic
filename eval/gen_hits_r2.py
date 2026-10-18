# -*- coding: utf-8 -*-
"""第二轮: 爬坡 + 新撒网
- 充电: s42(81.7 C4) 邻域爬坡 + 新量级撒网 (冲85+)
- 小卖部: s777(71.3) 邻域爬坡 + 新撒网 (当前 top 偏低, 补强)
- 发疯年假: s27182(73.7 C3) 邻域爬坡 + 新撒网
缓存复用 new_hits_v3_cache.json (已存 18 seed, 只跑新增)
"""
import json, subprocess, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(r"F:\yinyue")
sys.path.insert(0, str(ROOT))

HITS = {
    "充电": {
        "style": ("melancholic late-night ballad, sparse piano with silences, intimate, "
                  "breathing space between phrases, warm dark tone, restrained singing, "
                  "natural irregular breathing, soft dynamics"),
        "seeds": [41, 43, 40, 44, 1234, 9999, 31415, 99991],
    },
    "小卖部": {
        "style": ("warm nostalgic folk pop, acoustic guitar, bright but gentle, mid-tempo, "
                  "clean warm vocals, light percussion, tender and slightly bittersweet"),
        "seeds": [776, 778, 775, 1234, 9999, 31415, 99991],
    },
    "发疯年假": {
        "style": ("energetic pop rock, punchy drums, cathartic dynamic chorus, raw vocal energy, "
                  "expressive singing, gentle natural vibrato, legato consonant transitions, "
                  "anthemic but playful"),
        "seeds": [27181, 27183, 27184, 1234, 9999, 31415, 99991],
    },
}
V3C = ROOT / "eval" / "new_hits_v3_cache.json"
V35C = ROOT / "eval" / "new_hits_v35_cache.json"
MODEL_C = r"C:\yue2_model\models\m-a-p--YuE2-3B\snapshots\master"
VAE = r"F:\yinyue\models\YuE2-Vae"

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else {}
def save(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

def render(name, style, seed, dst):
    if dst.exists() and dst.stat().st_size > 10_000_000:
        return True
    tmp = ROOT / "eval" / f"_render_r2_{abs(hash(name))%10000}_{seed}.py"
    t = (ROOT / "novel_songs" / "new_hits" / f"{name}.txt").read_text(encoding="utf-8")
    tmp.write_text(f'''# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r"F:\\\\yinyue")
lyrics = {t!r}
style = {style!r}
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(r"{MODEL_C}", vae=r"{VAE}",
                                     device="cuda", progress=False, verify_hashes=False)
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
    logp = ROOT / "eval" / f"render_r2_{seed}_{abs(hash(name))%10000}.log"
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

def score_v35(name, p, key):
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
        print(f"\n===== {name} R2 seeds={cfg['seeds']} =====", flush=True)
        OUT = ROOT / "novel_songs" / "new_hits" / "gacha"
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
        # 阶段2: 全曲 top3 (含历史) → v3.5
        rows = sorted([(v["score"], k) for k, v in v3c.items()
                       if v and k.startswith(f"{name}_s")], key=lambda x: -x[0])[:3]
        print(f"  {name} top3: {rows}", flush=True)
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
        # 汇总该歌当前最佳
        best = sorted([(v["score"], k) for k, v in v35c.items()
                       if v and k.startswith(f"{name}_s")], key=lambda x: -x[0])
        if best:
            print(f"  ★ {name} v3.5 榜: {[(round(s,1), k) for s, k in best[:3]]}", flush=True)
    print("\nR2_DONE", flush=True)

if __name__ == "__main__":
    main()
