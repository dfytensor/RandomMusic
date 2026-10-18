# -*- coding: utf-8 -*-
"""v4.2 词批量抽卡: 剩余9首 × 5 seed (与旧词 express 批同 seed 段), 渲染→Demucs→v3.5 终评
对照: 旧词 express 分布 (gacha_ex_v3_stage2) + 全库现任 (rescan_v35_cache)
缓存: eval/ab_v42_v35.json  断点续跑
"""
import json, subprocess, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(r"F:\yinyue")
sys.path.insert(0, str(ROOT))

NOS = ["01", "02", "03", "06", "07", "09", "10", "11", "12"]
CHAMPS = {"01": "01_v2", "02": "02_wide", "03": "03_v2", "04": "04", "05": "05_wide",
          "06": "06_wide", "07": "07_adj", "08": "11", "09": "09", "10": "10",
          "11": "11", "12": "12"}
EXPRESS_SUFFIX = ("expressive but tasteful singing, gentle natural vibrato on sustained notes, "
                  "legato consonant transitions, subtle rubato micro-timing, "
                  "warm intimate tone, sparse piano with silences, breathing space between phrases")
CACHE = ROOT / "eval" / "ab_v42_v35.json"

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else {}
def save(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

def seeds_for(no):
    n = int(no)
    return [n * 10000 + 500, n * 10000 + 501, n * 10000 + 502, n * 10000 + 503, n * 10000 + 505]

def render(no, seed, dst):
    if dst.exists() and dst.stat().st_size > 10_000_000:
        return True
    tmp = ROOT / "eval" / f"_render_v42b_{no}_{seed}.py"
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
    logp = ROOT / "eval" / f"render_v42b_{no}_{seed}.log"
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
    for no in NOS:
        done = all(cache.get(f"{no}_seed{s}") is not None for s in seeds_for(no))
        if done:
            print(f"{no}: cached", flush=True)
            continue
        print(f"\n===== {no} v4.2 抽卡 =====", flush=True)
        for seed in seeds_for(no):
            key = f"{no}_seed{seed}"
            if cache.get(key) is not None:
                print(f"  {key}: cached {cache[key]['score']:.1f}", flush=True)
                continue
            dst = ROOT / "novel_songs" / "batch_gacha" / f"{no}_v42ab" / f"seed{seed}.flac"
            dst.parent.mkdir(parents=True, exist_ok=True)
            ok = False
            for attempt in range(2):
                print(f"  render {key} (try{attempt+1}) ...", flush=True)
                t0 = time.time()
                ok = render(no, seed, dst)
                print(f"  {key}: {'OK' if ok else 'FAIL'} {(time.time()-t0)/60:.1f}min", flush=True)
                if ok:
                    break
            if not ok:
                print(f"  {key}: 渲染失败跳过", flush=True)
                continue
            try:
                r = score(no, dst, key)
                cache[key] = r
                save(CACHE, cache)
                print(f"  {key}: v3.5={r['score']:.1f} [{r['cat']}]", flush=True)
            except Exception as e:
                print(f"  {key}: score ERR {type(e).__name__}: {e}", flush=True)
    # 汇总
    print("\n===== 汇总: v4.2 分布 vs 旧词 express 分布 vs 全库现任 =====", flush=True)
    expr = load(ROOT / "eval" / "gacha_ex_v3_stage2.json")
    rescan = load(ROOT / "eval" / "rescan_v35_cache.json")
    import statistics
    summary = {}
    for no in ["04", "05", "08"] + NOS:
        seeds = ([40500, 40501, 40502, 40503, 40505] if no == "04" else
                 [50500, 50501, 50502, 50503, 50505] if no == "05" else
                 [80500, 80501, 80502, 80503, 80505] if no == "08" else seeds_for(no))
        rows = [cache[k]["score"] for k in (f"{no}_seed{s}" for s in seeds) if cache.get(k)]
        if not rows:
            continue
        et = [v["score"] for k, v in expr.items() if v and k.startswith(f"{no}_s") and v["score"] > -100]
        rt = [v["score"] for k, v in rescan.items() if v and k.split("/")[2][:2] == no]
        incumbent = max((et if et else []) + rt)
        d = max(rows) - incumbent
        verdict = "★新冠军" if d > 0.5 else ("≈平" if d >= -0.5 else "未超")
        summary[no] = {"v42_top": max(rows), "v42_mean": statistics.mean(rows), "n": len(rows),
                       "old_top": max(et) if et else None, "old_mean": statistics.mean(et) if et else None,
                       "incumbent": incumbent, "diff": d, "verdict": verdict}
        print(f"{no}: v4.2 top={max(rows):.1f} mean={statistics.mean(rows):.1f}(n={len(rows)}) | "
              f"旧词 top={max(et):.1f} if et else 0 | 现任={incumbent:.1f} → {d:+.1f} {verdict}", flush=True)
    (ROOT / "eval" / "ab_v42_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("BATCH42_DONE", flush=True)

if __name__ == "__main__":
    main()
