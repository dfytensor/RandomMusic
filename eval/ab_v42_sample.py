# -*- coding: utf-8 -*-
"""样板 A/B: v4.2 词(05) 同 seed 50505 渲染 → v3.5 评分 vs 旧词 81.2"""
import json, subprocess, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(r"F:\yinyue")
sys.path.insert(0, str(ROOT))

NO = sys.argv[1] if len(sys.argv) > 1 else "05"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 50505
EXPRESS_SUFFIX = ("expressive but tasteful singing, gentle natural vibrato on sustained notes, "
                  "legato consonant transitions, subtle rubato micro-timing, "
                  "warm intimate tone, sparse piano with silences, breathing space between phrases")

def main():
    # 评分
    from lyrics_prosody_v4 import score_lyrics
    t = (ROOT / "novel_songs" / "v42_lyrics" / f"{NO}.txt").read_text(encoding="utf-8")
    r = score_lyrics(t)
    print(f"v4.2({NO}) = {r['total']} | rh={r['rhyme']} cv={r['len_cv']} zdiv={r['zhe_div']} "
          f"concrete={r['concrete']} open={r['openness']}", flush=True)

    # 渲染
    dst = ROOT / "novel_songs" / "batch_gacha" / f"{NO}_v42ab" / f"seed{SEED}.flac"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not (dst.exists() and dst.stat().st_size > 10_000_000):
        tmp = ROOT / "eval" / f"_render_v42_{NO}.py"
        tmp.write_text(f'''# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r"F:\\\\yinyue")
from novel_songs_manifest import SONGS
s = next(x for x in SONGS if x["no"] == "{NO}")
lyrics = {t!r}
style = s["style"] + ", " + {EXPRESS_SUFFIX!r}
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(Path(r"F:\\\\yinyue")/"models"/"YuE2-3B"),
                                     vae=str(Path(r"F:\\\\yinyue")/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    song = pipe(style=style, lyrics=lyrics, cot="full", seed={SEED})
    song.save(str(Path(r"{str(dst)}")))
    print("OK")
finally:
    pipe.close()
''', encoding="utf-8")
        import os
        env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1",
                   PYTORCH_ALLOC_CONF="expandable_segments:True",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        logp = ROOT / "eval" / f"render_v42_{NO}.log"
        t0 = time.time()
        with open(logp, "w", encoding="utf-8", errors="replace") as lf:
            res = subprocess.run([sys.executable, str(tmp)], cwd=str(ROOT), env=env,
                                 stdout=lf, stderr=subprocess.STDOUT)
        tmp.unlink(missing_ok=True)
        ok = res.returncode == 0 and dst.exists() and dst.stat().st_size > 10_000_000
        print(f"render: {'OK' if ok else 'FAIL'} {time.time()-t0:.0f}s", flush=True)
        if not ok:
            print(logp.read_text(encoding="utf-8", errors="replace")[-300:], flush=True)
            return

    # 评分 (demucs stems 可复用 05_v3ab 的 beats? 不, 歌不同重新分)
    from score_v3 import measure, score_of
    import torch, soundfile as sf, librosa, numpy as np
    key = f"{NO}_v42_seed{SEED}"
    sd = ROOT / "render_stems" / f"{NO}_v42ab" / key
    vpath, bpath = sd / "vocals.wav", sd / "beats.json"
    if not (vpath.exists() and bpath.exists()):
        sd.mkdir(parents=True, exist_ok=True)
        from demucs.api import Separator
        sep = Separator(model="htdemucs", device="cuda")
        origin, stems = sep.separate_audio_file(str(dst))
        vocals = stems["vocals"]; accomp = origin - vocals
        sf.write(str(vpath), vocals.cpu().numpy().T, sep.samplerate)
        acc_np = accomp.cpu().numpy().mean(axis=0)
        ya = librosa.resample(acc_np, orig_sr=sep.samplerate, target_sr=22050)
        tempo, beats = librosa.beat.beat_track(y=ya, sr=22050, hop_length=512)
        bt = librosa.frames_to_time(beats, sr=22050, hop_length=512)
        bpath.write_text(json.dumps({"tempo": float(np.atleast_1d(tempo)[0]),
            "beat_times": [round(float(x), 4) for x in bt]}), encoding="utf-8")
        del sep; torch.cuda.empty_cache()
    m = measure(dst, vocals_path=vpath, beats_path=bpath)
    sc = score_of(m)
    ref = json.loads((ROOT / "eval" / "ab_v3x_v35.json").read_text(encoding="utf-8")).get(f"{NO}_ref")
    print(f"v3.5: 新词={sc['score']:.1f} [{sc['cat']}] vs 旧词={ref:.1f} → {sc['score']-ref:+.1f}", flush=True)
    out = ROOT / "eval" / "ab_v42_v35.json"
    d = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    d[f"{NO}_new"] = sc["score"]
    d[f"{NO}_ref"] = ref
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
