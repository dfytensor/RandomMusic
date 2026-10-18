# -*- coding: utf-8 -*-
"""自优化闭环：生成 -> 转谱 -> Critic 评审 -> Reviser 定向改 -> 再生成 (map §6/§7/§8)。
用法: python self_optimize.py inspiration.json [--seeds 520,731] [--max-rounds 2]
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from critic_melody import evaluate, load_agg, format_report
import generate_from_inspiration as gfi
import reviser

SS2_PY = ROOT / ".venv-ss2" / "Scripts" / "python.exe"
TRANSCRIBE = ROOT / "transcribe.py"


def transcribe(flac_path, out_dir):
    import os
    import librosa
    import soundfile as sf
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "song.wav"
    y, _ = librosa.load(str(flac_path), sr=24000, mono=True, duration=300)
    sf.write(wav, y, 24000, subtype="PCM_16")
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1")
    subprocess.run([str(SS2_PY), str(TRANSCRIBE), str(wav), str(out_dir / "realized")],
                   check=True, capture_output=True, env=env)
    return out_dir / "realized" / "score.abc"


def gen(pipe, spec, lyrics, seed, abc_text, out_dir):
    out_dir = Path(out_dir)
    kwargs = dict(style=spec["style"], lyrics=lyrics, seed=seed)
    if abc_text:
        kwargs.update(abc=abc_text, cot="full")
    else:
        kwargs.update(cot="full")
    song = pipe(**kwargs)
    flac = out_dir / f"song_seed{seed}.flac"
    song.save(str(flac))
    song.save_artifacts(str(out_dir / f"artifacts_seed{seed}"))
    return flac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inspiration")
    ap.add_argument("--seeds", default="520,731")
    ap.add_argument("--max-rounds", type=int, default=2)
    a = ap.parse_args()

    inp = json.loads(Path(a.inspiration).read_text(encoding="utf-8"))
    spec = gfi.build_spec(inp)
    lyrics, has_lyrics = gfi.load_lyrics(inp, spec)
    agg = load_agg()
    seeds = [int(s) for s in a.seeds.split(",")]
    work = ROOT / "creations" / f"selfopt_{time.strftime('%Y%m%d_%H%M%S')}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"SPEC: {json.dumps(spec, ensure_ascii=False)}\nWORK: {work}", flush=True)

    from yue2 import YuE2Pipeline
    pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                        vae=str(ROOT / "models" / "YuE2-Vae"),
                                        device="cuda", progress=False)

    rounds = []
    best = None  # dict(seed, flac, ev, abc_plan_text, round)

    def assess(seed, flac, plan_text, rnd):
        realized = transcribe(flac, work / f"round{rnd}_seed{seed}")
        ev = evaluate(realized, agg)
        (work / f"round{rnd}_seed{seed}_critic.md").write_text(
            format_report(realized, ev), encoding="utf-8")
        print(f"  [round{rnd} seed{seed}] critic {ev['passed']}/{ev['total']}", flush=True)
        return dict(seed=seed, flac=str(flac), ev=ev, plan_text=plan_text, round=rnd)

    # Round 0: 自由生成
    print("[round 0] free generation ...", flush=True)
    for seed in seeds:
        d = work / f"round0_seed{seed}"
        d.mkdir(exist_ok=True)
        flac = gen(pipe, spec, lyrics, seed, None, work / f"round0_seed{seed}")
        plan_text = (work / f"round0_seed{seed}" / f"artifacts_seed{seed}" / "score.abc").read_text(encoding="utf-8")
        cand = assess(seed, flac, plan_text, 0)
        rounds.append(cand)
        if best is None or cand["ev"]["passed"] > best["ev"]["passed"]:
            best = cand

    # 后续轮：定向修改
    rng = random.Random(99)
    for r in range(1, a.max_rounds + 1):
        if best["ev"]["passed"] == best["ev"]["total"]:
            break
        print(f"[round {r}] revising plan of seed{best['seed']} ...", flush=True)
        new_text, log, ev_plan = reviser.fix(best["plan_text"], agg)
        applied = [l for l in log if l.get("result") == "applied"]
        print(f"  edits: {json.dumps(log, ensure_ascii=False)}", flush=True)
        if not applied:
            new_seed = rng.randrange(100, 99999)
            print(f"  no auto-fix applicable -> reroll seed {new_seed}", flush=True)
            d = work / f"round{r}_seed{new_seed}"
            d.mkdir(exist_ok=True)
            flac = gen(pipe, spec, lyrics, new_seed, None, d)
            cand = assess(new_seed, flac,
                          (d / f"artifacts_seed{new_seed}" / "score.abc").read_text(encoding="utf-8"), r)
        else:
            (work / f"round{r}_edited_plan.abc").write_text(new_text, encoding="utf-8")
            d = work / f"round{r}_revised"
            d.mkdir(exist_ok=True)
            flac = gen(pipe, spec, lyrics, best["seed"], new_text, d)
            plan_text = (d / f"artifacts_seed{best['seed']}" / "score.abc").read_text(encoding="utf-8")
            cand = assess(best["seed"], flac, plan_text, r)
        rounds.append(cand)
        if cand["ev"]["passed"] >= best["ev"]["passed"]:
            best = cand

    pipe.close()

    # 收尾
    final_flac = work / "best.flac"
    shutil.copy(best["flac"], final_flac)
    (work / "best_critic.md").write_text(format_report(f"round{best['round']}_seed{best['seed']}", best["ev"]),
                                         encoding="utf-8")
    (work / "rounds.json").write_text(json.dumps(
        [{k: (v if k != "ev" else {"passed": v["passed"], "total": v["total"],
                    "checks": v["checks"]}) for k, v in c.items()} for c in rounds],
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"DONE best={best['ev']['passed']}/{best['ev']['total']} -> {final_flac}", flush=True)


if __name__ == "__main__":
    main()
