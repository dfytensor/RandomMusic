# -*- coding: utf-8 -*-
"""《现贷修仙传》原声批量生成：每首独立管线(防显存累积) + OOM 重试。"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

OUT = ROOT / "novel_songs"
OUT.mkdir(exist_ok=True)


def gen_one(s):
    from yue2 import YuE2Pipeline
    pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                        vae=str(ROOT / "models" / "YuE2-Vae"),
                                        device="cuda", progress=False)
    try:
        return pipe(style=s["style"], lyrics=s["lyrics"], cot="full", seed=s["seed"])
    finally:
        pipe.close()


t0 = time.time()
results = []
for i, s in enumerate(SONGS):
    flac = OUT / f"{s['no']}_{s['title']}.flac"
    if flac.exists():
        print(f"[{i+1}/{len(SONGS)}] skip {s['title']} (exists)", flush=True)
        results.append(str(flac))
        continue
    print(f"[{i+1}/{len(SONGS)}] {s['title']} ({s['kind']}, {s['bpm']}BPM {s['key']} {s['mode']}) gen ...",
          flush=True)
    song = None
    for attempt in (1, 2):
        try:
            import torch
            torch.cuda.empty_cache()
            song = gen_one(s)
            break
        except Exception as e:
            name = type(e).__name__
            print(f"    attempt{attempt} failed: {name}: {str(e)[:120]}", flush=True)
            import torch
            torch.cuda.empty_cache()
            time.sleep(5)
    if song is None:
        print(f"    GIVE UP {s['title']}", flush=True)
        continue
    song.save(str(flac))
    (OUT / f"{s['no']}_{s['title']}_spec.json").write_text(
        json.dumps({k: v for k, v in s.items() if k != "lyrics"}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (OUT / f"{s['no']}_{s['title']}_lyrics.txt").write_text(s["lyrics"], encoding="utf-8")
    results.append(str(flac))
    print(f"    saved {flac.name} elapsed={time.time()-t0:.0f}s", flush=True)

ok_results = [r for r in results if r]
if ok_results:
    print("running technical checks ...", flush=True)
    import subprocess
    subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"),
                    str(ROOT / "reference_free_check.py"), *ok_results,
                    "--out", str(OUT / "tech_report.md")], check=True)
print(f"ALL DONE total={time.time()-t0:.0f}s", flush=True)
