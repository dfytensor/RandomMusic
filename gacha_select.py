# -*- coding: utf-8 -*-
"""抽卡选优工具：同一首歌 N 个种子 → 生成 → 母带 → 转谱 → AI味/自然度/技术三指标 → 择优入 final。
用法: python gacha_select.py <no> [--seeds 33191,34291,35391]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from remediate_novel import master, naturalness, transcribe_realized
from ai_flavor_check import score_song as flavor_score, load_reference

SS2_PY = ROOT / ".venv-ss2" / "Scripts" / "python.exe"
FINAL = ROOT / "novel_songs" / "final"


def tech_check(flac):
    out = flac.parent / (flac.stem + "_tech.md")
    subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"),
                    str(ROOT / "reference_free_check.py"), str(flac), "--out", str(out)],
                   check=True, capture_output=True)
    txt = out.read_text(encoding="utf-8")
    d = json.loads(txt.split("json")[1].split("```")[0])[0]
    return d["pass"], d["total"], d["duration_s"]


def find_lab(tdir):
    tdir = Path(tdir)
    for c in (tdir / "structure.lab", tdir / "realized" / "structure.lab"):
        if c.exists():
            return c.parent
    return None


def main():
    no = sys.argv[1]
    seeds = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")] \
        if "--seeds" in sys.argv else []
    s = next(x for x in SONGS if x["no"] == no)
    title = s["title"]
    final_flac = FINAL / f"{no}_{title}.flac"
    REF = load_reference()

    # 现役版为基准候选（现役渲染可能与旧转谱不一致 -> 一律重转谱）
    cands = []
    if final_flac.exists():
        maha, cos5 = naturalness(final_flac)
        tdir = FINAL / f"{no}_gacha_t"
        if not (tdir / "realized" / "structure.lab").exists():
            transcribe_realized(final_flac, tdir)
        sc, _ = flavor_score(final_flac, tdir / "realized", REF)
        p, t, dur = tech_check(final_flac)
        cands.append(dict(tag="incumbent", seed=None, flac=str(final_flac), ai=sc,
                          maha=round(maha), cos5=round(cos5, 3), tech=f"{p}/{t}", dur=round(dur)))
        print(f"incumbent: AI味={sc} cos={cos5:.3f} tech={p}/{t}", flush=True)

    from yue2 import YuE2Pipeline
    import torch
    for seed in seeds:
        work = ROOT / "novel_songs" / "gacha" / no / str(seed)
        work.mkdir(parents=True, exist_ok=True)
        raw = work / "raw.flac"
        mastered = work / "mastered.flac"
        if not (raw.exists() and mastered.exists()):
            ok = False
            for attempt in (1, 2):
                try:
                    torch.cuda.empty_cache()
                    r = subprocess.run([sys.executable, str(ROOT / "render_one.py"), no, str(seed)],
                                       capture_output=True, text=True)
                    ok = r.returncode == 0 and "OK" in r.stdout
                except Exception as e:
                    print(f"  seed{seed} attempt{attempt} ex: {type(e).__name__}", flush=True)
                if ok:
                    break
                print(f"  seed{seed} attempt{attempt} fail", flush=True)
                time.sleep(5)
            if not ok:
                print(f"  seed{seed} GIVE UP", flush=True)
                continue
        mastered = work / "mastered.flac"
        if not mastered.exists():
            master(raw, mastered)
        tdir = work / "verify"
        if not (tdir / "realized" / "structure.lab").exists():
            transcribe_realized(mastered, tdir)
        tdir = tdir / "realized"
        sc, _ = flavor_score(mastered, tdir, REF)
        maha, cos5 = naturalness(mastered)
        p, tt, dur = tech_check(mastered)
        cands.append(dict(tag=f"seed{seed}", seed=seed, flac=str(mastered), ai=sc,
                          maha=round(maha), cos5=round(cos5, 3), tech=f"{p}/{tt}", dur=round(dur)))
        print(f"  seed{seed}: AI味={sc} maha={maha:.0f} cos={cos5:.3f} tech={p}/{tt}", flush=True)

    # 择优：技术全过为先，再按 AI味升序，平手比马氏
    ok = [c for c in cands if c["tech"].startswith(tuple(str(i) for i in range(10))) and
          c["tech"].split("/")[0] == c["tech"].split("/")[1]]
    pool = ok if ok else cands
    pool.sort(key=lambda c: (c["ai"], c["maha"]))
    best = pool[0]
    if best["tag"] != "incumbent":
        shutil_copy_dst = final_flac
        import shutil
        shutil.copy(best["flac"], shutil_copy_dst)
    print(f"\n=== WINNER: {best['tag']} AI味={best['ai']} maha={best['maha']} "
          f"cos={best['cos5']} tech={best['tech']} ===", flush=True)
    (FINAL / f"{no}_gacha_report.json").write_text(
        json.dumps(cands, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
