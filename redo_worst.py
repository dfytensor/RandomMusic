# -*- coding: utf-8 -*-
"""重唱 AI 味最差的 4 首（新 seed）→ AI味分+自然度双指标择优替换 final。"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from ai_flavor_check import score_song, load_reference, PICK_DIRS
from remediate_novel import master, naturalness, transcribe_realized


def robust_span(tdir, dur):
    """取最长 chorus；过短则最长 verse；再不行取中段。返回 (t1, t2)。"""
    rows = []
    p = Path(tdir) / "structure.lab"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            q = line.split("\t")
            if len(q) >= 3:
                rows.append((float(q[0]), float(q[1]), q[2].strip().lower()))
    merged, order = {}, []
    for t1, t2, n in rows:
        if n == "silence":
            continue
        if n not in merged:
            merged[n] = [t1, t2]
            order.append(n)
        else:
            merged[n][1] = max(merged[n][1], t2)
    for name in ("chorus", "verse"):
        if name in merged:
            t1, t2 = merged[name]
            if t2 - t1 >= 6:
                t2 = min(t2, t1 + 30)
                return t1, max(t2, t1 + 8)
    t1 = max(10.0, dur / 3)
    return t1, t1 + 20


REDO = ["06", "05", "04", "08"]
NEW_SEED = {"06": 39006, "05": 39005, "04": 39004, "08": 39008}
REF = load_reference()
FINAL = ROOT / "novel_songs" / "final"

from yue2 import YuE2Pipeline

results = []
for i, no in enumerate(REDO):
    s = next(x for x in SONGS if x["no"] == no)
    title = s["title"]
    final_flac = FINAL / f"{no}_{title}.flac"
    old_tdir = PICK_DIRS[no]
    old_sc, _ = score_song(final_flac, old_tdir, REF)
    old_maha, old_cos = naturalness(final_flac)
    print(f"\n=== {no} {title} old: AI味={old_sc} cos={old_cos:.3f} ===", flush=True)

    work = ROOT / "novel_songs" / "redo" / no
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "raw.flac"
    if not raw.exists():
        pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                            vae=str(ROOT / "models" / "YuE2-Vae"),
                                            device="cuda", progress=False)
        try:
            song = pipe(style=s["style"], lyrics=s["lyrics"], cot="full", seed=NEW_SEED[no])
            song.save(str(raw))
        finally:
            pipe.close()
    mastered = work / "mastered.flac"
    if not mastered.exists():
        master(raw, mastered)

    new_tdir = work / "verify"
    abc2 = transcribe_realized(mastered, new_tdir)
    import soundfile as sf
    dur = sf.info(str(mastered)).frames / sf.info(str(mastered)).samplerate
    sp = robust_span(new_tdir, dur)
    from ai_segment_forensic import analyze_segment
    m_new = analyze_segment(mastered, new_tdir, sp[0], sp[1], want_symbolic=True)
    new_sc, _ = score_song(mastered, new_tdir, REF)
    if new_sc is None:
        from ai_flavor_check import ai_score
        new_sc, _ = ai_score(m_new, REF)
    new_maha, new_cos = naturalness(mastered)
    print(f"    new: AI味={new_sc} cos={new_cos:.3f}", flush=True)

    better = (new_sc < old_sc) and (new_cos >= old_cos - 0.012)
    if better:
        shutil.copy(mastered, final_flac)
        shutil.copytree(new_tdir, ROOT / "eval" / "novel_transcripts_redo" / f"novel_{no}_{title}",
                        dirs_exist_ok=True)
        PICK_DIRS[no] = ROOT / "eval" / "novel_transcripts_redo" / f"novel_{no}_{title}"
    results.append(dict(no=no, title=title, old_ai=old_sc, new_ai=new_sc,
                        old_cos=round(old_cos, 3), new_cos=round(new_cos, 3),
                        chosen=("new" if better else "old")))
    print(f"    -> {'REPLACED with new' if better else 'kept old'}", flush=True)

(FINAL / "redo_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=1), flush=True)
print("DONE", flush=True)
