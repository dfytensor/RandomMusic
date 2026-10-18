# -*- coding: utf-8 -*-
"""修复截断：08/11/12 用全曲转谱重做手术版，与现有 final 三版择优。"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from abc_edit import op_transpose, op_fold_range, op_tie_merge
from melody_stats import analyze_song
from ai_flavor_check import ai_score, load_reference
from ai_segment_forensic import analyze_segment, parse_lab
from remediate_novel import master, naturalness, transcribe_realized

SS2_PY = ROOT / ".venv-ss2" / "Scripts" / "python.exe"
REF = load_reference()
FIX = ["08", "11", "12"]
TARGET = {"density": 6.0, "lift_max": 6.0, "range_max": 20}


def eval_text(text):
    fd, tmp = tempfile.mkstemp(suffix=".abc")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        return analyze_song(tmp)
    finally:
        os.unlink(tmp)


def surgical(text, m0):
    log, cur, m = [], text, m0
    if m["density"] and m["density"] > TARGET["density"] + 0.3:
        t2 = op_tie_merge(cur, ["verse", "pre-chorus", "chorus"])
        m2 = eval_text(t2)
        if m2 and m2["density"] and abs(m2["density"] - 6) < abs(m["density"] - 6):
            cur, m = t2, m2
            log.append(f"density->{m['density']}")
    if m["lift_mean"] and m["lift_mean"] > TARGET["lift_max"]:
        t2 = op_transpose(cur, ["chorus"], -2)
        m2 = eval_text(t2)
        if m2 and m2["lift_mean"] and m2["lift_mean"] < m["lift_mean"]:
            cur, m = t2, m2
            log.append(f"lift->{m['lift_mean']}")
    if m["pitch_range"] and m["pitch_range"] > TARGET["range_max"] and m.get("tessitura"):
        lo, hi = max(45, m["tessitura"][0] - 1), min(84, m["tessitura"][1] + 2)
        t2 = op_fold_range(cur, lo, hi)
        m2 = eval_text(t2)
        if m2 and m2["pitch_range"] and m2["pitch_range"] < m["pitch_range"]:
            cur, m = t2, m2
            log.append(f"range->{m['pitch_range']}")
    return cur, log, m


def robust_span(tdir, dur):
    rows = parse_lab(tdir, "structure.lab")
    merged, order = {}, []
    for t1, t2, n in rows:
        n = n.lower()
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


def main():
    from yue2 import YuE2Pipeline
    report = []
    for no in FIX:
        s = next(x for x in SONGS if x["no"] == no)
        title = s["title"]
        stem = f"novel_{no}_{title}"
        v1_flac = ROOT / "novel_songs" / f"{no}_{title}.flac"
        final_flac = ROOT / "novel_songs" / "final" / f"{no}_{title}.flac"
        work = ROOT / "novel_songs" / "v2" / (stem + "_full")
        work.mkdir(parents=True, exist_ok=True)

        # 1) 全曲转谱
        full_wav = work / "full.wav"
        if not full_wav.exists():
            import librosa
            y, _ = librosa.load(str(v1_flac), sr=24000, mono=True, duration=300)
            sf.write(full_wav, y, 24000, subtype="PCM_16")
        ftdir = work / "transcript_full"
        if not (ftdir / "score.abc").exists():
            env = dict(os.environ)
            env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1")
            subprocess.run([str(SS2_PY), str(ROOT / "transcribe.py"), str(full_wav), str(ftdir)],
                           check=True, capture_output=True, env=env)
        full_abc = (ftdir / "score.abc").read_text(encoding="utf-8")
        m_full = analyze_song(str(ftdir / "score.abc"))
        print(f"\n=== {no} {title} 全曲转谱: dur={m_full['total_bars']}bars "
              f"density={m_full['density']} range={m_full['pitch_range']} lift={m_full['lift_mean']}",
              flush=True)

        # 2) 手术
        edited, log, m1 = surgical(full_abc, m_full)
        (work / "edited_full.abc").write_text(edited, encoding="utf-8")
        print(f"    edits={log}", flush=True)

        # 3) 重渲染 + 母带
        raw = work / "raw.flac"
        if not raw.exists():
            pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                                vae=str(ROOT / "models" / "YuE2-Vae"),
                                                device="cuda", progress=False)
            try:
                song = pipe(style=s["style"], lyrics=s["lyrics"], abc=edited, cot="melody",
                            seed=s["seed"] + 500)
                song.save(str(raw))
            finally:
                pipe.close()
        cand = work / "cand_mastered.flac"
        if not cand.exists():
            master(raw, cand)
        dur_c = (lambda i: i.frames / i.samplerate)(__import__("soundfile").info(str(cand)))
        v1_dur = (lambda i: i.frames / i.samplerate)(__import__("soundfile").info(str(v1_flac)))
        truncated = dur_c < v1_dur * 0.85
        print(f"    渲染时长 {dur_c:.0f}s (v1={v1_dur:.0f}s){' ❗仍截断' if truncated else ''}", flush=True)

        # 4) 指标：AI味 + 自然度
        tdir_c = work / "verify"
        transcribe_realized(cand, tdir_c)
        sp = robust_span(tdir_c, dur_c)
        m_c = analyze_segment(str(cand), tdir_c, sp[0], sp[1], want_symbolic=True)
        ai_c, _ = ai_score(m_c, REF)
        maha_c, cos_c = naturalness(cand)

        # 现役 final 的指标
        ai_f, _ = ai_score(*(() ,)) if False else (None, None)
        ftdir = ROOT / "eval" / "novel_transcripts" / stem
        cur_pick_dir = ftdir if ftdir.exists() else work / "verify_old"
        # final 可能是 v2 截断版：直接量 final 音频
        spf = robust_span_from_dur = robust_span(final_flac.parent / "x", dur_c)  # 占位，不用
        # 用 final 自身全曲转谱（若有）否则以中段分析
        final_tdir = ROOT / "novel_songs" / "v2" / stem / "verify" / "realized"
        spf = robust_span(final_tdir, dur_c) if (final_tdir / "structure.lab").exists() else (30, 60)
        m_f = analyze_segment(str(final_flac), final_tdir, spf[0], spf[1], want_symbolic=True)
        ai_f, _ = ai_score(m_f, REF)
        maha_f, cos_f = naturalness(final_flac)

        # 5) 择优：截断的一票否决；否则 AI味+自然度
        cand_ok = (not truncated)
        choose_cand = cand_ok and (ai_c < ai_f) and (cos_c >= cos_f - 0.012)
        if choose_cand:
            shutil.copy(cand, final_flac)
            choice = "new_full_surgery"
        else:
            choice = "kept_current"
        print(f"    对比: cand AI味={ai_c} cos={cos_c:.3f} | final AI味={ai_f} cos={cos_f:.3f} -> {choice}",
              flush=True)
        report.append(dict(no=no, title=title, cand_dur=round(dur_c), v1_dur=round(v1_dur),
                           truncated=truncated, edits=log, ai_cand=ai_c, ai_final=ai_f,
                           cos_cand=round(cos_c, 3), cos_final=round(cos_f, 3), chosen=choice))
    (ROOT / "novel_songs" / "v2" / "refix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
