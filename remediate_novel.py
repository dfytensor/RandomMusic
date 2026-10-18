# -*- coding: utf-8 -*-
"""按 §10 差距清单修复 12 首：ABC 手术(密度/音域/抬升) → 重唱 → 母带 → 复测。
输出 novel_songs/v2/*_mastered.flac + remediate_report.md
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from abc_edit import op_transpose, op_fold_range, op_tie_merge
from melody_stats import analyze_song

SS2_PY = ROOT / ".venv-ss2" / "Scripts" / "python.exe"
TRANSCRIBE = ROOT / "transcribe.py"
OUT = ROOT / "novel_songs" / "v2"
OUT.mkdir(parents=True, exist_ok=True)


def eval_text(text):
    fd, tmp = tempfile.mkstemp(suffix=".abc")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        return analyze_song(tmp)
    finally:
        os.unlink(tmp)


def surgical_edit(text, m0):
    """按 §10 清单定向手术；每步重测，改善才保留。"""
    log, text_now = [], text
    m = m0
    # 1) 密度过高 -> 同音合并
    if m["density"] and m["density"] > 6.3:
        t2 = op_tie_merge(text_now, ["verse", "pre-chorus", "chorus"])
        m2 = eval_text(t2)
        if m2 and m2["density"] and abs(m2["density"] - 6.0) < abs(m["density"] - 6.0):
            text_now, m = t2, m2
            log.append(f"density {m0['density']}->{m['density']} (merge)")
    # 2) 副歌抬升过猛 -> 副歌 -2 度
    if m["lift_mean"] and m["lift_mean"] > 6:
        t2 = op_transpose(text_now, ["chorus"], -2)
        m2 = eval_text(t2)
        if m2 and m2["lift_mean"] is not None and m2["lift_mean"] < m["lift_mean"]:
            text_now, m = t2, m2
            log.append(f"lift {m0['lift_mean']}->{m['lift_mean']} (chorus-2)")
    # 3) 音域过宽 -> 用本曲 tessitura 折返极值
    if m["pitch_range"] and m["pitch_range"] > 20 and m.get("tessitura"):
        lo = max(45, m["tessitura"][0] - 1)
        hi = min(84, m["tessitura"][1] + 2)
        t2 = op_fold_range(text_now, lo, hi)
        m2 = eval_text(t2)
        if m2 and m2["pitch_range"] and m2["pitch_range"] < m["pitch_range"]:
            text_now, m = t2, m2
            log.append(f"range {m0['pitch_range']}->{m['pitch_range']} (fold {lo}-{hi})")
    return text_now, log, m


def master(wav_in, flac_out, shelf_g=0.4, shelf_hz=5500, widen=1.4):
    import librosa
    import soundfile as sf
    from scipy.signal import butter, sosfilt
    y, sr = librosa.load(str(wav_in), sr=48000, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    L, R = y[0], y[1]
    mid, side = (L + R) / 2, (L - R) / 2
    side = side * widen
    L, R = mid + side, mid - side
    sos = butter(2, shelf_hz / (sr / 2), btype="low", output="sos")
    for ch in (L, R):
        hp = ch - sosfilt(sos, ch)
        ch += shelf_g * hp
    peak = max(np.abs(L).max(), np.abs(R).max(), 1e-9)
    L, R = L / peak * 0.89, R / peak * 0.89
    sf.write(flac_out, np.stack([L, R], axis=1), sr, format="FLAC")


def transcribe_realized(flac_path, out_dir):
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


# MERT 一次加载
from scipy.io import wavfile
from transformers import AutoModel
from transformers import AutoModel as _AM  # noqa

mert = AutoModel.from_pretrained(str(ROOT / "models" / "MERT-v2-FullSong"),
                                 trust_remote_code=True, local_files_only=True).eval().cuda()
zc = np.load(ROOT / "eval" / "embeddings.npz", allow_pickle=True)
C = zc["embs"][[i for i, g in enumerate(zc["groups"]) if g == "corpus"]]
MU = C.mean(0)
_X = C - MU
_cov = _X.T @ _X / (len(C) - 1)
_lam = 0.1 * np.trace(_cov) / len(MU)
INV = np.linalg.inv(0.9 * _cov + _lam * np.eye(len(MU)))


def naturalness(wav_path):
    import librosa
    import soundfile as sf
    data, sr0 = sf.read(str(wav_path), always_2d=True)
    data = data.mean(axis=1)
    if sr0 != 24000:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr0, target_sr=24000)
    w = torch.from_numpy(data.astype(np.float32))[None, :].cuda()
    with torch.no_grad():
        e = mert(w).last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
    maha = float(np.sqrt(max((e - MU) @ INV @ (e - MU), 0)))
    cos = (C @ e) / (np.linalg.norm(C, axis=1) * np.linalg.norm(e) + 1e-9)
    return maha, float(np.sort(cos)[-5:].mean())


def main():
    # v1 基线自然度
    zn = np.load(ROOT / "eval" / "novel_embeddings.npz", allow_pickle=True)
    base_nat = {}
    for i, n in enumerate(zn["names"]):
        e = zn["embs"][i]
        maha = float(np.sqrt(max((e - MU) @ INV @ (e - MU), 0)))
        cos = (C @ e) / (np.linalg.norm(C, axis=1) * np.linalg.norm(e) + 1e-9)
        base_nat[str(n)] = (maha, float(np.sort(cos)[-5:].mean()))

    from yue2 import YuE2Pipeline
    rows = []
    t0 = time.time()
    for i, s in enumerate(SONGS):
        stem = f"novel_{s['no']}_{s['title']}"
        src_abc = ROOT / "eval" / "novel_transcripts" / stem / "score.abc"
        if not src_abc.exists():
            print(f"skip {s['title']} (no v1 transcript)")
            continue
        work = OUT / stem
        work.mkdir(parents=True, exist_ok=True)
        v1_text = src_abc.read_text(encoding="utf-8")
        m0 = analyze_song(str(src_abc))
        edited, log, m1 = surgical_edit(v1_text, m0)
        (work / "edited.abc").write_text(edited, encoding="utf-8")
        print(f"[{i+1}/{len(SONGS)}] {s['title']} edits={log}", flush=True)

        flac_raw = work / "raw.flac"
        if not flac_raw.exists():
            pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                                vae=str(ROOT / "models" / "YuE2-Vae"),
                                                device="cuda", progress=False)
            try:
                song = pipe(style=s["style"], lyrics=s["lyrics"], abc=edited, cot="melody",
                            seed=s["seed"])
                song.save(str(flac_raw))
            finally:
                pipe.close()

        flac_m = work / f"{s['no']}_{s['title']}_mastered.flac"
        if not flac_m.exists():
            master(flac_raw, flac_m)

        abc2 = transcribe_realized(flac_m, work / "verify")
        m2r = analyze_song(abc2)
        maha, cos5 = naturalness(work / "verify" / "song.wav")
        b_maha, b_cos = base_nat.get(stem, (None, None))
        rows.append(dict(no=s["no"], title=s["title"],
                         density=[m0["density"], m2r["density"]],
                         range=[m0["pitch_range"], m2r["pitch_range"]],
                         lift=[m0["lift_mean"], m2r["lift_mean"]],
                         motif=[m0["motif_repeat"], m2r["motif_repeat"]],
                         cos5=[b_cos, cos5], maha=[b_maha, maha],
                         edits=log))
        print(f"    done density {m0['density']}->{m2r['density']} range {m0['pitch_range']}->"
              f"{m2r['pitch_range']} lift {m0['lift_mean']}->{m2r['lift_mean']} "
              f"cos {b_cos}->{cos5:.3f} elapsed={time.time()-t0:.0f}s", flush=True)

    # 报告
    lines = ["# 《现贷修仙传》v2 修复报告（ABC 手术 + 重唱 + 母带）", "",
             "| # | 曲目 | 密度 v1→v2 | 音域 v1→v2 | 抬升 v1→v2 | 自然度cos v1→v2 | 手术 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda x: "-" if x is None else (round(x, 2) if isinstance(x, float) else x)
        lines.append(f"| {r['no']} | {r['title']} | {f(r['density'][0])}→{f(r['density'][1])} "
                     f"| {f(r['range'][0])}→{f(r['range'][1])} | {f(r['lift'][0])}→{f(r['lift'][1])} "
                     f"| {r['cos5'][0]:.3f}→{r['cos5'][1]:.3f} | {len(r['edits'])} 项 |")
    (OUT / "remediate_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
