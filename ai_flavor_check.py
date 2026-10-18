# -*- coding: utf-8 -*-
"""AI 味打分器：四族指纹（唱太满/演唱不稳/动态失控/循环周期）对照曲库真副歌分布。
用法:
  python ai_flavor_check.py                      # 对 final 12 首打分排序
  python ai_flavor_check.py <flac> <transcript_dir>   # 单曲打分
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from ai_segment_forensic import analyze_segment, parse_lab, first_span, f0_metrics, dynamics_metrics, loopiness
from novel_songs_manifest import SONGS

# (键, AI方向(+1=越高越AI), 权重)  权重按全库标定(n=463)效应量分级：A(d>=0.8)=1.0, B(0.5-0.8)=0.6-0.7, C=0.15-0.4
METRICS = [
    ("dyn_range_db", +1, 1.0),        # A  d=+0.97
    ("crest_db", +1, 1.0),            # A  d=+1.44（波形层）
    ("low_attack_ms", +1, 0.8),       # A  d=+0.81（波形层）
    ("onset_iv_cv", +1, 0.8),         # A  d=+0.86（波形层）
    ("voiced_ratio", +1, 0.7),        # B  d=+0.67
    ("intra_note_drift_cents", +1, 0.6),  # B d=+0.61
    ("macro_dyn_std_db", +1, 0.35),   # C  d=+0.44
    ("beat_R", +1, 0.3),              # C  d=+0.37
    ("n_vocal_runs", +1, 0.35),       # C  d=+0.35
    ("notes", +1, 0.35),              # C  d≈+0.35
    ("vibrato_rate_hz", -1, 0.2),     # C  d=-0.17
    ("vibrato_extent_cents", +1, 0.15),  # C d=+0.08(无效应)
    ("loop_lag_s", +1, 0.15),         # C  d=-0.04
]

# final 12 首的择优来源 -> 对应实录转谱
PICK_DIRS = {
    "01": ROOT / "eval/novel_transcripts/novel_01_欠天地一笔",
    "02": ROOT / "eval/novel_transcripts/novel_02_平账人",
    "03": ROOT / "eval/novel_transcripts/novel_03_白户",
    "04": ROOT / "novel_songs/v2/novel_04_剁手魔/verify/realized",
    "05": ROOT / "eval/novel_transcripts/novel_05_心疼说成省钱",
    "06": ROOT / "novel_songs/v2/novel_06_第320代记账人/verify/realized",
    "07": ROOT / "novel_songs/v2/novel_07_规矩/verify/realized",
    "08": ROOT / "novel_songs/v2/novel_08_月供日/verify/realized",
    "09": ROOT / "eval/novel_transcripts/novel_09_断供",
    "10": ROOT / "eval/novel_transcripts/novel_10_糕与药包",
    "11": ROOT / "novel_songs/v2/novel_11_烂尾劫/verify/realized",
    "12": ROOT / "novel_songs/v2/novel_12_新契/verify/realized",
}


def load_reference():
    d = json.loads((ROOT / "eval" / "ai_segment_02.json").read_text(encoding="utf-8"))
    return d["corpus_chorus"]


def ai_score(metrics, ref):
    """返回 (总分, 各指标z明细)。z = 方向*(x-中位)/(0.7413*IQR)，只累计正向偏离。"""
    total, detail = 0.0, {}
    for key, sign, w in METRICS:
        vals = np.array([r[key] for r in ref if r.get(key) is not None], float)
        x = metrics.get(key)
        if len(vals) < 5 or x is None:
            detail[key] = None
            continue
        med = float(np.median(vals))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25)) + 1e-9
        z = sign * (x - med) / (0.7413 * iqr)
        pos = max(0.0, z)
        total += w * min(pos, 6.0)
        detail[key] = dict(x=round(float(x), 3), med=round(med, 3), z=round(float(z), 2),
                           weighted=round(w * min(pos, 6.0), 2))
    return round(total, 2), detail


def score_song(flac, tdir, ref, max_len=25):
    """三层指标：AI味指纹(音频) + 波形三指标 + 唱拍。返回 (score, info)。"""
    import librosa as _lb
    from scipy.signal import butter, sosfilt
    tdir = Path(tdir)
    abc = tdir / "score.abc"
    if not abc.exists():
        abc = tdir / "realized" / "score.abc"
    if not abc.exists():
        return None, None
    rows = parse_lab(tdir, "structure.lab")
    span = first_span(rows, "chorus", 1e9)
    if not span or span[1] - span[0] < 6:
        return None, None
    t2 = min(span[1], span[0] + max_len)

    y_seg, _ = _lb.load(str(flac), sr=SR, mono=True, offset=span[0], duration=t2 - span[0])
    f = f0_metrics(y_seg)
    f.update(dynamics_metrics(y_seg))
    f.update(loopiness(y_seg))
    # 唱拍锁死度
    tempo, beats = _lb.beat.beat_track(y=y_seg, sr=SR, units="time", hop_length=512)
    beat_t = np.atleast_1d(beats)
    if len(beat_t) >= 8:
        period = 60.0 / float(np.atleast_1d(tempo)[0])
        sos_b = butter(4, [300 / (SR / 2), 3400 / (SR / 2)], btype="band", output="sos")
        voc = sosfilt(sos_b, y_seg)
        ons_v = _lb.onset.onset_detect(y=voc, sr=SR, hop_length=256, units="time",
                                       delta=0.06, backtrack=False)
        if len(ons_v) >= 10:
            phases = np.array([((o - beat_t[0]) % period) / period for o in ons_v])
            f["beat_R"] = round(float(abs(np.exp(1j * phases * 2 * np.pi).mean())), 3)
        else:
            f["beat_R"] = None
    else:
        f["beat_R"] = None
    # 波形层
    rms_g = float(np.sqrt((y_seg ** 2).mean()) + 1e-9)
    f["crest_db"] = round(float(20 * np.log10(np.abs(y_seg).max() / rms_g + 1e-9)), 2)
    sos = butter(4, 150 / (SR / 2), btype="low", output="sos")
    low = sosfilt(sos, y_seg)
    lr = _lb.feature.rms(y=low, frame_length=1024, hop_length=256)[0]
    thr = lr.mean() + lr.std()
    atks, rise, st = [], False, 0
    for i, v in enumerate(lr):
        if not rise and v > thr:
            rise, st = True, i
        elif rise and v < thr:
            if i > st:
                atks.append((i - st) * 256 / SR * 1000)
            rise = False
    f["low_attack_ms"] = round(float(np.median(atks)), 1) if len(atks) >= 3 else None
    ons = _lb.onset.onset_detect(y=y_seg, sr=SR, hop_length=256, units="time",
                                 delta=0.06, backtrack=False)
    iv = np.diff(ons)
    f["onset_iv_cv"] = round(float(iv.std() / iv.mean()), 2) if len(iv) > 5 else None

    s, detail = ai_score(f, ref)
    return s, dict(metrics=f, detail=detail, chorus_span=span)


def main():
    ref = load_reference()
    rows = []
    for s in SONGS:
        flac = ROOT / "novel_songs" / "final" / f"{s['no']}_{s['title']}.flac"
        tdir = PICK_DIRS[s["no"]]
        if not flac.exists() or not tdir.exists():
            print(f"skip {s['no']} {s['title']}")
            continue
        sc, info = score_song(flac, tdir, ref)
        rows.append((sc, s["no"], s["title"], info))
    rows.sort(key=lambda r: -r[0])
    print(f"{'排名':>4s} {'AI味分':>7s}  曲目")
    for rank, (sc, no, title, info) in enumerate(rows, 1):
        bar = "█" * int(min(sc, 20))
        print(f"{rank:>4d} {sc:7.2f}  {no} {title} {bar}")
    print("\n最差曲目明细:")
    for sc, no, title, info in rows[:3]:
        print(f"\n{no} {title}:")
        for k, d in info["detail"].items():
            if d and d["z"] > 0.5:
                print(f"  {k}: x={d['x']} (真曲中位 {d['med']}, z=+{d['z']})")
    (ROOT / "eval" / "ai_flavor_rank.json").write_text(
        json.dumps([dict(no=no, title=title, score=sc, metrics=info["metrics"])
                    for sc, no, title, info in rows], ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("\nSAVED eval/ai_flavor_rank.json")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        ref = load_reference()
        sc, info = score_song(sys.argv[1], sys.argv[2], ref)
        print(json.dumps(dict(score=sc, detail=info["detail"]), ensure_ascii=False, indent=1))
    else:
        main()
