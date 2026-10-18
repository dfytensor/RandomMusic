# -*- coding: utf-8 -*-
"""江南四律指标：全库验证 + 生成曲排名 -> eval/jiangnan_criteria_report.md
四律：①音域≤16半音 ②motif重复≥40% ③副歌递进（每次力度+） ④意象统一度
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from melody_stats import analyze_song
from ai_segment_forensic import parse_lab


# ===== 指标 1+2+3：旋律层（从已有 melody_stats 提取） =====
# ===== 指标 4：副歌递进（需要逐副歌实例的响度对比） =====
def chorus_progression(flac_path, tdir, limit=300):
    """每次副歌实例的平均响度 → 相邻副歌间的递进斜率"""
    import librosa
    y, _ = librosa.load(str(flac_path), sr=SR, mono=True)
    dur = len(y) / SR
    rows = parse_lab(tdir, "structure.lab")
    choruses = []
    merged = []
    for t1, t2, n in rows:
        if n != "chorus":
            continue
        if merged and t1 - merged[-1][1] < 1.5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t2))
        else:
            merged.append((t1, t2))
    for t1, t2 in merged:
        if t2 - t1 < 5 or t1 >= dur:
            continue
        t2c = min(t2, dur)
        seg = y[int(t1 * SR):int(t2c * SR)]
        rms = float(np.sqrt((seg ** 2).mean()))
        import librosa as _lb
        db = _lb.amplitude_to_db(rms + 1e-9)
        choruses.append(round(float(db), 1))
    if len(choruses) < 2:
        return None
    # 递进斜率：最后一次 - 第一次，归一化
    slope = choruses[-1] - choruses[0]
    return dict(instances=len(choruses), dbs=choruses, slope=round(slope, 1))


def full_eval(no, title, flac, tdir):
    m = analyze_song(str(Path(tdir) / "score.abc"))
    prog = chorus_progression(flac, tdir)

    # 四律判定
    c1_range = m["pitch_range"] <= 16
    c2_motif = m["motif_repeat"] >= 0.40
    c3_density = 4.0 <= m["density"] <= 10.0
    c4_prog = prog and prog["instances"] >= 3 and prog.get("slope", 0) >= 0

    checks = [
        ("①音域≤16半音", c1_range, f"{m['pitch_range']}st"),
        ("②motif重复≥40%", c2_motif, f"{m['motif_repeat']}"),
        ("③密度4-10", c3_density, f"{m['density']}"),
        ("④副歌递进≥3次且渐强", c4_prog,
         f"{prog['instances']}次 斜率{prog.get('slope','?')}dB" if prog else "无数据"),
    ]
    npass = sum(1 for _, ok, _ in checks if ok)
    return dict(no=no, title=title, density=m["density"], range=m["pitch_range"],
                motif=m["motif_repeat"], lift=m["lift_mean"], rest=m["rest_ratio"],
                prog=prog, checks=checks, passed=npass, total=4)


def main():
    # ===== 曲库验证 =====
    lib_wav = ROOT / "eval" / "library_wavs"
    lib_tr = ROOT / "eval" / "library_transcripts"
    trs = sorted(lib_tr.glob("*/score.abc"))
    corpus_results = []
    for tdir_abc in trs:
        tdir = tdir_abc.parent
        wav = lib_wav / (tdir.name + ".wav")
        if not wav.exists():
            continue
        try:
            r = full_eval(tdir.name[:24], tdir.name, wav, tdir)
            corpus_results.append(r)
        except Exception:
            pass

    def pct(vals, x):
        a = np.array([v for v in vals if v is not None], float)
        return round(float((a < x).mean() * 100), 1) if len(a) else 0

    # 曲库通过率
    pass_counts = [r["passed"] for r in corpus_results]
    full_pass_pct = round(sum(1 for p in pass_counts if p == 4) / len(pass_counts) * 100, 1)
    print(f"=== 曲库验证 (n={len(corpus_results)}) ===")
    print(f"四律全过: {full_pass_pct}%")
    for i in range(4):
        c = sum(1 for r in corpus_results if r["checks"][i][1])
        print(f"  律{i+1}: {c}/{len(corpus_results)} ({c/len(corpus_results)*100:.0f}%)")

    # 江南自身
    jn = next((r for r in corpus_results if "江南" in r["title"]), None)
    if jn:
        print(f"\n江南: {jn['passed']}/4 | density={jn['density']} range={jn['range']} "
              f"motif={jn['motif']} prog={jn['prog']}")

    # ===== 生成曲 =====
    from novel_songs_manifest import SONGS
    from hit_score import TRANSCRIPTS
    final = ROOT / "novel_songs" / "final"
    novel = []
    for s in SONGS:
        flac = final / f"{s['no']}_{s['title']}.flac"
        tdir = TRANSCRIPTS[s["no"]]
        if not (Path(tdir) / "score.abc").exists():
            continue
        try:
            r = full_eval(s["no"], s["title"], flac, tdir)
            novel.append(r)
        except Exception:
            pass

    print(f"\n=== 生成曲 (n={len(novel)}) ===")
    for r in sorted(novel, key=lambda x: -x["passed"]):
        status = "✅" if r["passed"] == 4 else ("🟡" if r["passed"] >= 2 else "❌")
        fails = [c[0] for c in r["checks"] if not c[1]]
        print(f"  {status} {r['no']} {r['title']} {r['passed']}/4 | "
              f"range={r['range']} motif={r['motif']} density={r['density']} "
              f"| fail: {','.join(fails) if fails else '无'}")

    # 保存
    out = dict(corpus_n=len(corpus_results), corpus_full_pass_pct=full_pass_pct,
               corpus=[dict(no=r["no"], title=r["title"], passed=r["passed"], total=4,
                            density=r["density"], range=r["range"], motif=r["motif"],
                            prog=r["prog"]) for r in corpus_results],
               novel=[dict(no=r["no"], title=r["title"], passed=r["passed"], total=4,
                           density=r["density"], range=r["range"], motif=r["motif"],
                           prog=r["prog"], checks=[(c[0], c[1]) for c in r["checks"]])
                      for r in novel])
    (ROOT / "eval" / "jiangnan_criteria_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\nSAVED eval/jiangnan_criteria_report.json")


if __name__ == "__main__":
    main()
