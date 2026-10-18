# -*- coding: utf-8 -*-
"""旋律学/和声/节奏 新指标族：设计 → 全库采集 → 热度相关性验证
新指标：
1. pitch_class_entropy  音高分布熵（越低越聚焦=越易记）
2. interval_entropy     音程分布熵（越低越重复=越易哼）
3. direction_changes    旋律方向变化次数/小节（转折越多越有张力）
4. chromatic_fraction   离调音占比（越高越有色彩）
5. chord_change_rate    和弦变化率/小节（和声节奏）
6. syncopation         切分音占比（律动活性）
7. tonic_resolution    主音解决率（乐句结尾归主音的比例）
8. phrase_arch         乐句拱形度（音高轮廓的二次拟合R²）
"""
import json, math, re, sys, warnings
from pathlib import Path
import numpy as np
import librosa
from scipy import stats as sps
from scipy.stats import entropy

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from ai_segment_forensic import parse_lab, first_span

# ===== 热度函数 =====
MEGA_ARTISTS = {"周杰伦","林俊杰","五月天","张学友","刘德华","邓紫棋","陈奕迅","王菲",
                "张惠妹","孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","Tank","胡歌"}
KNOWN_ARTISTS = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
                 "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA = {"青花瓷","东风破","菊花台","发如雪","夜曲","七里香","江南","曹操","一千年以后",
        "隐形的翅膀","偏爱","死了都要爱","离歌","突然的自我","知足","温柔","倔强",
        "Super Star","不想长大","恋人未满","遇见","天黑黑","绿光"}

def tier_of(name):
    parts = re.split(r"[-_—]", name)
    artist = parts[0].strip() if parts else ""
    if artist in MEGA_ARTISTS: return 3
    for h in MEGA:
        if h in name: return 3
    if artist in KNOWN_ARTISTS: return 2
    return 1

# ===== ABC 解析 =====
def parse_vocal_notes(abc_path):
    """解析 ABC 中 Vocal 声部的 (midi, onset_sec, dur_sec) 列表"""
    from ai_segment_forensic import parse_abc
    meta, vocal, secs = parse_abc(str(abc_path))
    return meta, vocal, secs


def pitch_class_entropy(vocal_notes):
    """音高类分布熵（0-1 归一化）"""
    from collections import Counter
    pcs = [int(m) % 12 for _, m, _ in vocal_notes]
    if not pcs: return None
    c = np.bincount(pcs, minlength=12).astype(float)
    p = c / c.sum()
    p = p[p > 0]
    h = -sum(x * np.log2(x) for x in p)
    return round(h / np.log2(12), 3)  # 归一到 0-1


def interval_entropy(vocal_notes):
    """音程分布熵"""
    from collections import Counter
    notes = sorted(vocal_notes)
    if len(notes) < 2: return None
    ivs = [int(b[1] - a[1]) % 12 for a, b in zip(notes, notes[1:])]
    if not ivs: return None
    c = np.bincount(ivs, minlength=12).astype(float)
    p = c / c.sum()
    p = p[p > 0]
    h = -sum(x * np.log2(x) for x in p)
    return round(h / np.log2(12), 3)


def direction_changes(vocal_notes, total_bars):
    """旋律方向变化次数 / 小节"""
    pitches = [m for _, m, _ in vocal_notes]
    if len(pitches) < 3: return None
    dirs = np.sign(np.diff(pitches))
    dirs = dirs[dirs != 0]
    if len(dirs) < 2: return None
    changes = np.sum(np.diff(dirs) != 0)
    return round(changes / max(total_bars, 1), 2)


def syncopation(abc_path, vocal_notes, chorus_t1):
    """人声起音在反拍上的占比（基于节拍网格）"""
    import librosa
    flac_dir = Path(abc_path).parent
    # 找音频
    for ext in (".wav", ".flac"):
        for f in Path(flac_dir).parent.glob(f"*{Path(abc_path).stem[-4:]}*{ext}"):
            try:
                y, sr = librosa.load(str(f), sr=22050, mono=True, duration=300)
            except:
                continue
    return None  # 暂不实现，需要音频文件定位


def chromatic_fraction(vocal_notes, tonic_pc, minor):
    """离调音占比"""
    if minor:
        diatonic = {0, 2, 3, 5, 7, 8, 10}
    else:
        diatonic = {0, 2, 4, 5, 7, 9, 11}
    total = len(vocal_notes)
    if total == 0: return None
    chrom = sum(1 for _, m, _ in vocal_notes
                if (int(m) - tonic_pc) % 12 not in diatonic)
    return round(chrom / total, 3)


def tonic_resolution(vocal_notes, tonic_pc):
    """乐句末音落在主音的比率"""
    # 按 rest_ratio 间隔分段
    phrases = []
    cur = []
    for onset, midi, dur in vocal_notes:
        if dur > 0.5:  # 长音=乐句结束
            if cur:
                cur.append((onset, midi, dur))
                phrases.append(cur)
                cur = []
            else:
                phrases.append([(onset, midi, dur)])
            continue
        cur.append((onset, midi, dur))
    if cur:
        phrases.append(cur)
    if len(phrases) < 3:
        return None
    ends = [ph[-1][1] for ph in phrases if ph]
    on_tonic = sum(1 for m in ends if (int(m) - tonic_pc) % 12 == 0)
    return round(on_tonic / len(ends), 3)


def phrase_arch(vocal_notes):
    """乐句拱形度：音高轮廓的二次拟合R²"""
    pitches = [m for _, m, _ in vocal_notes]
    if len(pitches) < 5:
        return None
    x = np.arange(len(pitches), dtype=float)
    try:
        coeffs = np.polyfit(x, pitches, 2)
        pred = np.polyval(coeffs, x)
        ss_res = np.sum((np.array(pitches) - pred) ** 2)
        ss_tot = np.sum((np.array(pitches) - np.mean(pitches)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-9)
        return round(r2, 3)
    except Exception:
        return None


def chord_change_rate(tdir, total_bars):
    """和弦变化率（chord.lab 中不同和弦数 / 总小节数）"""
    p = Path(tdir) / "chord.lab"
    if not p.exists():
        return None
    chords = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.split("\t")
        if len(q) >= 3 and q[2].strip() not in ("N", "X"):
            chords.add(q[2].strip())
    return round(len(chords) / max(total_bars, 1), 2)


# ===== 主流程 =====
def main():
    lib_wav = ROOT / "eval" / "library_wavs"
    lib_tr = ROOT / "eval" / "library_transcripts"
    trs = sorted(lib_tr.glob("*/score.abc"))

    cache_p = ROOT / "eval" / "melody_theory_cache.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}

    import time
    t0 = time.time()
    corpus = []
    for i, abc_path in enumerate(trs):
        stem = abc_path.parent.name
        if stem in cache:
            corpus.append(cache[stem])
            continue
        try:
            from melody_stats import analyze_song
            m = analyze_song(str(abc_path))
            meta, vocal, secs = parse_abc_data(str(abc_path))
            total_bars = max(1, int(m.get("total_bars", 50)))
            tonic_pc = get_tonic_pc(abc_text=None, tdir=abc_path.parent)

            r = dict(
                file=stem,
                density=m["density"], pitch_range=m["pitch_range"],
                motif=m["motif_repeat"], leap=m["leap_ratio"],
                pc_entropy=pitch_class_entropy(vocal),
                int_entropy=interval_entropy(vocal),
                dir_changes=direction_changes(vocal, total_bars),
                chrom_frac=chromatic_fraction(vocal, tonic_pc, minor=True),
                tonic_res=tonic_resolution(vocal, tonic_pc),
                arch=phrase_arch(vocal),
                chord_rate=chord_change_rate(abc_path.parent, total_bars),
            )
            cache[stem] = r
            corpus.append(r)
        except Exception as e:
            pass
        if (i + 1) % 50 == 0:
            cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"[{i+1}/{len(trs)}] el={time.time()-t0:.0f}s", flush=True)
    cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    # 生成曲
    from novel_songs_manifest import SONGS
    from hit_score import TRANSCRIPTS
    novel = []
    for s in SONGS:
        abc_p = Path(str(TRANSCRIPTS[s["no"]])) / "score.abc"
        if not abc_p.exists():
            continue
        try:
            m = analyze_song(str(abc_p))
            meta, vocal, secs = parse_abc_data(str(abc_p))
            tonic_pc = tonic_midi_near_key(s["key"].split()[0], m.get("tessitura", [62])[0]) % 12
            minor = "minor" in s["mode"]
            r = dict(
                file=f"{s['no']}_{s['title']}",
                density=m["density"], pitch_range=m["pitch_range"],
                motif=m["motif_repeat"], leap=m["leap_ratio"],
                pc_entropy=pitch_class_entropy(vocal),
                int_entropy=interval_entropy(vocal),
                dir_changes=direction_changes(vocal, max(1, int(m["total_bars"]))),
                chrom_frac=chromatic_fraction(vocal, tonic_pc, minor),
                tonic_res=tonic_resolution(vocal, tonic_pc),
                arch=phrase_arch(vocal),
            )
            novel.append(r)
        except Exception:
            pass

    # 热度相关性
    manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
    feats = json.loads((ROOT / "corpus" / "features.json").read_text(encoding="utf-8"))
    tier_map = {}
    for name in feats:
        tier_map[Path(name).stem[:20]] = tier_of(name)
    for r in corpus:
        r["tier"] = tier_map.get(r["file"][:20], 1)

    def cohen_d(a, b):
        a, b = np.array(a, float), np.array(b, float)
        s = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
        return float((a.mean() - b.mean()) / (s + 1e-9))

    new_keys = [
        ("pc_entropy", "音高分布熵", "低=聚焦"),
        ("int_entropy", "音程分布熵", "低=重复"),
        ("dir_changes", "方向变化/小节", "高=张力"),
        ("chrom_frac", "离调音占比", "高=色彩"),
        ("tonic_res", "主音解决率", "高=收束好"),
        ("arch", "乐句拱形度", "高=有弧线"),
        ("chord_rate", "和弦变化率", "适中=好"),
    ]

    lines = ["# 旋律学/和声新指标 × 热度相关性", "",
             f"真曲 n={len(corpus)} 生成 n={len(novel)}", "",
             "| 指标 | 真曲中位 | 生成中位 | d(生成vs真曲) | 热度ρ | 级别 |", "|---|---|---|---|---|---|"]

    for key, cname, meaning in new_keys:
        cv = [r[key] for r in corpus if r.get(key) is not None]
        nv = [r[key] for r in novel if r.get(key) is not None]
        tv = [(tier_map.get(r["file"][:20], 1), r[key]) for r in corpus if r.get(key) is not None]
        if len(cv) < 30:
            lines.append(f"| {cname} | - | - | - | 样本不足 |")
            continue
        cm = np.median(cv)
        nm = np.median(nv) if nv else 0
        d = cohen_d(nv, cv)
        # 热度ρ（仅真曲）
        tv_pairs = [(t, v) for t, v in tv]
        if len(set(tv_pairs[0] for tv_pairs in tv_pairs)) >= 2:
            rho = round(float(spearmanr([t for t, _ in tv_pairs], [v for _, v in tv_pairs]).statistic), 3)
        else:
            rho = 0
        g = "A" if abs(d) >= 0.8 else ("B" if abs(d) >= 0.5 else "C")
        lines.append(f"| {cname} | {np.median(cv):.3f} | {np.median(nv):.3f} | {d:+.2f} | ρ={rho:+.3f} | {g} |")

    (ROOT / "eval" / "melody_theory_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-20:]))


def parse_abc_data(abc_path):
    from ai_segment_forensic import parse_abc
    return parse_abc(abc_path)


def get_tonic_pc(tdir):
    from ai_segment_forensic import keysig_alterations
    kl = Path(tdir) / "key.lab"
    if kl.exists():
        first = kl.read_text(encoding="utf-8").splitlines()[0].split("\t")
        if len(first) >= 3:
            label = first[2].strip()  # e.g. "E:min"
            root_s = label.split(":")[0]
            from ai_segment_forensic import LETTER
            # parse root
            letter = root_s[0].upper()
            acc = root_s[1:] if len(root_s) > 1 else ""
            base = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}[letter]
            pc = (base + (1 if acc=="#" else -1 if acc=="b" else 0)) % 12
            return pc
    return 0


if __name__ == "__main__":
    main()
