# -*- coding: utf-8 -*-
"""极简旋律学指标 × 热度分析（无外部依赖）"""
import json, math, re, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ===== ABC 解析（内联，无外部依赖） =====
STEPS_MAJ = [2,2,1,2,2,1,2]
STEPS_MIN = [2,1,2,2,1,2,2]
LETTER_PC = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
NOTE_RE = re.compile(r"[_^=]?([A-Ga-gzZ])([,']*)(\d+)?(/\d+)?(-)?")
TOKEN_RE = re.compile(r"[_^=]?[A-Ga-gzZ][,']*\d*(?:/\d+)?-?")
SECTION_RE = re.compile(r"^\s*%\s*(\S+)\s*$")


def keysig_alterations(kline):
    m = re.match(r"([A-G])(#|b)?", kline)
    if not m: return {}
    tonic, acc = m.group(1), m.group(2) or ""
    SHARP_ORDER = ["F","C","G","D","A","E","B"]
    FLAT_ORDER = ["B","E","A","D","G","C","F"]
    SIG_SHARPS = {"G":1,"D":2,"A":3,"E":4,"B":5}
    SIG_FLATS = {"F":1,"B":2,"E":3,"A":4,"D":5,"G":6}
    if acc == "#":
        n = SIG_SHARPS.get(tonic+"#", SIG_SHARPS.get(tonic, 0))
        return {L: 1 for L in SHARP_ORDER[:n]}
    if acc == "b":
        n = SIG_FLATS.get(tonic+"b", SIG_FLATS.get(tonic, 0))
        return {L: -1 for L in FLAT_ORDER[:n]}
    return {}


def parse_chorus_notes(path):
    """解析副歌 Vocal 音符 -> (pitch_class_entropy, interval_entropy, direction_changes, chrom_frac, tonic_pc, arch_r2)"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    keysig, q, bar_u, voice = {}, 120.0, 16.0, "Vocal"
    pos, notes, sections = 0.0, [], []
    TOKEN = re.compile(r"([_^=]?)([A-Ga-gzZ])([,']*)(\d+)?(/\d+)?(-)?")
    NOTE_TOKEN = re.compile(r"[_^=]?[A-Ga-gzZ][,']*\d*(?:/\d+)?-?")
    SECTION = re.compile(r"^\s*%\s*(\S+)\s*$")

    for raw in lines:
        s = raw.strip()
        if s.startswith("%%"): continue
        if s.startswith("%"):
            m = SECTION.match(s)
            if m: sections.append((m.group(1).lower(), pos))
            continue
        if s.startswith("Q:"):
            qm = re.search(r"=([\d.]+)", s)
            if qm: q = float(qm.group(1))
            continue
        if s.startswith("L:"):
            lm = re.match(r"L:1/(\d+)", s)
            if lm: unit_q = 4.0/int(lm.group(1))
            continue
        if s.startswith("K:"):
            km = re.match(r"K:\s*(.+)", s)
            if km: keysig = keysig_alterations(km.group(1).strip())
            continue
        if s.startswith("M:"):
            if "2/4" in s: bar_u = 8.0
            elif "6/8" in s: bar_u = 12.0
            else: bar_u = 16.0
            continue
        if s.startswith("V:"):
            voice = "Ins" if "Ins" in s else ("Vocal" if "Vocal" in s else voice)
            continue
        if s.startswith(("X:","T:","C:")): continue
        if not s or s.startswith(("X:","T:","C:")): continue

        # 去和弦
        s_music = re.sub(r'"[^"]*"', "", s)
        if voice != "Vocal": continue

        for m in NOTE_TOKEN.finditer(s_music):
            tok = m.group(0)
            t = TOKEN.fullmatch(tok)
            if not t: continue
            let = t.group(2)
            if let in ("z","Z"):
                n = int(t.group(4)) if t.group(4) else 1
                pos += n * bar_u if let == "Z" else n * (bar_u/16)
                continue
            acc_s, octs = t.group(1), t.group(3)
            oct_d = octs.count("'") - octs.count(",")
            LETTER = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
            midi = 60 + LETTER[let.upper()] + 12*oct_d
            if let.islower(): midi += 12
            if acc_s == "^": midi += 1
            elif acc_s == "_": midi -= 1
            elif acc_s == "": midi += keysig.get(let.upper(), 0)
            n_dur = int(t.group(4)) if t.group(4) else 1
            frac = 1/int(t.group(5)[1:]) if t.group(5) else 1
            dur_u = n_dur * frac
            notes.append((pos, midi, dur_u))
            pos += dur_u

    sec_per_unit = (15.0 / q) if bar_u == 16 else (7.5 / q)
    vocal = [(p * sec_per_unit, mi, d * sec_per_unit) for p, mi, d in notes]
    return meta_sec_per_unit(q, bar_u), vocal, [(n, p * sec_per_unit) for n, p in sections]


def meta_sec_per_unit(q, bar_u):
    return (bar_u / 4) * 60.0 / q


def meta_sec_per_unit(q, bar_units):
    sec_per_bar = bar_units / 4 * 60 / q
    return sec_per_bar / bar_units  # per unit


def meta_sec_per_unit(q, bar_units):
    return (60.0 / q) / 4  # per 16th unit


def pc_entropy(vocal):
    from collections import Counter
    pcs = [int(m) % 12 for _, m, _ in vocal]
    if not pcs: return None
    c = np.bincount(pcs, minlength=12).astype(float)
    p = c / c.sum(); p = p[p > 0]
    h = -sum(x * math.log2(x) for x in p)
    return round(h / math.log2(12), 3)


def int_entropy(vocal):
    from collections import Counter
    notes = sorted(vocal)
    if len(notes) < 2: return None
    ivs = [abs(int(b[1] - a[1])) % 12 for a, b in zip(notes, notes[1:])]
    if not ivs: return None
    c = np.bincount(ivs, minlength=13).astype(float)
    p = c / c.sum(); p = p[p > 0]
    h = -sum(x * math.log2(x) for x in p)
    return round(h / math.log2(12), 3)


def dir_changes(vocal, total_bars):
    pitches = sorted([(o, m) for o, m, d in vocal])
    pitches = [m for _, m in pitches]
    if len(pitches) < 3: return None
    dirs = np.sign(np.diff(pitches))
    dirs = dirs[dirs != 0]
    if len(dirs) < 2: return None
    return round(np.sum(np.diff(dirs) != 0) / max(total_bars, 1), 2)


def chrom_frac(vocal, tonic_pc):
    if minor_scale(tonic_pc):
        dia = {0, 2, 3, 5, 7, 8, 10}
    else:
        dia = {0, 2, 4, 5, 7, 9, 11}
    total = len(vocal)
    if total == 0: return None
    chrom = sum(1 for _, m, _ in vocal if (int(m) - tonic_pc) % 12 not in dia)
    return round(chrom / total, 3)


def minor_scale(tonic_pc):
    return None  # placeholder


def tonic_resolution(vocal, tonic_pc):
    phrases, cur = [], []
    for onset, midi, dur in sorted(vocal):
        if dur > 0.5 * (60/120):  # 长音=乐句结束
            if cur: cur.append((onset, midi, dur)); phrases.append(cur); cur = []
            continue
        cur.append((onset, midi, dur))
    if cur: phrases.append(cur)
    if len(phrases) < 3: return None
    ends = [p[-1][1] for p in phrases if p]
    on_t = sum(1 for m in ends if (int(m) - tonic_pc) % 12 == 0)
    return round(on_t / len(ends), 3)


def phrase_arch(vocal):
    pitches = sorted([(o, m) for o, m, d in vocal])
    p = [m for _, m in pitches]
    if len(p) < 5: return None
    x = np.arange(len(p), dtype=float)
    try:
        c = np.polyfit(x, p, 2)
        pred = np.polyval(c, x)
        ss_r = np.sum((np.array(p) - pred)**2)
        ss_t = np.sum((np.array(p) - np.mean(p))**2)
        return round(1 - ss_r/(ss_t+1e-9), 3)
    except: return None


def main():
    trs = sorted(Path(ROOT / "eval" / "library_transcripts").glob("*/score.abc"))
    manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
    feats = json.loads((ROOT / "corpus" / "features.json").read_text(encoding="utf-8"))
    lib_wav = ROOT / "eval" / "library_wavs"

    # 热度
    MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
              "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
    KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团"}
    MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
              "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见",
              "天黑黑","绿光","月亮代表我的心","甜蜜蜜","后来","为爱痴狂","朋友",
              "第一次","勇气","老鼠爱大米","两只蝴蝶","秋天不回来","求佛","香水有毒",
              "该死的温柔","专属天使","北极星的眼泪","我难过","倒带","说爱你","那些花儿",
              "生如夏花","平凡之路","董小姐","成都","南山南","小苹果","最炫民族风"}

    def tier(stem):
        for a in MEGA_A:
            if a in stem: return 3
        for h in MEGA_S:
            if h in stem: return 3
        for a in KNOWN_A:
            if a in stem: return 2
        return 1

    rows = []
    for i, abc_p in enumerate(trs):
        stem = abc_p.parent.name
        wav = lib_wav / (stem + ".wav")
        if not wav.exists(): continue
        try:
            # 简化解析（直接用文件级函数）
            # 只需要 notes 和 tonic
            text = abc_p.read_text(encoding="utf-8")
            tonic_pc = 0
            km = re.search(r"K:\s*(\S)", text)
            if km:
                LETTER = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
                tonic_pc = LETTER.get(km.group(1).upper(), 0)

            # 手动解析
            all_notes = []
            voice = "Vocal"
            pos = 0.0
            q = 120.0
            bar_u = 16.0
            for raw in text.splitlines():
                s = raw.strip()
                if s.startswith("Q:"):
                    qm = re.search(r"=([\d.]+)", s)
                    if qm: q = float(qm.group(1))
                elif s.startswith("V:") and "Ins" in s:
                    voice = "Ins"
                elif s.startswith("V:") and "Vocal" in s:
                    voice = "Vocal"
                elif not s or s[0] in "|": continue
                elif s[0] in "XTCKMLV%": continue
                if voice != "Vocal": continue
                sec_per_unit = (60.0 / q) / 4
                for tok_m in re.finditer(r"[_^=]?[A-Ga-gzZ][,']*\d*(?:/\d+)?-?", s):
                    tok = tok_m.group(0)
                    tm = re.match(r"([_^=]?)([A-Ga-gzZ])([,']*)(\d+)?(/\d+)?(-)?", tok)
                    if not tm: continue
                    let = tm.group(2)
                    n = int(tm.group(4)) if tm.group(4) else 1
                    frac = 1/int(tm.group(5)[1:]) if tm.group(5) else 1.0
                    if let in ("z","Z"):
                        u = n * bar_u if let == "Z" else n * frac
                        pos += u; continue
                    octs = tm.group(3).count("'") - tm.group(3).count(",")
                    midi = 60 + {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}[let.upper()] + 12*octs
                    if let.islower(): midi += 12
                    if tm.group(1) == "^": midi += 1
                    elif tm.group(1) == "_": midi -= 1
                    notes_list = [(pos * sec_per_unit, midi, n * frac * sec_per_unit)]
                    all_notes.extend(notes_list)
                    pos += n * frac

            if len(all_notes) < 10: continue

            t = tier(stem.replace("lib_","",1))
            mnotes = [(o, int(mi), d) for o, mi, d in all_notes]
            total_bars = max(1, int(pos / bar_u))

            # 新指标
            from collections import Counter
            pcs = [int(m) % 12 for _, m, _ in mnotes]
            p = np.bincount(pcs, minlength=12).astype(float); p /= p.sum(); p = p[p>0]
            pc_h = round(float(-sum(x*math.log2(x) for x in p)/math.log2(12)), 3)

            sorted_n = sorted(mnotes)
            ivs = [abs(b[1]-a[1]) for a, b in zip(sorted_n, sorted_n[1:])]
            ivs = [x % 12 for x in ivs]
            c = np.bincount(ivs, minlength=12).astype(float); c /= c.sum(); c = c[c>0]
            int_h = round(float(-sum(x*math.log2(x) for x in c)/math.log2(12)), 3)

            rows.append(dict(file=stem, tier=t, density=len(mnotes)/total_bars,
                           pitch_range=max(x[1] for x in mnotes)-min(x[1] for x in mnotes),
                           pc_entropy=pc_h, int_entropy=int_h))
        except Exception:
            pass
        if (i+1) % 100 == 0:
            print(f"[{i+1}/{len(trs)}]", flush=True)

    # ===== 热度相关 =====
    from scipy.stats import spearmanr
    new_keys = ["pc_entropy", "int_entropy", "pitch_range"]
    names = ["音高分布熵", "音程分布熵", "音域跨度"]
    for key, cname in zip(new_keys, names):
        xs = [r["tier"] for r in rows]
        ys = [r[key] for r in rows if r.get(key) is not None]
        if len(ys) < 50: continue
        rho, _ = spearmanr(xs[:len(ys)], ys)
        s_med = np.median([r[key] for r in rows if r["tier"] == 3])
        b_med = np.median([r[key] for r in rows if r["tier"] == 1])
        mark = "✅" if abs(rho) >= 0.12 else "❌"
        print(f"{cname}: ρ={rho:+.3f} {mark} | S层={s_med:.2f} B层={b_med:.2f}")

    print(f"\nDone: {len(rows)} songs")


if __name__ == "__main__":
    main()
