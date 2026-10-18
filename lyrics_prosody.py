# -*- coding: utf-8 -*-
"""歌词韵律量化评价 (古诗级指标): 押韵率/字数整齐/平仄收尾/对句字数
用法: python lyrics_prosody.py [lyrics.json]  默认 eval/lyrics_current.json
"""
import json, re, sys
from pathlib import Path
from collections import Counter
from pypinyin import pinyin, Style

ROOT = Path(__file__).parent

def syllables(line: str) -> str:
    """去表演标记/标点, 只留汉字"""
    line = re.sub(r"\(.*?\)", "", line)
    return re.sub(r"[^\u4e00-\u9fff]", "", line)

def tone_cls(ch: str) -> str:
    """平/仄 (1,2声=平, 3,4声=仄; 轻声归仄)"""
    py = pinyin(ch, style=Style.TONE3, errors=lambda x: [[x]])[0][0]
    m = re.search(r"(\d)$", py)
    if not m:
        return "z"  # 非汉字/轻声
    return "p" if int(m.group(1)) in (1, 2) else "z"

def rhyme_key(ch: str) -> str:
    """末字十三辙 (歌曲押韵标准: 发花/江阳/言前...)"""
    py = pinyin(ch, style=Style.FINALS_TONE3, errors=lambda x: [[x]])[0][0]
    py = re.sub(r"\d$", "", py).replace("v", "ü")
    # 十三辙归并
    table = [
        (("iang", "uang", "ang"), "江阳"),
        (("ian", "uan", "van", "an"), "言前"),
        (("ing", "eng", "ong", "iong", "ueng"), "中东"),
        (("en", "in", "un", "ün"), "人辰"),
        (("ai", "uai"), "怀来"),
        (("ei", "ui", "uei"), "灰堆"),
        (("ao", "iao"), "遥条"),
        (("ou", "iu", "iou"), "由求"),
        (("a", "ia", "ua"), "发花"),
        (("o", "uo", "e", "ê"), "梭波"),
        (("u",), "姑苏"),
        (("i", "ü", "er"), "衣期"),
    ]
    for finals, name in table:
        if py in finals:
            return name
    # 独立元音/特殊: zhi chi shi ri zi ci si → 衣期(支思); 以主要元音兜底
    if py and py[0] in "zcs" and len(py) <= 3:
        return "衣期"
    if py and py[0] == "r" and py[1:2] in ("i", ""):
        return "衣期"
    return py or "?"

SECTIONS = ("Verse", "Pre-Chorus", "Chorus", "Bridge", "Intro", "Outro")

def parse(text: str):
    sections, cur = {}, None
    for ln in text.splitlines():
        m = re.match(r"\[(.+?)\]", ln.strip())
        if m:
            name = m.group(1)
            cur = name if name in SECTIONS else None
            if cur:
                sections.setdefault(cur, [])
            continue
        syl = syllables(ln)
        if cur and syl:
            sections[cur].append(syl)
    return sections

def score_lyrics(text: str) -> dict:
    secs = parse(text)
    per = {}
    # 押韵+平仄: 全部段落; 对句/整齐: 仅 Verse+Pre-Chorus (副歌 hook 短行是歌曲结构, 不罚)
    all_rhyme, all_tonepair, all_uniform, all_couplet = [], [], [], []
    for name, lines in secs.items():
        if name in ("Intro", "Outro") or len(lines) < 2:
            continue
        strict = name in ("Verse", "Pre-Chorus")
        if strict:
            lens = [len(l) for l in lines]
            mean = sum(lens) / len(lens)
            var = sum((x - mean) ** 2 for x in lens) / len(lens)
            cv = (var ** 0.5) / mean if mean else 1
            all_uniform.append(1 / (1 + cv))
            all_couplet.append(sum(1 for a, b in zip(lens, lens[1:]) if a == b) / (len(lens) - 1))
        # 押韵率: 众数占比 与 隔行(对句)押韵率 取大者 (歌曲以对句押韵为主)
        finals = [rhyme_key(l[-1]) for l in lines]
        cnt = Counter(finals)
        dom, dom_n = cnt.most_common(1)[0]
        rhyme_rate = dom_n / len(lines)
        pair_rh = (sum(1 for a, b in zip(finals[::2], finals[1::2]) if a == b)
                   / max(1, len(finals) // 2))
        rhyme_rate = max(rhyme_rate, pair_rh)
        tails = [tone_cls(l[-1]) for l in lines]
        pairs = list(zip(tails, tails[1:]))
        tp = sum(1 for a, b in pairs if a != b) / len(pairs) if pairs else 0
        per[name] = {"n": len(lines), "rhyme": round(rhyme_rate, 3), "dom_rhyme": dom,
                     "tone_pair": round(tp, 3), "strict": strict}
        all_rhyme.append(rhyme_rate); all_tonepair.append(tp)
    if not all_rhyme:
        return {"total": 0, "per": {}}
    u = sum(all_uniform)/len(all_uniform) if all_uniform else 0.5
    c = sum(all_couplet)/len(all_couplet) if all_couplet else 0.5
    total = (0.35 * sum(all_rhyme)/len(all_rhyme)
             + 0.25 * u
             + 0.20 * sum(all_tonepair)/len(all_tonepair)
             + 0.20 * c) * 100
    return {"total": round(total, 1),
            "rhyme": round(sum(all_rhyme)/len(all_rhyme), 3),
            "uniformity": round(u, 3),
            "tone_pair": round(sum(all_tonepair)/len(all_tonepair), 3),
            "couplet_len": round(c, 3),
            "per": per}

def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"eval"/"lyrics_current.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    print(f"{'no':4s} {'总分':>6s} {'押韵':>6s} {'整齐':>6s} {'平仄':>6s} {'对句':>6s}")
    report = {}
    for no in sorted(data):
        r = score_lyrics(data[no]["lyrics"] if isinstance(data[no], dict) else data[no])
        report[no] = r
        print(f"{no:4s} {r['total']:>6.1f} {r['rhyme']:>6.3f} {r['uniformity']:>6.3f} "
              f"{r['tone_pair']:>6.3f} {r['couplet_len']:>6.3f}")
    out = ROOT/"eval"/("prosody_" + src.stem + ".json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"SAVED {out}")

if __name__ == "__main__":
    main()
