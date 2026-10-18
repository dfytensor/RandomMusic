# -*- coding: utf-8 -*-
"""歌词质量检查器（生成前的词面保障，map §5.5）：
1) 段落标签合法且与模板匹配  2) 行数 vs 模板  3) 行长分布  4) 押韵覆盖(中华新韵十三辙)
5) 意外重复短语  6) hook 规则(副歌首行=歌名, 可选传入)
用法: python lyrics_check.py <lyrics.txt> [T1|T2|T3|T4] [--title 歌名]
"""
import re
import sys
from collections import Counter
from pathlib import Path

from pypinyin import pinyin, Style

ROOT = Path(__file__).parent

VALID_TAGS = {"Intro", "Verse", "Pre-Chorus", "Chorus", "Interlude", "Bridge", "Outro"}
TEMPLATE_LINES = {
    "T1": {"Verse": 8, "Pre-Chorus": 4, "Chorus": 6, "Bridge": 2},
    "T2": {"Verse": 8, "Pre-Chorus": 4, "Chorus": 6, "Bridge": 2},
    "T3": {"Verse": 8, "Pre-Chorus": 4, "Chorus": 6, "Bridge": 2},
    "T4": {"Verse": 8, "Pre-Chorus": 4, "Chorus": 6, "Bridge": 2},
}
# 中华新韵·十三辙 韵母分组
RHYME_GROUPS = {
    "发花": {"a", "ia", "ua"}, "梭波": {"o", "e", "uo"}, "乜斜": {"ie", "üe"},
    "怀来": {"ai", "uai"}, "灰堆": {"ei", "ui"}, "遥条": {"ao", "iao"},
    "由求": {"ou", "iu"}, "言前": {"an", "ian", "uan", "üan"},
    "人辰": {"en", "in", "un", "ün"}, "江阳": {"ang", "iang", "uang"},
    "中东": {"eng", "ing", "ong", "iong"}, "衣期": {"i", "ü", "er"},
}
FINAL2GROUP = {}
for g, finals in RHYME_GROUPS.items():
    for f in finals:
        FINAL2GROUP[f] = g


def line_final(text):
    """取一行最后一个有效字的韵母 -> 韵辙。"""
    han = re.findall(r"[\u4e00-\u9fff]", text)
    if not han:
        return None
    py = pinyin(han[-1], style=Style.FINALS, strict=False)
    f = py[0][0] if py and py[0] else ""
    return FINAL2GROUP.get(f)


def parse_lyrics(text):
    sections, cur = [], None
    for raw in text.splitlines():
        s = raw.strip()
        m = re.match(r"\[([^\]]+)\]", s)
        if m:
            cur = m.group(1)
            sections.append({"tag": cur, "lines": []})
            continue
        if s and cur:
            sections[-1]["lines"].append(s)
    return sections


def check(text, template=None, title=None):
    checks, warns = [], []

    def add(name, ok, detail=""):
        checks.append((name, ok, detail))

    sections = parse_lyrics(text)
    tags = [s["tag"] for s in sections]
    bad_tags = [t for t in tags if t not in VALID_TAGS]
    add("段落标签合法", not bad_tags, f"非法: {bad_tags}" if bad_tags else f"{len(tags)} 段")

    if template:
        need = TEMPLATE_LINES[template]
        line_ok, detail = True, []
        for sec in sections:
            if sec["tag"] in need:
                lo, hi = need[sec["tag"]] - 2, need[sec["tag"]] + 2
                if not (lo <= len(sec["lines"]) <= hi):
                    line_ok = False
                    want = need[sec["tag"]]
                    detail.append(f"{sec['tag']}:{len(sec['lines'])}行(期望{want}±2)")
        add("行数 vs 模板", line_ok, "; ".join(detail) or "全部符合")

    # 行长
    lens = []
    long_lines = []
    for s in sections:
        if s["tag"] in ("Intro", "Interlude", "Outro"):
            continue
        for l in s["lines"]:
            n = len(re.findall(r"[\u4e00-\u9fff]", l))
            if n:
                lens.append(n)
                if n > 14:
                    long_lines.append(l)
    med = sorted(lens)[len(lens) // 2] if lens else 0
    add("行长中位 7-12 字", 7 <= med <= 12, f"中位 {med}")
    add("无超长行(>14字)", not long_lines, f"{long_lines[:2]}" if long_lines else "")

    # 押韵覆盖（有 ≥4 行的歌词段）
    covs = []
    for s in sections:
        if s["tag"] in ("Intro", "Interlude", "Outro") or len(s["lines"]) < 4:
            continue
        groups = [line_final(l) for l in s["lines"]]
        groups = [g for g in groups if g]
        if not groups:
            continue
        top, cnt = Counter(groups).most_common(1)[0]
        covs.append(cnt / len(groups))
    rhyme = sum(covs) / len(covs) if covs else 0
    add("押韵覆盖 ≥50%", rhyme >= 0.5, f"{rhyme*100:.0f}% ({len(covs)} 段)")

    # 意外重复短语（≥4字词块在非副歌段重复≥3次）
    body = " ".join(l for s in sections if s["tag"] not in ("Chorus",) for l in s["lines"])
    grams = Counter(body[i:i + 4] for i in range(len(body) - 3))
    dup = [g for g, c in grams.items() if c >= 3 and not re.match(r"^[\u4e00-\u9fff]{0,2}$", g)]
    add("无意外重复词块(≥4字×3次)", not dup, f"{dup[:3]}" if dup else "")

    # hook
    if title:
        chorus_secs = [s for s in sections if s["tag"] == "Chorus"]
        ok = any(title in l for s in chorus_secs for l in s["lines"][:2]) if chorus_secs else False
        add("副歌前两行含歌名(hook)", ok, f"歌名={title}")

    return checks, dict(rhyme_coverage=round(rhyme, 2), line_len_median=med, n_sections=len(tags))


def main():
    path = Path(sys.argv[1])
    template = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    title = None
    if "--title" in sys.argv:
        title = sys.argv[sys.argv.index("--title") + 1]
    text = path.read_text(encoding="utf-8")
    checks, summary = check(text, template, title)
    print(f"# 歌词质量检查: {path.name}")
    print(f"模板: {template or '未指定'} | 摘要: {summary}")
    print()
    print("| 检查项 | 判定 | 详情 |")
    print("|---|---|---|")
    for name, ok, detail in checks:
        print(f"| {name} | {'✅' if ok else '❌'} | {detail} |")
    npass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n通过 {npass}/{len(checks)}")
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
