# -*- coding: utf-8 -*-
"""歌词可唱性评价器 v4.2
实证链: v3(古诗韵律) vs 渲染分 ρ=0.077; 同seed A/B 显示 v3 词平均 -17.7 (10负1平1胜)
诊断: 全篇等长(cv↓)与单韵到底伤乐句呼吸; 格言化改写伤意象具体性
v4.2 修订:
  - 押韵率 = 对句押韵率 (偶句入韵), 单韵到底不再加分
  - 新增 行长变化度: cv∈[0.12,0.45] 满分, 奖励乐句呼吸
  - 新增 辙多样性: 全篇主辙数≥3 满分, 惩罚一韵到底
  - 保留: 平仄/对句等长/常用字/开口韵/具象密度
权重: 押韵0.15 平仄0.10 对句0.10 常用字0.10 开口韵0.15 具象0.15 行长变化0.15 辙多样0.10
校准目标: 旧词冠军词(渲染好)应高分, v3韵文(AB受损)应低分 — 诗(五绝)不再作为金标
"""
import re
import math
from collections import Counter
from pathlib import Path

try:
    from pypinyin import pinyin, Style
except ImportError:
    raise SystemExit("pip install pypinyin")

SECTION_RE = re.compile(r"^\s*\[([A-Za-z -]+)\]\s*$")
MARK_RE = re.compile(r"^\s*\([^)]*\)\s*")
PERF_MARK = re.compile(r"\((?:build|breathy|whisper|soft|belt|ad-?lib|light)[^)]*\)", re.I)

# 十三辙: finals(tone3 去调号后) → 辙
RHyme_GROUPS = {
    "发花": ("a", "ia", "ua"),
    "梭波": ("o", "e", "uo"),
    "乜斜": ("ie", "üe"),
    "怀来": ("ai", "uai"),
    "灰堆": ("ei", "ui"),
    "遥条": ("ao", "iao"),
    "由求": ("ou", "iu"),
    "言前": ("an", "ian", "uan", "üan"),
    "人辰": ("en", "in", "un", "ün"),
    "江阳": ("ang", "iang", "uang"),
    "中东": ("eng", "ing", "ong", "iong"),
    "衣期": ("i", "er", "ü"),
    "姑苏": ("u"),
}
FINAL2ZHE = {f: z for z, fs in RHyme_GROUPS.items() for f in fs}

# 辙的开口度 (演唱响亮度 0-1): 遥条含-a-最亮; 衣期/姑苏齐齿合口最闷
ZHE_OPEN = {
    "发花": 1.0, "江阳": 1.0, "遥条": 0.95, "言前": 0.85, "中东": 0.8,
    "怀来": 0.75, "由求": 0.6, "人辰": 0.55, "梭波": 0.5, "灰堆": 0.4,
    "乜斜": 0.35, "衣期": 0.25, "姑苏": 0.25,
}
STRICT_SECTIONS = {"verse", "pre-chorus", "prechorus"}
FREE_SECTIONS = {"chorus", "bridge", "intro", "outro", "interlude", "hook"}

# CJK 生僻判定: 基本区 4E00-9FFF 之外(扩展A/B/+)视为生僻 — 常用字全在基本区高频段
def _is_rare(ch):
    o = ord(ch)
    return not (0x4E00 <= o <= 0x9FFF)


def _parse(text):
    """→ [(section, [line, ...]), ...] 剥离表演标记与空行"""
    out, cur = [], None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = SECTION_RE.match(s)
        if m:
            cur = (m.group(1).lower(), [])
            out.append(cur)
            continue
        s = MARK_RE.sub("", s).strip()
        if not s:
            continue
        if cur is None:
            cur = ("verse", [])
            out.append(cur)
        cur[1].append(s)
    return out


def _finals(line):
    py = pinyin(line, style=Style.FINALS_TONE3, errors="ignore")
    return [x[0] for x in py if x and x[0]]


def _zhe(final):
    f = re.sub(r"[0-9]$", "", final)
    return FINAL2ZHE.get(f)


def _tone(final):
    m = re.search(r"([0-9])$", final)
    if not m:
        return None
    return "p" if m.group(1) in ("1", "2") else "z"


def score_lyrics(text, detail=False):
    sections = _parse(text)
    all_lines = [l for _, ls in sections for l in ls]

    # ---- 押韵率: 对句押韵率 (偶句入韵; 单韵到底不额外加分) & 开口韵 ----
    rh_num = rh_den = 0
    open_num = open_den = 0
    strict_lines = []
    dom_zhes = []
    for name, lines in sections:
        if not lines:
            continue
        fins = [_finals(l)[-1] if _finals(l) else "" for l in lines]
        zhes = [_zhe(f) for f in fins]
        if zhes:
            ev, od = zhes[0::2], zhes[1::2]
            ok = tot = 0
            for a, b in zip(ev, od):
                if a and b:
                    tot += 1
                    ok += 1 if a == b else 0
            if tot:
                rh_num += ok / tot * len(zhes)
                rh_den += len(zhes)
            known = [z for z in zhes if z]
            if known:
                open_num += ZHE_OPEN.get(Counter(known).most_common(1)[0][0], 0.5) * len(zhes)
                open_den += len(zhes)
        if name in STRICT_SECTIONS:
            strict_lines.extend(lines)
            known = [z for z in zhes if z]
            if known:
                dom_zhes.append(Counter(known).most_common(1)[0][0])

    rhyme = rh_num / rh_den if rh_den else 0.0
    openness = open_num / open_den if open_den else 0.0
    zhe_div = min(len(set(dom_zhes)) / 3, 1.0) if dom_zhes else 0.0  # strict段主辙多样性

    # ---- 平仄收尾相异率 (相邻行) ----
    tn_num = tn_den = 0
    for name, lines in sections:
        fins = [_finals(l)[-1] if _finals(l) else "" for l in lines]
        ts = [_tone(f) for f in fins]
        for a, b in zip(ts, ts[1:]):
            if a and b:
                tn_den += 1
                tn_num += 1 if a != b else 0
    tone_pair = tn_num / tn_den if tn_den else 0.0

    # ---- 对句等长率 (Verse+Pre-Chorus) ----
    cp_num = cp_den = 0
    for name, lines in sections:
        if name not in STRICT_SECTIONS:
            continue
        for a, b in zip(lines[0::2], lines[1::2]):
            cp_den += 1
            cp_num += 1 if len(a) == len(b) else 0
    couplet = cp_num / cp_den if cp_den else 0.0

    # ---- 常用字率 ----
    chars = [c for l in all_lines for c in l if "\u4e00" <= c <= "\u9fff"]
    common = sum(1 for c in chars if not _is_rare(c)) if chars else 0
    common_rate = common / len(chars) if chars else 0.0

    # ---- 具象密度 (名词+动词占比; jieba 词性) ----
    try:
        import jieba.posseg as pseg
        words = [w for l in all_lines for w, _f in pseg.cut(l)]
        nv = sum(1 for w, f in pseg.cut("".join(all_lines)) if f.startswith("n") or f.startswith("v"))
        concrete = nv / len(words) if words else 0.0
    except ImportError:
        concrete = 0.5

    # ---- 行长变化度: cv∈[0.12,0.45] 满分 (乐句呼吸), 两侧线性衰减 ----
    lens = [len(l) for l in all_lines]
    mean_len = sum(lens) / len(lens) if lens else 0
    cv = (sum((x - mean_len) ** 2 for x in lens) / len(lens)) ** 0.5 / mean_len if mean_len else 0.0
    if cv < 0.12:
        len_var = max(0.0, cv / 0.12)
    elif cv <= 0.45:
        len_var = 1.0
    else:
        len_var = max(0.0, 1.0 - (cv - 0.45) / 0.45)

    # ---- 表演标记保留 (软提示, 不计权重, 输出供人工检查) ----
    n_marks = len(PERF_MARK.findall(text))

    total = (0.15 * rhyme + 0.10 * tone_pair + 0.10 * couplet +
             0.10 * common_rate + 0.15 * openness +
             0.15 * min(concrete / 0.65, 1.0) + 0.15 * len_var + 0.10 * zhe_div) * 100

    r = {"total": round(total, 1), "rhyme": round(rhyme, 3), "tone_pair": round(tone_pair, 3),
         "couplet": round(couplet, 3), "common_rate": round(common_rate, 3),
         "openness": round(openness, 3), "concrete": round(concrete, 3),
         "len_cv": round(cv, 3), "len_var": round(len_var, 3),
         "zhe_div": round(zhe_div, 3), "n_marks": n_marks}
    return r


def score_file(path):
    return score_lyrics(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1:
        print(json.dumps(score_file(sys.argv[1]), ensure_ascii=False, indent=2))
