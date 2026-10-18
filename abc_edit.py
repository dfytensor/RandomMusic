# -*- coding: utf-8 -*-
"""ABC 定向手术工具：移调/折返音域/减少跳进/合并稀疏化/拆分加密。
和弦 "..." 引号内容不动；只处理 Vocal 声部。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from abc_compare import TOKEN, NOTE_TOKEN, keysig_alterations

CHORD = re.compile(r'("[^"]*")')
SECTION = re.compile(r"^\s*%\s*(\S+)\s*$")
LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PC2LETTER = {0: "C", 1: "C", 2: "D", 3: "D", 4: "E", 5: "F", 6: "F",
             7: "G", 8: "G", 9: "A", 10: "A", 11: "B"}


def midi_to_token(midi, dur_str="", tie=""):
    octv = (midi - 60) // 12
    pc = midi - 60 - 12 * octv
    letter = PC2LETTER[pc]
    diff = pc - LETTERS[letter]
    acc = "^" * diff if diff > 0 else "_" * (-diff)
    if octv >= 0:
        octs = "'" * octv
    else:
        octs = "," * (-octv)
    return f"{acc}{letter}{octs}{dur_str}{tie}"


def _token_midi(tok, keysig):
    m = TOKEN.fullmatch(tok)
    if not m or m.group("let") in ("z", "Z"):
        return None
    let = m.group("let")
    octv = m.group("oct").count("'") - m.group("oct").count(",")
    midi = 60 + LETTERS[let.upper()] + 12 * octv
    if let.islower():
        midi += 12
    acc = m.group("acc")
    if acc == "^":
        midi += 1
    elif acc == "_":
        midi -= 1
    elif acc == "":
        midi += keysig.get(let.upper(), 0)
    return midi


def _map_notes(text, fn, only_sections=None, keysig=None):
    """fn(midi, tok)->new midi or None(keep). 只动 Vocal 的非和弦部分。"""
    if keysig is None:
        keysig = _read_keysig(text)
    cur_sec, voice = "", "Vocal"
    out_lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("%"):
            m = SECTION.match(s)
            if m:
                cur_sec = m.group(1).lower()
            out_lines.append(line)
            continue
        if s.startswith("V:"):
            voice = "Ins" if "Ins" in s else ("Vocal" if "Vocal" in s else voice)
            out_lines.append(line)
            continue
        if s.startswith(("X:", "T:", "C:", "L:", "M:", "K:", "Q:")) or not s:
            out_lines.append(line)
            continue
        if voice != "Vocal" or (only_sections is not None and cur_sec not in only_sections):
            out_lines.append(line)
            continue
        parts = re.split(r'("[^"]*")', line)
        new_parts = []
        for p in parts:
            if p.startswith('"') or not p:
                new_parts.append(p)
                continue

            def repl(m):
                tok = m.group(0)
                midi = _token_midi(tok, keysig)
                if midi is None:
                    return tok
                new = fn(midi, tok)
                if new is None or new == midi:
                    return tok
                t = TOKEN.fullmatch(tok)
                dur = (t.group("len") or "") + (t.group("frac") or "")
                return midi_to_token(new, dur, t.group("tie") or "")

            new_parts.append(NOTE_TOKEN.sub(repl, p))
        out_lines.append("".join(new_parts))
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def _read_keysig(text):
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("K:"):
            return keysig_alterations(s[2:].strip())
    return {}


def op_transpose(text, sections, delta):
    return _map_notes(text, lambda midi, tok: midi + delta, only_sections=set(sections))


def op_fold_range(text, lo, hi):
    def fn(midi, tok):
        m = midi
        while m > hi:
            m -= 12
        while m < lo:
            m += 12
        if lo <= m <= hi:
            return m
        return min(max(m, lo), hi)
    return _map_notes(text, fn)


def op_reduce_leaps(text, lo, hi, max_interval=7):
    keysig = _read_keysig(text)
    # 收集 Vocal 音符流（整行坐标，跳过和弦引号区）
    spans = []
    cur_sec, voice = "", "Vocal"
    for li, line in enumerate(text.splitlines()):
        s = line.strip()
        if s.startswith("%"):
            m = SECTION.match(s)
            if m:
                cur_sec = m.group(1).lower()
            continue
        if s.startswith("V:"):
            voice = "Ins" if "Ins" in s else ("Vocal" if "Vocal" in s else voice)
            continue
        if voice != "Vocal" or s.startswith(("X:", "T:", "C:", "L:", "M:", "K:", "Q:")) or not s:
            continue
        chord_spans = [(m.start(), m.end()) for m in re.finditer(r'"[^"]*"', line)]
        for m in NOTE_TOKEN.finditer(line):
            if any(a <= m.start() < b for a, b in chord_spans):
                continue
            midi = _token_midi(m.group(0), keysig)
            if midi is not None:
                spans.append([li, m.start(), m.end(), m.group(0), midi])
    prev = None
    for sp in spans:
        midi = sp[4]
        if prev is not None and abs(midi - prev) >= max_interval + 1:
            for cand in (midi - 12, midi + 12):
                if lo <= cand <= hi and abs(cand - prev) < abs(midi - prev):
                    sp[4] = cand
                    break
        prev = sp[4]
    # 重建（按行、从右往左替换，位置为整行坐标）
    lines = text.splitlines()
    by_line = {}
    for li, a, b, tok, newmidi in spans:
        by_line.setdefault(li, []).append((a, b, tok, newmidi))
    for li, items in by_line.items():
        for a, b, tok, newmidi in sorted(items, key=lambda x: -x[1]):
            t = TOKEN.fullmatch(tok)
            dur = (t.group("len") or "") + (t.group("frac") or "")
            lines[li] = lines[li][:a] + midi_to_token(newmidi, dur, t.group("tie") or "") + lines[li][b:]
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def op_tie_merge(text, sections, max_merges=40):
    """同小节内相邻同音合并（长度相加），降低密度。"""
    merges = {"n": 0}

    def line_fn(line):
        keysig = _read_keysig(text)
        parts = re.split(r'("[^"]*")', line)
        out = []
        for p in parts:
            if p.startswith('"') or not p:
                out.append(p)
                continue
            toks = []
            pos = 0
            for m in NOTE_TOKEN.finditer(p):
                if m.start() > pos:
                    out.append(p[pos:m.start()])
                toks.append(m.group(0))
                pos = m.end()
            if pos < len(p):
                out.append(p[pos:])
            res = []
            i = 0
            while i < len(toks):
                if i + 1 < len(toks):
                    t1, t2 = toks[i], toks[i + 1]
                    m1, m2 = TOKEN.fullmatch(t1), TOKEN.fullmatch(t2)
                    k1 = _token_midi(t1, keysig)
                    k2 = _token_midi(t2, keysig)
                    if (k1 is not None and k2 is not None and k1 == k2
                            and not m1.group("tie") and not m2.group("acc")
                            and (m1.group("len") or "1").isdigit() and (m2.group("len") or "1").isdigit()):
                        n1 = int(m1.group("len") or 1) + int(m2.group("len") or 1)
                        dur = m1.group("frac") or ""
                        res.append(midi_to_token(k1, f"{n1}{dur}"))
                        merges["n"] += 1
                        i += 2
                        continue
                res.append(toks[i])
                i += 1
            out.extend(res)
        return "".join(out)

    prev = None
    cur = text
    for _ in range(4):
        if merges["n"] >= max_merges:
            break
        prev = cur
        cur = _rewrite_music(cur, line_fn, only_sections=set(sections))
        if cur == prev:
            break
    return cur


def _rewrite_music(text, line_fn, only_sections=None):
    cur_sec, voice = "", "Vocal"
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("%"):
            m = SECTION.match(s)
            if m:
                cur_sec = m.group(1).lower()
            out.append(line)
            continue
        if s.startswith("V:"):
            voice = "Ins" if "Ins" in s else ("Vocal" if "Vocal" in s else voice)
            out.append(line)
            continue
        if voice != "Vocal" or (only_sections is not None and cur_sec not in only_sections) \
                or s.startswith(("X:", "T:", "C:", "L:", "M:", "K:", "Q:")) or not s:
            out.append(line)
            continue
        out.append(line_fn(line))
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def op_split_long(text, sections, min_units=4, max_splits=40):
    """长音拆成同音两次发音，提高密度。"""
    splits = {"n": 0}

    def line_fn(line):
        parts = re.split(r'("[^"]*")', line)
        out = []
        for p in parts:
            if p.startswith('"') or not p:
                out.append(p)
                continue

            def repl(m):
                if splits["n"] >= max_splits:
                    return m.group(0)
                tok = m.group(0)
                t = TOKEN.fullmatch(tok)
                keysig = _read_keysig(text)
                midi = _token_midi(tok, keysig)
                if midi is None:
                    return tok
                n = t.group("len")
                if not n or not n.isdigit() or int(n) < min_units or t.group("tie"):
                    return tok
                half = int(n) // 2
                if half < 1:
                    return tok
                splits["n"] += 1
                return midi_to_token(midi, f"{half}") + midi_to_token(midi, f"{int(n)-half}")

            out.append(NOTE_TOKEN.sub(repl, p))
        return "".join(out)

    return _rewrite_music(text, line_fn, only_sections=set(sections))
