# -*- coding: utf-8 -*-
"""能量弧线 + 和弦撞音率：曲库基线 vs YuE2 生成曲 -> eval/arc_consonance.json"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from abc_compare import parse_abc

ROOT = Path(__file__).parent
SR = 22050

LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_chord_label(label):
    """Harte label -> root pc + chord tone offsets."""
    if label in ("N", "X", "*"):
        return None
    if ":" in label:
        root_s, qual = label.split(":", 1)
    else:
        root_s, qual = label, "maj"
    i = 1
    if len(root_s) > 1 and root_s[1] in "#b":
        i = 2
    root = LETTER_PC[root_s[0].upper()]
    if root_s[1:2] == "#":
        root += 1
    elif root_s[1:2] == "b":
        root -= 1
    q = qual.lower()
    if q.startswith("min") or q.startswith("m-"):
        tones = [0, 3, 7]
    elif q.startswith("dim"):
        tones = [0, 3, 6]
    elif q.startswith("aug"):
        tones = [0, 4, 8]
    elif q.startswith("sus4"):
        tones = [0, 5, 7]
    elif q.startswith("sus2"):
        tones = [0, 2, 7]
    else:
        tones = [0, 4, 7]
    if "maj7" in q:
        tones = tones + [11]
    elif "7" in q:
        tones = tones + [10]
    if "6" in q:
        tones = tones + [9]
    return root % 12, sorted(set(t % 12 for t in tones))


def load_chords(lab_path):
    chords = []
    for line in Path(lab_path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        t1, t2, label = float(parts[0]), float(parts[1]), parts[2].strip()
        pc = parse_chord_label(label)
        chords.append((t1, t2, pc))
    return chords


def chord_at(chords, t):
    for t1, t2, pc in chords:
        if t1 <= t < t2:
            return pc
    return None


def clash_rate(notes, chords):
    """音符 pc 不在当前和弦 tones ±1 半音邻音(经过音)内的时长占比。"""
    sounded = clash = 0.0
    for onset, midi, dur in notes:
        ch = chord_at(chords, onset)
        sounded += dur
        if ch is None:
            continue
        root, tones = ch
        mpc = int(midi) % 12
        rel = (mpc - root) % 12
        allowed = set(tones)
        for t in tones:
            allowed.add((t + 1) % 12)
            allowed.add((t - 1) % 12)
        if rel not in allowed:
            clash += dur
    return round(clash / sounded, 3) if sounded > 0 else None


def rms_db(y, sr, t1, t2):
    a, b = int((t1 + 0.5) * sr), min(len(y), int(t2 * sr))
    if b - a < sr // 2:
        return None
    seg = y[a:b]
    v = float(librosa.amplitude_to_db(np.sqrt((seg ** 2).mean()) + 1e-9))
    return v if v > -55 else None  # 近静音窗口无效


def analyze(name, wav, tdir):
    tdir = Path(tdir)
    meta, vocal, secs = parse_abc(str(tdir / "score.abc"))
    chords = load_chords(tdir / "chord.lab")
    y, sr = librosa.load(str(wav), sr=SR, mono=True)
    dur = len(y) / sr

    def sec_span(names):
        for n, t in secs:
            if n.lower() in names:
                t2 = next((tt for nn, tt in secs if tt > t), dur)
                for nn, tt in secs:
                    if nn.lower() in names and tt > t:
                        t2 = tt
                        break
                return t, min(t2, dur)
        return None

    out = dict(name=name)
    iv, vv, cc = sec_span({"intro"}), sec_span({"verse"}), sec_span({"chorus"})
    r_intro = rms_db(y, sr, *iv) if iv else None
    r_verse = rms_db(y, sr, *vv) if vv else None
    r_chorus = rms_db(y, sr, *cc) if cc else None
    if r_chorus is not None and r_intro is not None:
        out["arc_chorus_minus_intro_db"] = round(r_chorus - r_intro, 1)
    if r_chorus is not None and r_verse is not None:
        out["arc_chorus_minus_verse_db"] = round(r_chorus - r_verse, 1)
    out["clash_rate"] = clash_rate(vocal, chords)
    return out


def main():
    corpus = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
    entries = []
    for m in corpus:
        stem = Path(m["wav"]).stem
        tdir = ROOT / "corpus" / "transcriptions" / stem
        if (tdir / "chord.lab").exists():
            entries.append((f"corpus::{stem}", m["wav"], tdir, "corpus"))
    creation_map = [
        ("creation::深夜便利店(520)", "eval/wavs/create_520.wav",
         "creations/20260920_043058/self_score"),
        ("creation::selfopt731", "eval/wavs/create_731.wav",
         "creations/selfopt_20260920_061912/round1_seed731/realized"),
        ("creation::cover江南", "eval/wavs/cover_jiangnan.wav", "covers/jiangnan/cover_score"),
    ]
    for name, wav, tdir in creation_map:
        if (ROOT / tdir / "chord.lab").exists():
            entries.append((name, str(ROOT / wav), tdir, "creation"))
    demo_tdir = ROOT / "eval" / "transcriptions" / "demo_tonight"
    if (demo_tdir / "chord.lab").exists():
        entries.append(("creation::今晚不眠(官方示例曲)", str(ROOT / "eval/wavs/demo_tonight.wav"),
                        demo_tdir, "creation"))

    res = {"corpus": [], "creation": []}
    for name, wav, tdir, grp in entries:
        try:
            r = analyze(name, wav, tdir)
            res[grp].append(r)
            print(f"ok {name}: {r}", flush=True)
        except Exception as e:
            print(f"FAIL {name}: {str(e)[:100]}", flush=True)
    (ROOT / "eval" / "arc_consonance.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SAVED eval/arc_consonance.json", flush=True)


if __name__ == "__main__":
    main()
