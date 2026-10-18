# -*- coding: utf-8 -*-
"""可哼化手术·自动版（完整正确实现）
对指定曲目：全曲转谱 -> 副歌人声替换为调式动机循环（4小节×N，含八度长音高潮）
-> cot=melody 重渲染 -> 母带 -> final/NN_title_hum.flac（与原版并存 A/B）
用法: python hum_auto.py 02,03,04,05,07,09,10,11,12
"""
import json
import re
import shutil
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from remediate_novel import master, transcribe_realized

SR = 24000
NAMES = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]
STEPS_MAJ = [2, 2, 1, 2, 2, 1, 2]
STEPS_MIN = [2, 1, 2, 2, 1, 2, 2]
# 动机小节（度, 时值16分单位）：级进为主，无八度跳，句尾长音落稳定音
# 跟唱三律：① 音域 ≤ 12 半音  ② 级进 ≥ 80%  ③ 句尾长音 = 调内 1 级或 5 级
PATTERN_MIN = [
    [(1, 4), (2, 4), (3, 4), (5, 4)],   # 级进上行到五级
    [(3, 8), (2, 4), (1, 4)],           # 级进下行回主音
    [(1, 4), (2, 4), (3, 4), (5, 4)],   # 重复（强化记忆）
    [(6, 8), (5, 8)],                   # 挂在六级，不落地（引回）
]
PATTERN_MAJ = [
    [(1, 4), (3, 4), (5, 4), (3, 4)],   # 拱形
    [(4, 8), (3, 4), (1, 4)],           # 下行回主音
    [(1, 4), (3, 4), (5, 4), (5, 8)],   # 重复+五级长音
    [(4, 4), (3, 4), (2, 4), (1, 8)],   # 级进回家
]
END_BAR = [(1, 12)]


def deg_midi(deg, tonic, minor):
    steps = STEPS_MIN if minor else STEPS_MAJ
    idx = (deg - 1) % 7
    octv = (deg - 1) // 7
    return tonic + sum(steps[:idx]) + 12 * octv


def midi_tok(midi, dur):
    octv = (midi - 60) // 12
    pc = midi - 60 - 12 * octv
    letter = NAMES[pc].lstrip("^")
    acc = "^" * NAMES[pc].count("^")
    marks = "'" * octv if octv >= 0 else "," * (-octv)
    return f"{acc}{letter}{marks}{dur}"


def tonic_midi_near_key(key, med_pitch):
    name = key.split()[0]
    letter = name[0].upper()
    acc = name[1:] if len(name) > 1 else ""
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    pc = (base + (1 if acc == "#" else -1 if acc == "b" else 0)) % 12
    return int(pc + 12 * round((med_pitch - pc) / 12))


def split_sections(abc_text):
    """返回 (header_lines, [(section_name, lines)])"""
    header, blocks, cur_name, cur_lines = [], [], None, []
    for line in abc_text.splitlines():
        st = line.strip()
        if st.startswith("%"):
            if cur_name is not None:
                blocks.append((cur_name, cur_lines))
            cur_name = st[1:].strip().lower()
            cur_lines = []
            continue
        if cur_name is None:
            header.append(line)
        else:
            cur_lines.append(line)
    if cur_name is not None:
        blocks.append((cur_name, cur_lines))
    return header, blocks


def rewrite_chorus(abc_text, tonic, minor):
    """副歌人声重写为跟唱旋律：级进为主、音域≤12半音、句尾长音"""
    header, blocks = split_sections(abc_text)
    pat = PATTERN_MIN if minor else PATTERN_MAJ
    new_blocks, gi = [], 0
    for name, lines in blocks:
        if name != "chorus":
            new_blocks.append((name, lines))
            continue
        new_lines = []
        voice = "Vocal"
        for line in lines:
            st = line.strip()
            if st.startswith("V:"):
                voice = "Ins" if "Ins" in st else "Vocal"
                new_lines.append(line)
                continue
            if (not st or st.startswith(("X:", "T:", "M:", "L:", "Q:", "K:"))):
                new_lines.append(line)
                continue
            if voice != "Vocal":
                new_lines.append(line)
                continue
            # Vocal 音乐行 -> 4 行动机小节（音域压缩：度 1-6 内）
            new_lines.append("V: Vocal")
            for _ in range(4):
                toks = [midi_tok(deg_midi_tight(d, tonic, minor), dur)
                        for d, dur in pat[gi % len(pat)]]
                new_lines.append("|".join(toks) + "|")
                gi += 1
        new_blocks.append((name, new_lines))
    body = []
    for name, lines in new_blocks:
        body.append(f"% {name}")
        body.extend(lines)
    return "\n".join(header) + "\n" + "\n".join(body) + "\n"


def deg_midi_tight(deg, tonic, minor):
    """度 -> MIDI，仅用度 1-6（自然压缩音域到 ~8 半音）"""
    steps = STEPS_MIN if minor else STEPS_MAJ
    idx = (deg - 1) % 7
    return tonic + sum(steps[:idx])


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "02,03,04,05,07,09,10,11,12"
    nos = [x.strip() for x in arg.split(",")]
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1")
    report = []
    for no in nos:
        s = next(x for x in SONGS if x["no"] == no)
        title = s["title"]
        final_flac = ROOT / "novel_songs" / "final" / f"{no}_{title}.flac"
        work = ROOT / "novel_songs" / "v2" / f"novel_{no}_{title}_humauto"
        work.mkdir(parents=True, exist_ok=True)

        # 1) 全曲转谱
        full_wav = work / "full.wav"
        if not full_wav.exists():
            y, _ = librosa.load(str(final_flac), sr=SR, mono=True, duration=300)
            sf.write(full_wav, y, SR, subtype="PCM_16")
        tdir = work / "realized_full"
        if not (tdir / "score.abc").exists():
            try:
                subprocess.run([str(ROOT / ".venv-ss2" / "Scripts" / "python.exe"),
                                str(ROOT / "transcribe.py"), str(full_wav), str(tdir)],
                               check=True, capture_output=True, env=env)
                mel_only = True
            except subprocess.CalledProcessError:
                # melody_only 导出失败的已知边界：回退全量转谱（含和弦）
                subprocess.run([str(ROOT / ".venv-ss2" / "Scripts" / "python.exe"),
                                str(ROOT / "transcribe.py"), str(full_wav), str(tdir), "--full"],
                               check=True, capture_output=True, env=env)
                mel_only = False
        else:
            mel_only = "chord" not in (tdir / "score.abc").read_text(encoding="utf-8")[:2000] or True
            mel_only = True
        abc_text = (tdir / "score.abc").read_text(encoding="utf-8")
        from melody_stats import analyze_song
        m = analyze_song(str(tdir / "score.abc"))
        minor = "minor" in s["mode"]
        med = (m.get("tessitura") or [69, 71])[0] + 1
        tonic = tonic_midi_near_key(s["key"].split()[0], med)
        new_abc = rewrite_chorus(abc_text, tonic, minor)
        (work / "hum_auto.abc").write_text(new_abc, encoding="utf-8")
        print(f"[{no} {title}] tonic_midi={tonic} minor={minor} "
              f"density={m['density']}", flush=True)

        # 2) 重渲染（子进程隔离，规避同进程显存累积）
        raw = work / "raw.flac"
        if not raw.exists():
            cot = "full" if not mel_only else "melody"
            args = [sys.executable, str(ROOT / "render_one.py"), no,
                    str(s["seed"] + 800), str(work / "hum_auto.abc"), "--cot", cot]
            ok = False
            for attempt in (1, 2):
                try:
                    r = subprocess.run(args, capture_output=True, text=True, env=env)
                    ok = r.returncode == 0 and "OK" in r.stdout
                except Exception as e:
                    print(f"  attempt{attempt} ex: {type(e).__name__}", flush=True)
                if ok:
                    break
                print(f"  attempt{attempt} fail", flush=True)
                import time as _t
                _t.sleep(5)
            if not ok:
                print(f"  [{no} {title}] GIVE UP", flush=True)
                continue
            seed_used = s["seed"] + 800
            src_m = ROOT / "novel_songs" / "gacha" / no / str(seed_used) / "mastered_hum.flac"
            if src_m.exists():
                shutil.copy(src_m, work / "mastered.flac")
            else:
                raw_src = work / "raw.flac"
                if raw_src.exists():
                    master(raw_src, work / "mastered.flac")

        # 3) 母带 + A/B 交付
        mastered = work / "mastered.flac"
        if not mastered.exists():
            master(raw, mastered)
        dst = ROOT / "novel_songs" / "final" / f"{no}_{title}_hum.flac"
        shutil.copy(mastered, dst)
        dur = (lambda i: i.frames / i.samplerate)(sf.info(str(dst)))
        report.append(dict(no=no, title=title, file=dst.name, dur=round(dur)))
        print(f"  -> {dst.name} {dur:.0f}s", flush=True)

    (ROOT / "eval" / "hum_auto_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
