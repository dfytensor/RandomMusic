# -*- coding: utf-8 -*-
"""灵感 -> 规格 -> YuE2 生成 -> 自检。用法:
    python generate_from_inspiration.py [inspiration.json]
inspiration.json 字段见 map §3/§6；lyrics 可内联或 lyrics_file 指向文本。
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

CLUSTERS = {
    "C1": ("T1", 69, 67, "Mandopop piano ballad, gentle arpeggiated piano, warm string pads, "
                          "soft fretless bass, brushed drums"),
    "C2": ("T2", 96, 59, "Mandopop R&B pop, piano and clean electric guitar, warm groove bass, "
                         "soft pop drums with ghost notes and rimshots, light strings layer"),
    "C3": ("T3", 129, 65, "Mandopop pop-rock anthem, overdriven and clean electric guitars, driving bass, "
                          "energetic stadium drums, anthemic backing in final chorus"),
    "C4": ("T4", 129, 40, "Mandopop dance pop, four-on-the-floor kick, bright pluck synths, sidechained pads, "
                          "punchy saturated bass, catchy synth hook"),
}
MOOD2CLUSTER = [
    (r"温暖|治愈|陪伴|城市|回忆|便利店|流动|心动|甜蜜", "C2"),
    (r"心碎|思念|遗憾|眼泪|告别|暗恋|孤独|深夜", "C1"),
    (r"青春|热血|梦想|毕业|奔跑|呐喊|不服|燃", "C3"),
    (r"派对|夏夜|舞池|微醺|发光|跳舞", "C4"),
]
TEMPO2BPM = [(r"很慢|慢板|静态|呼吸", -4), (r"慢", -2), (r"中速|流动|叙事|摇曳", 0), (r"快|燃|跳跃|躁", +6)]
VOCAL_WORDS = {"女": "bright female lead vocal", "男": "emotive male lead vocal",
               "对唱": "male-female duet vocals", "气声": "airy breathy vocal",
               "沙哑": "raspy expressive male vocal"}
SCENE_RE = re.compile(r"[，,。；;、]")

TEMPLATE_BARS = {
    "T1": [("Intro", 4), ("Verse", 8), ("Pre-Chorus", 4), ("Chorus", 8), ("Interlude", 4),
           ("Verse", 8), ("Pre-Chorus", 4), ("Chorus", 8), ("Bridge", 4), ("Chorus", 8), ("Outro", 8)],
    "T2": [("Intro", 4), ("Verse", 16), ("Pre-Chorus", 8), ("Chorus", 16), ("Interlude", 4),
           ("Verse", 12), ("Pre-Chorus", 8), ("Chorus", 16), ("Bridge", 4), ("Chorus", 12), ("Outro", 4)],
    "T3": [("Intro", 8), ("Verse", 16), ("Pre-Chorus", 8), ("Chorus", 16), ("Interlude", 8),
           ("Verse", 12), ("Pre-Chorus", 8), ("Chorus", 16), ("Bridge", 8), ("Chorus", 16), ("Outro", 8)],
    "T4": [("Intro", 8), ("Verse", 16), ("Pre-Chorus", 8), ("Chorus", 16), ("Interlude", 8),
           ("Verse", 12), ("Pre-Chorus", 8), ("Chorus", 16), ("Bridge", 4), ("Chorus", 16), ("Outro", 8)],
}
FRIENDLY_KEYS = {"major": ["C", "G", "D", "A", "E"], "minor": ["A", "E", "D", "G", "C"]}


def build_spec(inp):
    text = inp.get("inspiration", "")
    cluster = inp.get("cluster")
    if not cluster:
        for pat, c in MOOD2CLUSTER:
            if re.search(pat, text):
                cluster = c
                break
        cluster = cluster or "C2"
    template, base_bpm, _, inst = CLUSTERS[cluster]

    bpm = inp.get("bpm")
    if not bpm:
        delta = 0
        for pat, d in TEMPO2BPM:
            if re.search(pat, text):
                delta = d
                break
        bpm = base_bpm + delta
    bpm = max(64, min(148, int(bpm)))

    mode = inp.get("mode") or ("minor" if re.search(r"忧郁|遗憾|孤独|雨|眼泪|思念|心碎", text) else "major")
    key = inp.get("key") or FRIENDLY_KEYS[mode][0]
    vocal = inp.get("vocal") or next((v for k, v in VOCAL_WORDS.items() if k in text), "emotive male lead vocal")

    scenes = [w for w in SCENE_RE.split(text) if 2 <= len(w.strip()) <= 8][:2]
    en_scenes = [s.strip() for s in scenes if re.fullmatch(r"[A-Za-z0-9 ]+", s.strip())]
    atmosphere = f", {' and '.join(en_scenes)} atmosphere" if en_scenes else ""
    style = (f"{inst}, {{VOCAL}}, R&B-influenced phrasing, romantic and introspective, "
             f"{key} {mode}, {bpm} BPM{atmosphere}") if cluster == "C2" else \
            (f"{inst}, {{VOCAL}}, {key} {mode}, {bpm} BPM{atmosphere}")
    style = style.replace("{VOCAL}", vocal)

    bars = sum(b for _, b in TEMPLATE_BARS[template])
    expected_dur = round(bars * 240 / bpm)
    return dict(cluster=cluster, template=template, bpm=bpm, key=key, mode=mode, vocal=vocal,
                style=style, atmosphere=atmosphere.strip(", "), expected_dur=expected_dur, bars=bars)


def load_lyrics(inp, spec):
    if inp.get("lyrics"):
        return inp["lyrics"], True
    f = inp.get("lyrics_file")
    if f and Path(f).exists():
        return Path(f).read_text(encoding="utf-8"), True
    tags = []
    for name, _ in TEMPLATE_BARS[spec["template"]]:
        if not tags or tags[-1] != f"[{name}]":
            tags.append(f"[{name}]")
    return "\n\n".join(tags) + "\n", False


def main():
    inp_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "inspiration.json"
    inp = json.loads(inp_path.read_text(encoding="utf-8"))
    spec = build_spec(inp)
    lyrics, has_lyrics = load_lyrics(inp, spec)
    out_dir = ROOT / "creations" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = inp.get("seeds", [inp.get("seed", 20260920)])
    (out_dir / "spec.json").write_text(json.dumps(
        dict(spec, seeds=seeds, has_lyrics=has_lyrics, inspiration=inp.get("inspiration", "")),
        ensure_ascii=False, indent=1), encoding="utf-8")
    if not has_lyrics:
        (out_dir / "lyrics_scaffold.txt").write_text(lyrics, encoding="utf-8")
        print("[WARN] 未提供歌词，仅生成段落脚手架，建议补词后重跑", flush=True)

    print("SPEC:", json.dumps(spec, ensure_ascii=False), flush=True)

    from yue2 import YuE2Pipeline
    pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                        vae=str(ROOT / "models" / "YuE2-Vae"),
                                        device="cuda", progress=False)
    results = []
    for i, seed in enumerate(seeds):
        print(f"[candidate {i+1}/{len(seeds)}] seed={seed} generating ...", flush=True)
        song = pipe(style=spec["style"], lyrics=lyrics, cot="full", seed=seed)
        p = out_dir / f"song_seed{seed}.flac"
        song.save(str(p))
        song.save_artifacts(str(out_dir / f"artifacts_seed{seed}"))
        results.append(str(p))
    pipe.close()

    # 无参考自检
    print("running reference-free checks ...", flush=True)
    cmd = [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "reference_free_check.py"),
           *[str(r) for r in results], "--expected", str(spec["expected_dur"]),
           "--out", str(out_dir / "quality_report.md")]
    subprocess.run(cmd, check=True)
    print(f"DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
