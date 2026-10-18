# -*- coding: utf-8 -*-
"""三重涌现互相调节机制：缺哪维 → 调 prompt(演唱/声学杠杆) → 同seed重渲染 → 评分 → 迭代
维度因果（prompt A/B 实证）:
  orn高→加 straight tones 剔 ornamentation | gap低(太规律)→加 rubato irregular breathing
  cent低(太暗)→剔 sparse piano 加 bright | cent高(太亮)→剔 bright 加 warm dark
  dyn低→加 quiet verses building to explosive | dyn高→加 even dynamics gentle
"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, r'F:\yinyue')
import librosa
from category_score import score_audio, score_vs_category, _TG

PY = r'F:\yinyue\.venv\Scripts\python.exe'
SR = 22050
CHAMPS = {'02': 7778, '03': 63001, '05': 1111, '06': 3690, '07': 1357}

def score(f):
    y, _ = librosa.load(str(f), sr=SR, mono=True, offset=78, duration=30)
    r = score_audio(y)
    return r, score_vs_category(r)

def fix_plan(r, cat):
    """缺维 → (修正词, 剔除词组)"""
    t = _TG[cat]["targets"]
    def rng(key):
        lo, hi = _TG[cat][key].split("~"); return float(lo), float(hi)
    mods, drops = [], []
    if r["orn"] and r["orn"] > rng("orn_p25_75")[1]:
        mods.append("absolutely no melisma, plain flat delivery")
    if r["gap"] is not None:
        lo, hi = rng("gap_p25_75")
        if r["gap"] < lo:
            mods.append("rubato, free timing, uneven breath pauses")
        elif r["gap"] > hi:
            mods.append("steady pulse, even breathing")
    lo, hi = rng("cent_p25_75")
    if r["cent"] < lo:
        drops.append("sparse piano with silences")
        mods.append("bright clear forward tone")
    elif r["cent"] > hi:
        mods.append("dark mellow rounded tone")
    lo, hi = rng("dyn_p25_75")
    if r["dyn"] < lo:
        mods.append("hushed verse then explosive cathartic chorus, big drama")
    elif r["dyn"] > hi:
        mods.append("even gentle dynamics, no dramatic swings")
    return ", ".join(mods), ";".join(drops)

for no, sd in CHAMPS.items():
    f = Path(rf'F:\yinyue\novel_songs\batch_gacha\{no}_wide\seed{sd}.flac')
    if not f.exists():
        f = Path(rf'F:\yinyue\novel_songs\batch_gacha\{no}\seed{sd}.flac')
    try:
        r0, sc0 = score(f)
    except Exception as e:
        print(f'{no}: base ERR {type(e).__name__}', flush=True); continue
    print(f'{no} seed{sd} base: {sc0["score"]} ({sc0["n_hits"]}/4) {sc0["hits"]}', flush=True)
    outd = Path(rf'F:\yinyue\novel_songs\batch_gacha\{no}_adj'); outd.mkdir(parents=True, exist_ok=True)
    best = (sc0["score"], f, sc0, r0)
    for rnd in range(1, 3):
        mods, drops = fix_plan(best[3], best[2]["cat"])
        if not mods and not drops:
            print(f'{no}: 无缺维可调', flush=True); break
        out = outd / f'seed{sd}_r{rnd}.flac'
        rr = subprocess.run([PY, r'F:\yinyue\render_adj.py', no, str(sd), str(out), mods, drops],
                            capture_output=True, text=True, timeout=1500)
        if 'OK' not in (rr.stdout or ''):
            print(f'{no} r{rnd}: RENDER FAIL', flush=True); break
        r1, sc1 = score(out)
        print(f'{no} r{rnd} adj[{mods[:40]}]: {sc1["score"]} ({sc1["n_hits"]}/4)', flush=True)
        if sc1["score"] > best[0]:
            best = (sc1["score"], out, sc1, r1)
        if best[0] >= 100:
            break
    print(f'== {no} FINAL {best[0]} -> {best[1].name} ==', flush=True)
