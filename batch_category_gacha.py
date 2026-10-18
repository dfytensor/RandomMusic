# -*- coding: utf-8 -*-
"""类别感知批量抽卡：已渲染文件直接重评分（曲库同款方法），再按类别分爬坡
- 01-07,09-12: novel_songs/batch_gacha/<no>/seed*.flac
- 08: novel_songs/triple_emergence/seed*.flac（17 个已知地形）
- 评分: score_audio(78s副歌段) → classify → 落同类 P25~75 每维25分
- 平局: 距同类中位Σ越小越好
- 缺失邻域 → render_one.py 补渲染
输出 eval/batch_category_gacha_report.md
"""
import json, subprocess, sys, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
REPORT = ROOT / "eval" / "batch_category_gacha_report.md"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
TITLES = {"01":"欠天地一笔","02":"平账人","03":"白户","04":"剁手魔","05":"心疼说成省钱",
          "06":"第320代记账人","07":"规矩","08":"月供日","09":"断供","10":"糕与药包",
          "11":"烂尾劫","12":"新契"}

def _work(args):
    no, f = args
    try:
        import librosa
        sys.path.insert(0, str(ROOT))
        from category_score import score_audio, classify, score_vs_category
        y, _ = librosa.load(str(f), sr=SR, mono=True, offset=78, duration=30)
        r = score_audio(y, SR)
        sc = score_vs_category(r)
        return (no, f.stem, r, sc, None)
    except Exception as e:
        return (no, f.stem, None, None, f"{type(e).__name__}")

def collect():
    jobs = []
    for no in TITLES:
        d = ROOT/"novel_songs"/"batch_gacha"/no
        if no == "08":
            d = ROOT/"novel_songs"/"triple_emergence"
        if d.exists():
            jobs += [(no, f) for f in sorted(d.glob("seed*.flac"))]
    out = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for no, stem, r, sc, err in ex.map(_work, jobs):
            out.setdefault(no, {})[stem] = (r, sc, err)
    return out

def dist_to_median(r, cat):
    sys.path.insert(0, str(ROOT))
    from category_score import _TG
    t = _TG[cat]["targets"]
    s = abs((r["orn"] or 0)-t["orn"])/0.5 + abs((r["gap"] or 0)-t["gap"])/1.0
    s += abs(r["cent"]-t["cent"])/600 + abs(r["dyn"]-t["dyn"])/8
    return round(s, 3)

def render(no, seed, out):
    try:
        rr = subprocess.run([PY, str(ROOT/"render_one.py"), no, str(seed), str(out)],
                            capture_output=True, text=True, timeout=1500)
        return "OK" in (rr.stdout or "")
    except Exception:
        return False

def main():
    res = collect()
    lines = ["# 类别感知批量抽卡（曲库同款方法·P25~75 命中制·满分100）", "",
             "| 歌 | 类 | 冠军seed | 得分 | 命中 | orn | gapCV | cent | dyn | BPM |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for no in TITLES:
        pool = res.get(no, {})
        pool = {s: v for s, v in pool.items() if v[0] is not None}
        if not pool:
            lines.append(f"| {no} {TITLES[no]} | ? | 无有效文件 | | | | | | | |")
            continue
        def key(item):
            stem, (r, sc, _) = item
            return (sc["score"], -dist_to_median(r, sc["cat"]))
        best_stem, (br, bsc, _) = max(pool.items(), key=key)
        # 爬坡：冠军±1 若缺失则补渲染重评
        seed_num = int(best_stem.replace("seed", ""))
        for nb in [seed_num-1, seed_num+1]:
            nkey = f"seed{nb}"
            if nkey in pool:
                continue
            d = ROOT/"novel_songs"/"triple_emergence" if no=="08" else ROOT/"novel_songs"/"batch_gacha"/no
            out = d/f"{nkey}.flac"
            if render(no, nb, out):
                no2, st2, r2, sc2, err2 = _work((no, out))
                if r2:
                    pool[nkey] = (r2, sc2, None)
                    if (sc2["score"], -dist_to_median(r2, sc2["cat"])) > key((best_stem, (br,bsc,None))):
                        best_stem, br, bsc = nkey, r2, sc2
        h = bsc["hits"]
        hstr = "".join(["装" if h["orn"] else "", "息" if h["gap"] else "",
                        "质" if h["cent"] else "", "动" if h["dyn"] else ""]) or "-"
        lines.append(f"| {no} {TITLES[no]} | {bsc['cat']} | {best_stem} | **{bsc['score']}** "
                     f"| {bsc['n_hits']}/4 [{hstr}] | {br['orn']} | {br['gap']} | {br['cent']} | {br['dyn']} | {br['bpm']} |")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        print(f"{no}: {best_stem} score={bsc['score']} cat={bsc['cat']}", flush=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
