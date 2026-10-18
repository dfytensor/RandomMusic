# -*- coding: utf-8 -*-
"""Critic: 对照语料参考区间评审旋律 (map §7)。供 CLI 与 self_optimize 复用。
用法: python critic_melody.py <score.abc> [melody_stats.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from melody_stats import analyze_song


def load_agg(stats_path=None):
    stats_path = Path(stats_path) if stats_path else ROOT / "corpus" / "melody_stats.json"
    return json.loads(stats_path.read_text(encoding="utf-8"))["aggregate"]


def evaluate(abc_path, agg):
    r = analyze_song(abc_path)
    if not r:
        return None

    def rng(key):
        s = agg[key]
        return s["p25"], s["p75"], s["med"]

    checks = []

    def add(name, val, key, floor=None, tol=0.0):
        lo, hi, med = rng(key)
        if val is None:
            checks.append(dict(name=name, val=None, ok=False, note=f"参考 {lo}-{hi}"))
            return
        if floor is not None:
            ok = val >= max(lo - tol, floor)
            note = f"参考 {lo}-{hi} · 下限 {max(lo - tol, floor)}"
        else:
            ok = lo - tol <= val <= hi + tol
            note = f"参考 {lo}-{hi}"
        checks.append(dict(name=name, val=val, ok=bool(ok), note=note))

    add("音符密度 (音/小节)", r["density"], "density_notes_per_bar", tol=1.0)
    add("人声音域 (半音)", r["pitch_range"], "pitch_range_semitones", tol=3)
    add("跳进率", r["leap_ratio"], "leap_ratio", tol=0.05)
    add("motif 重复率", r["motif_repeat"], "motif_repeat", floor=0.20)
    add("副歌平均抬升 (半音)", r["lift_mean"], "chorus_lift_mean_semitones", floor=1.0)
    add("人声休止占比", r["rest_ratio"], "rest_ratio", tol=0.05)
    return dict(metrics=r, structure_chain=r["chain"], total_bars=r["total_bars"],
                checks=checks, passed=sum(1 for c in checks if c["ok"]), total=len(checks))


def format_report(abc_path, ev):
    lines = [f"# Critic 旋律评审: {abc_path}",
             f"结构链: {ev['structure_chain']} · 总小节: {ev['total_bars']}", "",
             "| 检查项 | 数值 | 判定 | 参考区间 |",
             "|---|---|---|---|"]
    for c in ev["checks"]:
        v = "-" if c["val"] is None else c["val"]
        lines.append(f"| {c['name']} | {v} | {'✅' if c['ok'] else '❌'} | {c['note']} |")
    lines += ["", f"通过 {ev['passed']}/{ev['total']}"]
    lines.append("结论: 旋律质量达标" if ev["passed"] == ev["total"]
                 else "结论: 存在偏离，按 map §7 Critic 规则定向修改后重生成")
    return "\n".join(lines)


def main():
    abc_path = sys.argv[1]
    ev = evaluate(abc_path, load_agg(sys.argv[2] if len(sys.argv) > 2 else None))
    print(format_report(abc_path, ev))


if __name__ == "__main__":
    main()
