# -*- coding: utf-8 -*-
"""Reviser: 根据 Critic 诊断对 plan ABC 做定向手术 (map §7 规则 1-4)。
每步操作后重测目标指标，改善才保留，否则回滚。
"""
import sys
from pathlib import Path

import abc_edit

sys.path.insert(0, str(Path(__file__).parent))
from critic_melody import evaluate, load_agg
from melody_stats import analyze_song

import tempfile
import os


def _eval_text(text, agg):
    fd, tmp = tempfile.mkstemp(suffix=".abc")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        return evaluate(tmp, agg)
    finally:
        os.unlink(tmp)


def _metric(ev, key_map, name):
    m = ev["metrics"]
    return key_map.get(name)


def fix(plan_text, agg, max_steps=6):
    """返回 (new_text, log)。log 为每步操作的描述与效果。"""
    text = plan_text
    log = []
    ev = _eval_text(text, agg)

    def val_of(ev, key):
        return ev["metrics"].get(key)

    steps = [
        # (诊断条件, 目标指标键, 操作函数, 方向)
        ("lift_low", "lift_mean",
         lambda t: abc_edit.op_transpose(t, ["chorus"], 2), "副歌抬升不足", "+2"),
        ("range_high", "pitch_range",
         lambda t: abc_edit.op_fold_range(t, agg["pitch_range_semitones"]["med"] - 9,
                                          agg["pitch_range_semitones"]["med"] + 10), "音域过宽", "fold"),
        ("range_low", "pitch_range",
         lambda t: abc_edit.op_transpose(t, ["chorus"], 2), "音域过窄", "chorus+2"),
        ("leap_high", "leap_ratio",
         lambda t: abc_edit.op_reduce_leaps(t, agg["pitch_range_semitones"]["med"] - 10,
                                            agg["pitch_range_semitones"]["med"] + 11), "跳进过多", "smooth"),
        ("density_high", "density",
         lambda t: abc_edit.op_tie_merge(t, ["verse", "pre-chorus", "chorus"]), "密度过高", "merge"),
        ("density_low", "density",
         lambda t: abc_edit.op_split_long(t, ["verse", "chorus"]), "密度过低", "split"),
    ]

    # 判定诊断
    checks = {c["name"]: c for c in ev["checks"]}
    issues = set()
    if not checks["副歌平均抬升 (半音)"]["ok"]:
        issues.add("lift_low")
    if not checks["人声音域 (半音)"]["ok"] and val_of(ev, "pitch_range") is not None:
        med = agg["pitch_range_semitones"]["med"]
        issues.add("range_high" if val_of(ev, "pitch_range") > med else "range_low")
    if not checks["跳进率"]["ok"] and val_of(ev, "leap_ratio") > agg["leap_ratio"]["p75"]:
        issues.add("leap_high")
    if not checks["音符密度 (音/小节)"]["ok"]:
        med = agg["density_notes_per_bar"]["med"]
        if val_of(ev, "density") > med:
            issues.add("density_high")
        else:
            issues.add("density_low")
    if not checks["motif 重复率"]["ok"]:
        log.append({"op": "motif_low", "result": "无自动修复，建议换 seed"})
    if not checks["人声休止占比"]["ok"]:
        log.append({"op": "rest_ratio_out", "result": "无自动修复，随其他操作间接改善"})

    improved_any = False
    for key, metric, op, desc, tag in steps:
        if key not in issues or len(log) >= max_steps:
            continue
        before = val_of(ev, metric)
        cand = op(text)
        ev2 = _eval_text(cand, agg)
        after = val_of(ev2, metric)
        if after is None:
            continue
        better = False
        if key == "lift_low":
            better = before is None or after > before
        elif key == "range_high":
            better = after < before
        elif key == "range_low":
            better = after > before
        elif key == "leap_high":
            better = after < before
        elif key == "density_high":
            better = abs(after - agg["density_notes_per_bar"]["med"]) < abs(before - agg["density_notes_per_bar"]["med"])
        elif key == "density_low":
            better = abs(after - agg["density_notes_per_bar"]["med"]) < abs(before - agg["density_notes_per_bar"]["med"])
        if better and ev2["passed"] >= ev["passed"]:
            text = cand
            ev = ev2
            improved_any = True
            log.append({"op": tag, "target": metric, "before": before, "after": after,
                        "result": "applied"})
        else:
            log.append({"op": tag, "target": metric, "before": before, "after": after,
                        "result": "rejected(no improvement)"})
    return text, log, ev


if __name__ == "__main__":
    import json
    plan_path = sys.argv[1]
    out_path = sys.argv[2]
    agg = load_agg(sys.argv[3] if len(sys.argv) > 3 else None)
    text = Path(plan_path).read_text(encoding="utf-8")
    new_text, log, ev = fix(text, agg)
    Path(out_path).write_text(new_text, encoding="utf-8")
    print(json.dumps({"passed_after": f"{ev['passed']}/{ev['total']}", "log": log},
                     ensure_ascii=False, indent=1))
