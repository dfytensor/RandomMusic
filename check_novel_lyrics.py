# -*- coding: utf-8 -*-
"""批量歌词检查: novel_songs_manifest -> lyrics_check.check()"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lyrics_check import check
from novel_songs_manifest import SONGS

all_ok = True
for s in SONGS:
    checks, summary = check(s["lyrics"], s["template"], s["title"])
    npass = sum(1 for _, ok, _ in checks if ok)
    status = "OK " if npass == len(checks) else "FAIL"
    print(f"[{status}] {s['no']} {s['title']} {npass}/{len(checks)} rhyme={summary['rhyme_coverage']} "
          f"len_med={summary['line_len_median']}")
    for name, ok, detail in checks:
        if not ok:
            print(f"    ❌ {name}: {detail}")
            all_ok = False
print("ALL PASS" if all_ok else "HAS FAILURES")
