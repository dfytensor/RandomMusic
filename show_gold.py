# -*- coding: utf-8 -*-
import json
d = json.load(open(r'F:\yinyue\eval\v3_validation.json', encoding='utf-8'))
det = sorted(d['gold_detail'], key=lambda x: -x['pop'])
print('流行度Top10 -> 我们的分:')
for r in det[:10]:
    print(f"  pop={r['pop']:5.1f} score={r['score']:5.1f} {r['song']}")
print('流行度Bottom10 -> 我们的分:')
for r in det[-10:]:
    print(f"  pop={r['pop']:5.1f} score={r['score']:5.1f} {r['song']}")
# 分数排序 vs 流行度排序的一致性
import numpy as np
by_score = sorted(det, key=lambda x: -x['score'])
print('我们分数Top10:')
for r in by_score[:10]:
    print(f"  score={r['score']:5.1f} pop={r['pop']:5.1f} {r['song']}")
