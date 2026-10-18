# -*- coding: utf-8 -*-
"""新爬 559 匹配的标题质量审计: 严格标题匹配率 + 与 v2 backup 对比"""
import json, re
from pathlib import Path

d = json.loads(Path(r'F:\yinyue\eval\netease_popularity_v2.json').read_text(encoding='utf-8'))

def titles_of(k):
    s = re.sub(r'\.mp3$', '', k)
    s = re.sub(r'[《\[（(].*?[》\])）)]', '', s)
    s = re.sub(r'电影.*|电视剧.*|片尾曲|主题曲|插曲', '', s)
    parts = re.split(r'\s*-\s*|_|\s+', s)
    parts = [p for p in parts if len(p) >= 2]
    return parts  # 候选标题片段

def strict_ok(k, v):
    name = v.get('netease_name', '')
    frags = titles_of(k)
    if not frags: return False
    # 任一主要片段(≥3字优先)与匹配名双向包含
    for f in sorted(frags, key=len, reverse=True):
        if len(f) >= 3 and (f in name or name in f): return True
        if len(f) >= 2 and f in name: return True
    # 尾段常见歌名
    tail = frags[-1]
    return tail in name or name in tail

with_id = [(k, v) for k, v in d.items() if isinstance(v, dict) and v.get('netease_id')]
ok = [(k, v) for k, v in with_id if strict_ok(k, v)]
bad = [(k, v) for k, v in with_id if not strict_ok(k, v)]
print('with_id=%d  strict_ok=%d  suspect=%d' % (len(with_id), len(ok), len(bad)))
print('\n可疑样本(前20):')
for k, v in bad[:20]:
    print('  期望[%s] -> 匹配[%s]' % (k[:30], v.get('netease_name', '')[:30]))
