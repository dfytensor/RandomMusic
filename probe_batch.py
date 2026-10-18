# -*- coding: utf-8 -*-
"""测 detail 批量 ids + 限流状态"""
import json, urllib.request, urllib.parse, time
from pathlib import Path
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
           'Referer': 'https://music.163.com'}

def fetch(ids):
    url = 'https://music.163.com/api/song/detail?' + urllib.parse.urlencode({'ids': json.dumps(ids)})
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.loads(r.read().decode('utf-8'))

# 1) 限流是否恢复(单个)
try:
    d = fetch([186016])
    print('single:', 'songs' in d, d.get('code'), flush=True)
except Exception as e:
    print('single ERR', e, flush=True)
time.sleep(3)
# 2) 批量 5 个
d = json.loads(Path(r'F:\yinyue\eval\netease_popularity_v2.json').read_text(encoding='utf-8'))
ids = [v['netease_id'] for v in d.values() if isinstance(v, dict) and v.get('netease_id')][:5]
try:
    d2 = fetch(ids)
    print('batch5:', 'songs' in d2 and len(d2.get('songs', [])), d2.get('code'), flush=True)
    if d2.get('songs'):
        for s in d2['songs'][:5]:
            print('  ', s.get('name'), 'pop=', s.get('popularity'), flush=True)
except Exception as e:
    print('batch ERR', e, flush=True)
