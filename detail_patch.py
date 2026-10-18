# -*- coding: utf-8 -*-
"""补齐缺失的 popularity: 慢速 + 405退避"""
import json, urllib.request, urllib.parse, time, sys
from pathlib import Path

P = Path(r'F:\yinyue\eval\netease_popularity_v2.json')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
           'Referer': 'https://music.163.com'}
PAUSE = 2.0

def fetch_pop(sid):
    url = 'https://music.163.com/api/song/detail?' + urllib.parse.urlencode({'ids': '[%s]' % sid})
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=12) as r:
        data = json.loads(r.read().decode('utf-8'))
    if data.get('code') == 405:
        return 'THROTTLED'
    songs = data.get('songs')
    if not songs:
        return None
    s = songs[0]
    return dict(popularity=float(s.get('popularity') or 0),
                score=s.get('score'), pop_name=s.get('name'))

d = json.loads(P.read_text(encoding='utf-8'))
todo = [(k, v['netease_id']) for k, v in d.items()
        if isinstance(v, dict) and v.get('netease_id') and not v.get('popularity')]
print('待补 %d' % len(todo), flush=True)

ok = fail = throttle_wait = 0
t0 = time.time()
for i, (k, sid) in enumerate(todo):
    tries = 0
    while tries < 5:
        try:
            r = fetch_pop(sid)
        except Exception as e:
            r = None; print('  exc %s' % str(e)[:50], flush=True)
        if r == 'THROTTLED':
            w = 60
            print('  [%d/%d] 限流, 等 %ds...' % (i+1, len(todo), w), flush=True)
            time.sleep(w); throttle_wait += w; tries += 1
            continue
        if isinstance(r, dict):
            d[k].update(r)
            ok += 1
        else:
            fail += 1
        break
    else:
        fail += 1
        print('  %s 放弃' % k[:20], flush=True)
    if (i+1) % 20 == 0:
        P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
        print('[%d/%d] ok=%d fail=%d el=%.0fs' % (i+1, len(todo), ok, fail, time.time()-t0), flush=True)
    time.sleep(PAUSE)

P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
with_pop = sum(1 for v in d.values() if isinstance(v, dict) and v.get('popularity'))
print('完成: 本轮 ok=%d fail=%d throttle_wait=%ds | 总计有pop=%d' % (ok, fail, throttle_wait, with_pop), flush=True)
