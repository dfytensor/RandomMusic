# -*- coding: utf-8 -*-
"""耐心补齐 popularity: 限流探活 → 批量50/请求 → 405长等待, 后台运行"""
import json, urllib.request, urllib.parse, time
from pathlib import Path

P = Path(r'F:\yinyue\eval\netease_popularity_v2.json')
LOG = Path(r'F:\yinyue\detail_patch_log.txt')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
           'Referer': 'https://music.163.com'}

def log(msg):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def fetch(ids):
    url = 'https://music.163.com/api/song/detail?' + urllib.parse.urlencode({'ids': json.dumps(ids)})
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode('utf-8'))

d = json.loads(P.read_text(encoding='utf-8'))
missing = [(k, v['netease_id']) for k, v in d.items()
           if isinstance(v, dict) and v.get('netease_id') and not v.get('popularity')]
log('待补 %d' % len(missing))

# 1) 探活
rounds = 0
while rounds < 60:
    try:
        r = fetch([186016])
        if r.get('songs'):
            log('限流解除'); break
        log('probe code=%s' % r.get('code'))
    except Exception as e:
        log('probe ERR %s' % str(e)[:50])
    rounds += 1
    time.sleep(300)
else:
    log('60轮探活失败(5h), 退出'); raise SystemExit(1)

# 2) 批量补齐
ids_all = [(k, sid) for k, sid in missing]
CH = 50
i = 0
ok = 0
while i < len(ids_all):
    chunk = ids_all[i:i+CH]
    try:
        r = fetch([sid for _, sid in chunk])
    except Exception as e:
        log('chunk %d ERR %s, 等300s' % (i//CH, str(e)[:50]))
        time.sleep(300); continue
    if r.get('code') == 405 or not r.get('songs'):
        log('chunk %d 限流, 等300s' % (i//CH))
        time.sleep(300); continue
    got = {s['id']: s for s in r['songs']}
    for k, sid in chunk:
        s = got.get(sid)
        if s:
            d[k]['popularity'] = float(s.get('popularity') or 0)
            d[k]['pop_name'] = s.get('name')
            ok += 1
    P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    log('chunk %d/%d 本轮累计ok=%d' % (i//CH + 1, (len(ids_all)+CH-1)//CH, ok))
    i += CH
    time.sleep(4)

with_pop = sum(1 for v in d.values() if isinstance(v, dict) and v.get('popularity'))
log('全部完成: 有pop=%d' % with_pop)
