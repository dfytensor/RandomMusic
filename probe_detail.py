# -*- coding: utf-8 -*-
import json, urllib.request, urllib.parse, random
from pathlib import Path
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 'Referer': 'https://music.163.com'}
d = json.loads(Path(r'F:\yinyue\eval\netease_popularity_v2.json').read_text(encoding='utf-8'))
missing = [(k, v['netease_id']) for k, v in d.items()
           if isinstance(v, dict) and v.get('netease_id') and not v.get('popularity')]
print('missing pop:', len(missing), flush=True)
random.seed(1)
for k, sid in random.sample(missing, 6):
    url = 'https://music.163.com/api/song/detail?' + urllib.parse.urlencode({'ids': '[%s]' % sid})
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode('utf-8'))
        s = data['songs'][0]
        print('%-24s id=%s name=%-14s pop=%s score=%s' % (
            k[:22], sid, s.get('name', '')[:12], s.get('popularity'), s.get('score')), flush=True)
    except Exception as e:
        print('%-24s ERR %s' % (k[:22], str(e)[:60]), flush=True)
