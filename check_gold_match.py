# -*- coding: utf-8 -*-
import json, re
gold = json.load(open(r'F:\yinyue\eval\netease_popularity_v2.json', encoding='utf-8'))
valid = {k: v for k, v in gold.items() if v.get('popularity') and v.get('netease_id')}
print(f'共 {len(valid)}')
bad = 0
for k, v in sorted(valid.items(), key=lambda x: x[1]['popularity']):
    expected = re.sub(r'\.mp3$', '', k)
    expected_title = expected.split(' - ')[-1].split('-')[-1].strip()
    got = v['netease_name']
    # 标题是否含在期望里 / 期望是否含在 got 里
    ok = expected_title[:3] in got or got[:3] in expected_title
    if not ok:
        bad += 1
        print(f"SUSPECT pop={v['popularity']:5.1f} 期望[{expected}] -> 匹配[{v['netease_name']}]")
print(f'suspect={bad}/{len(valid)}')
print('\n周杰伦相关:')
for k, v in valid.items():
    if '周杰' in k:
        print(f"  {k} pop={v['popularity']} -> netease[{v['netease_name']}]")
