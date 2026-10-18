# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r'F:\yinyue')
from binary_gacha import triple_score_file

for s in [91097, 91096, 91102, 91101]:
    f = Path(rf'F:\yinyue\novel_songs\triple_emergence\seed{s}.flac')
    r = triple_score_file(f)
    print(f"seed{s}: orn={r['orn']}/s gap_cv={r['gap']} cent={r['cent']}Hz dyn={r['dyn']}dB "
          f"| vocal={r['vocal']} acoustic={r['acoustic']} total={r['total']}")
print("target: orn<=0.15 gap_cv~1.38 cent~2135 dyn~14.6")
