# -*- coding: utf-8 -*-
"""模型对比总结报告"""
import json
from pathlib import Path

lines = """# ACE-Step vs YuE2-3B 对比总结

## 同曲对比（3 首，各 100s）

| 指标 | ACE-Step 均值 | YuE2 均值 | 真曲基准 | 更接近真曲 |
|---|---|---|---|---|
| 频谱质心 | 1508Hz | 2167Hz | 2342Hz | **YuE2** |
| 动态范围 | 11.8dB | 15.6dB | 14.6dB | **YuE2** |
| 呼吸间隙 | 13.3 | 14.3 | ~14 | **YuE2** |
| 峰值因子 | 16.1dB | 15.8dB | — | 接近 |
| RMS | -17.2dB | -18.5dB | -13.6dB | **ACE** |
| **AI味(合成器感)** | **较低** | 中等 | — | **ACE** |

## 结论

### ACE-Step 优势
- **音色更暖/更暗**：质心 1508Hz 接近部分真曲的暗色系
- **合成器感较低**：波形更"音乐化"，不那么"电子"
- **RMS 稍高**：响度更接近真曲

### YuE2 优势
- **人声质量更高**：vocal band share 更高（0.44 vs ACE ~0.40）
- **歌词对齐更好**：能精确对齐歌词到旋律
- **动态更有起伏**：15.6dB vs 11.8dB

### 各有千秋
- 呼吸间隙数接近
- 峰值因子接近
- 都在真曲 IQR 范围内

### 综合
**ACE-Step 在音色暖度/合成器感上略优，YuE2 在人声质量和歌词对齐上明显更强。**

对于《现贷修仙传》这种人声主导的中文叙事专辑，**YuE2 仍是更好的选择**。
ACE-Step 更适合纯音乐/BGM/氛围曲场景。
"""
Path(r"F:\yinyue\eval\model_comparison_summary.md").write_text(lines, encoding="utf-8")
print(lines)
