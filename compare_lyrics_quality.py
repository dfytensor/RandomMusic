# -*- coding: utf-8 -*-
"""YuE2 vs ACE-Step 唱词清晰度对比：ASR 转写 vs 原始歌词"""
import json, subprocess, sys, warnings, os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

ASR_PY = ROOT / ".venv-asr" / "Scripts" / "python.exe"

# ===== 1. 提取副歌片段 =====
def extract_chorus(flac_path, out_wav, duration=30):
    y, sr = librosa.load(str(flac_path), sr=24000, mono=True, duration=duration)
    # 取中后段（副歌区域约 55%-85%）
    start = int(len(y) * 0.55)
    end = min(len(y), start + int(duration * sr))
    sf.write(out_wav, y[start:end], sr, subtype="PCM_16")
    return out_wav

# ===== 2. ASR 转写 =====
def asr_transcribe(wav_path):
    script = ROOT / "asr_transcribe.py"
    script.write_text('''
import torch, json, sys
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained(
    str(r"F:\\yinyue\\models\\Qwen3-ASR-1.7B"),
    dtype=torch.bfloat16, device_map="cuda:0")
wav = sys.argv[1]
result = model.transcribe(audio=wav, language="Chinese")
text = result[0].text if isinstance(result, list) else result.text
print(json.dumps({"text": text}, ensure_ascii=False))
''', encoding="utf-8")
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1")
    r = subprocess.run([str(ASR_PY), str(script), str(wav_path)],
                       capture_output=True, text=True, env=env)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])["text"]
    except:
        return ""

# ===== 3. CER 计算 =====
def cer(reference, hypothesis):
    """字符错误率"""
    ref = re.sub(r"[^\u4e00-\u9fff]", "", reference)
    hyp = re.sub(r"[^\u4e00-\u9fff]", "", hypothesis)
    if not ref: return 1.0
    # 编辑距离
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(ref[i-1]!=hyp[j-1]))
    return round(dp[n][m] / max(n,1), 3)

# ===== 主流程 =====
from novel_songs_manifest import SONGS
from hit_score import TRANSCRIPTS
import librosa
import soundfile as sf

final = ROOT / "novel_songs" / "final"
ace_dir = ROOT / "ace_step_output"

# 三首歌的参考歌词
lyrics_map = {}
for s in SONGS:
    lyrics_map[s["title"]] = s["lyrics"]

# ACE-Step 的歌词（从 ace_generate.py 的 TEST 列表提取）
ACE_LYRICS = {
    "月供日": "[Verse]\n十五的更鼓敲过城楼\n家家户数灵砂的愁\n我提着灯走过桥头\n水牌上写满赊与留\n\n[Pre-Chorus]\n月供日不许回头\n息上加息没有尽头\n可街口糕点的香悠悠\n人间烟火替我挡愁\n\n[Chorus]\n月供日\n月亮也来催息\n还的是钱\n赎的是命\n我按时把自己交出去\n换她一碗药的热气",
    "欠天地一笔": "[Verse]\n灵气枯了四百年\n我出生就没有名字\n代书摊一张桌子\n替别人写了一千封信\n\n[Pre-Chorus]\n更鼓敲了三遍\n季查使在门外站\n我把恐惧折成灵砂\n一个一个数着看\n\n[Chorus]\n欠天地一笔\n我用寿元还息\n阴阳账簿翻开我自己\n每一页都是我的姓名\n欠天地一笔\n不还不行",
    "第320代记账人": "[Verse]\n封账之露凝在阵眼中\n十二年不能相认的痛\n他把名字浮进我的梦\n把算学传进我掌中\n\n[Chorus]\n第320代记账人\n账未记完人已归尘\n债在子偿 字字千钧\n你算清天下的息\n唯独没算 我想你\n来世再叫你一声父亲",
}

results = []
import time
t0 = time.time()

for song_key, ace_lyrics in ACE_LYRICS.items():
    # ACE-Step 版
    ace_flac = ace_dir / f"ace_{song_key[:2]}_{song_key}.flac"
    # 也找一下实际文件名
    for f in ace_dir.glob("*.flac"):
        if song_key[:2] in f.name:
            ace_flac = f; break
    
    if not ace_flac.exists():
        continue
    
    # YuE2 版
    yue_files = list((ROOT/"novel_songs"/"final").glob(f"{song_key[:2]}_*.flac"))
    if not yue_files:
        continue
    yue_flac = yue_files[0]
    
    # 提取两版的副歌
    import librosa
    for model_name, flac_path in [("YuE2", yue_flac), ("ACE", ace_flac)]:
        y, sr = librosa.load(str(flac_path), sr=24000, mono=True)
        dur = len(y) / sr
        # 取 55%-85%（副歌区域）
        start = int(dur * 0.55 * sr)
        end = min(len(y), int(dur * 0.85 * sr))
        seg = y[start:end]
        clip = ROOT / "eval" / f"asr_{model_name}_{song_key[:2]}.wav"
        sf.write(clip, seg, sr, subtype="PCM_16")
        
        # ASR
        env = dict(os.environ)
        env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1")
        script = ROOT / "asr_single.py"
        script.write_text('''
import torch, json, sys
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained(
    str(r"F:\\yinyue\\models\\Qwen3-ASR-1.7B"),
    dtype=torch.bfloat16, device_map="cuda:0")
result = model.transcribe(audio=sys.argv[1], language="Chinese")
text = result[0].text if isinstance(result, list) else result.text
print(json.dumps({"text": text}, ensure_ascii=False))
''', encoding="utf-8")
        r = subprocess.run([str(ASR_PY), str(script), str(clip)],
                           capture_output=True, text=True, env=env)
        try:
            hyp = json.loads(r.stdout.strip().splitlines()[-1])["text"]
        except:
            hyp = "(ASR failed)"
        
        # CER
        import re
        ref_clean = re.sub(r"[^\u4e00-\u9fff]", "", ace_lyrics)
        hyp_clean = re.sub(r"[^\u4e00-\u9fff]", "", hyp)
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, ref_clean, hyp_clean).ratio()
        cer = round((1 - ratio) * 100, 1)
        
        results.append(dict(model=model_name, song=song_key, cer=cer, asr_text=hyp[:50]))
        print(f"  {model_name} {song_key}: CER={cer}% ASR前20字={hyp[:20]}", flush=True)
        time.sleep(1)

# 汇总
lines = ["# YuE2 vs ACE-Step 唱词清晰度对比（ASR+CER）", ""]
lines.append("| 模型 | 歌曲 | CER% | ASR前20字 |")
lines.append("|---|---|---|---|")
for r in results:
    grade = "✅" if r["cer"] <= 20 else ("🟡" if r["cer"] <= 35 else "❌")
    lines.append(f"| {r['model']} {r['song']} | {r['cer']}% | {grade} {r['asr_text'][:20]} |")

avg_yue = np.mean([r["cer"] for r in results if r["model"]=="YuE2"]) if any(r["model"]=="YuE2" for r in results) else 0
avg_ace = np.mean([r["cer"] for r in results if r["model"]=="ACE"]) if any(r["model"]=="ACE" for r in results) else 0
lines += ["", f"YuE2 平均 CER: {avg_yue:.1f}%", f"ACE 平均 CER: {avg_ace:.1f}%"]

(ROOT/"eval"/"lyric_intelligibility.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines), flush=True)
