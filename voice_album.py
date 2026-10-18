# -*- coding: utf-8 -*-
"""12 冠军专辑声纹替换管线: Demucs 分离 -> Seed-VC(f0) 换嗓 -> 重混
用法: python voice_album.py --ref 目标歌手干声.wav [--song 08] [--steps 30]
"""
import argparse, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "seed-vc"))

CHAMPS = [
    ("01", ROOT / r"novel_songs\batch_gacha\01_v2\seed58888.flac"),
    ("02", ROOT / r"novel_songs\batch_gacha\02_wide\seed7778.flac"),
    ("03", ROOT / r"novel_songs\batch_gacha\03_v2\seed61001.flac"),
    ("04", ROOT / r"novel_songs\batch_gacha\04\seed90406.flac"),
    ("05", ROOT / r"novel_songs\batch_gacha\05_wide\seed1111.flac"),
    ("06", ROOT / r"novel_songs\batch_gacha\06_wide\seed3690.flac"),
    ("07", ROOT / r"novel_songs\batch_gacha\07_adj\seed1357_r1.flac"),
    ("08", ROOT / r"novel_songs\batch_gacha\11\seed91104.flac"),
    ("09", ROOT / r"novel_songs\batch_gacha\09\seed90904.flac"),
    ("10", ROOT / r"novel_songs\batch_gacha\10\seed91007.flac"),
    ("11", ROOT / r"novel_songs\batch_gacha\11\seed91103.flac"),
    ("12", ROOT / r"novel_songs\batch_gacha\12\seed91202.flac"),
]
OUT_DIR = ROOT / "album_voice"
DEMUCS_DIR = ROOT / "album_voice" / "_stems"


def separate_one(song_path: Path, device="cuda") -> tuple[Path, Path]:
    """Demucs htdemucs: 返回 (vocals_path, accomp_path)"""
    from demucs.api import Separator
    out = DEMUCS_DIR / song_path.stem
    out.mkdir(parents=True, exist_ok=True)
    voc = out / "vocals.wav"
    acc = out / "accomp.wav"
    if voc.exists() and acc.exists():
        return voc, acc
    sep = Separator(model="htdemucs", device=device)
    origin, stems = sep.separate_audio_file(str(song_path))
    import torch, soundfile as sf
    vocals = stems["vocals"]
    accomp = origin - vocals
    sf.write(str(voc), vocals.cpu().numpy().T, sep.samplerate)
    sf.write(str(acc), accomp.cpu().numpy().T, sep.samplerate)
    return voc, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="目标歌手干声 1-30s wav/flac")
    ap.add_argument("--song", default=None, help="只处理指定编号如 08")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    import numpy as np, soundfile as sf, torch, librosa
    from seed_vc_wrapper import SeedVCWrapper

    print("加载 Seed-VC 模型 (首次会下载权重) ...", flush=True)
    svc = SeedVCWrapper(device=torch.device(args.device))
    print("模型就绪", flush=True)

    todo = [(n, p) for n, p in CHAMPS if args.song is None or n == args.song]
    for no, song in todo:
        if not song.exists():
            print(f"[{no}] 缺文件 {song}", flush=True); continue
        print(f"[{no}] {song.name} 分离中 ...", flush=True)
        voc, acc = separate_one(song, device=args.device)
        print(f"[{no}] 声纹转换中 (steps={args.steps}) ...", flush=True)
        gen = svc.convert_voice(
            source=str(voc), target=str(args.ref),
            diffusion_steps=args.steps, length_adjust=1.0,
            inference_cfg_rate=0.7, f0_condition=True,
            auto_f0_adjust=True, pitch_shift=0, stream_output=False,
        )
        try:
            while True:
                next(gen)  # stream_output=False 时不会产出, 只等 StopIteration
        except StopIteration as e:
            wav = e.value
        sr = svc.sr_f0  # 44100
        acc_w, acc_sr = sf.read(str(acc), dtype="float32")  # (n, ch)
        if acc_sr != sr:
            import torchaudio as _ta
            t = _ta.functional.resample(
                torch.from_numpy(acc_w.T), acc_sr, sr).numpy().T
            acc_w = t
        vocal_t = np.asarray(wav, dtype=np.float32)  # (n,)
        n = min(len(vocal_t), acc_w.shape[0])
        mix = acc_w[:n] + vocal_t[:n, None]
        peak = float(np.abs(mix).max())
        if peak > 1.0:
            mix = mix / peak * 0.98
        dst = OUT_DIR / f"{no}_seedvc.flac"
        sf.write(str(dst), mix, sr)
        print(f"[{no}] 完成 -> {dst.name} ({n/sr:.1f}s)", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
