
import json, sys, torch, soundfile as sf
from pathlib import Path
from yue2 import YuE2Pipeline
ROOT = Path(r"F:\yinyue")
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    out = pipe(prompt=cfg["style"], lyrics=cfg["lyrics"], cot="full", seed=cfg["seed"])
    out.save(str(Path(sys.argv[2]) / "out.flac"))
finally:
    pipe.close()
print("OK")
