# -*- coding: utf-8 -*-
"""模型文件读取稳定性采样: 多偏移点读 hash + corrupt_bak 复测"""
import hashlib, sys

def sample(path, label):
    pts = [0, 1 << 30, 2 << 30, 3 << 30, 4 << 30, 5 << 30, 6 << 30, 7 << 30]  # 每 1GB 采样 1MB
    print(f"--- {label} ---", flush=True)
    try:
        with open(path, "rb") as f:
            size = f.seek(0, 2)
            for off in pts:
                if off >= size:
                    continue
                f.seek(off)
                d = f.read(1024 * 1024)
                print(f"  @{off>>30}GB: {hashlib.sha256(d).hexdigest()[:16]}", flush=True)
    except Exception as e:
        print(f"  ERR {e}", flush=True)

sample(r"C:\yue2_model\models\m-a-p--YuE2-3B\snapshots\master\model.safetensors", "C copy #1")
sample(r"C:\yue2_model\models\m-a-p--YuE2-3B\snapshots\master\model.safetensors", "C copy #2 (重复读)")
sample(r"F:\yinyue\models\YuE2-3B\model.safetensors.corrupt_bak", "F corrupt_bak #1")
sample(r"F:\yinyue\models\YuE2-3B\model.safetensors", "F current #1")
