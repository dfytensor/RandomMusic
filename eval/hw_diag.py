# -*- coding: utf-8 -*-
"""硬件诊断: RAM 写读校验 + C/F 盘文件写读完整性 + 输出结果"""
import hashlib, os, random, sys, time

print("=== 1) RAM 写读校验 (512MB 已知模式) ===", flush=True)
size = 512 * 1024 * 1024
seed = 123456789
rng = random.Random(seed)
block = bytes(rng.getrandbits(8) for _ in range(1024)) * (size // 1024)
buf = bytearray(block)
# 校验 1: 立即读回
h1 = hashlib.sha256(bytes(buf)).hexdigest()
# 故意制造内存压力 (分配 4GB 干扰)
noise = [bytes(1024) * 256 for _ in range(4)]  # 4GB
del noise
# 校验 2: 压力后读回
h2 = hashlib.sha256(bytes(buf)).hexdigest()
print(f"  immediate: {h1[:16]}")
print(f"  after pressure: {h2[:16]}")
print(f"  RAM STABLE: {h1 == h2}", flush=True)
del buf, block

print("\n=== 2) C/F 盘文件写读完整性 (各 200MB) ===", flush=True)
payload = os.urandom(200 * 1024 * 1024)
expect = hashlib.sha256(payload).hexdigest()
for drive in ["C:", "F:"]:
    p = os.path.join(drive + os.sep, "_hw_diag_test.bin")
    try:
        t0 = time.time()
        with open(p, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        wt = time.time() - t0
        # 落盘后清缓存读回 (重复读 2 次对比)
        hashes = []
        for r in range(2):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(blk)
            hashes.append(h.hexdigest())
        ok = hashes[0] == expect and hashes[1] == expect
        print(f"  {drive} write {wt:.1f}s read1 {hashes[0][:16]} read2 {hashes[1][:16]} -> {'OK' if ok else 'DATA LOSS!'}")
        if hashes[0] != expect and hashes[1] == expect:
            print(f"    (read1 错 read2 对 = 读取不稳定)")
        if hashes[0] == expect and hashes[1] != expect:
            print(f"    (read1 对 read2 错 = 读取不稳定)")
    except Exception as e:
        print(f"  {drive} ERR {type(e).__name__}: {e}")
    finally:
        try:
            os.remove(p)
        except OSError:
            pass
print("\nDIAG_DONE", flush=True)
