# -*- coding: utf-8 -*-
"""全库 NetEase 播放量爬虫：571 首 → 真实热度标签
断点续跑，输出 eval/netease_popularity.json
"""
import json, re, time, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path(__file__).parent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://music.163.com",
}


def search_song(song_name, artist="", limit=1):
    query = f"{song_name} {artist}".strip()
    url = "https://music.163.com/api/search/get/web"
    params = urllib.parse.urlencode({"s": query, "type": 1, "limit": limit})
    req = urllib.request.Request(f"{url}?{params}", headers=HEADERS)
    import urllib.request as ur
    try:
        with ur.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        songs = data.get("result", {}).get("songs", [])
        if songs:
            return songs[0]["id"], songs[0].get("name", ""), songs[0].get("popularity", 0)
    except Exception:
        pass
    return None, None, 0


def main():
    feats = json.loads((ROOT / "corpus" / "features.json").read_text(encoding="utf-8"))
    out_p = ROOT / "eval" / "netease_popularity.json"
    done = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}

    todo = [n for n in sorted(feats) if n not in done]
    print(f"总计 {len(todo)} 首待爬", flush=True)

    t0 = time.time()
    for i, name in enumerate(todo):
        # 解析歌名和艺人
        clean = name.replace(".mp3", "").strip()
        parts = re.split(r"\s*[-—]\s*", clean, maxsplit=1)
        artist = parts[0].strip() if parts else ""
        song = parts[1].strip() if len(parts) > 1 else clean
        
        sid, sname, pop = search_song(song, artist)
        if sid:
            # 获取详情（评论数等）
            url = "https://music.163.com/api/song/detail"
            params = urllib.parse.urlencode({"ids": f"[{sid}]"})
            req = urllib.request.Request(f"{url}?{params}", headers=HEADERS)
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                songs = data.get("songs", [])
                if songs:
                    s = songs[0]
                    done[name] = {
                        "netease_id": sid,
                        "netease_name": s.get("name", ""),
                        "popularity": s.get("popularity", 0),
                        "comment_count": s.get("commentCount", 0),
                        "fee": s.get("fee", 0),
                    }
            except Exception as e:
                done[name] = {"error": str(e)[:60]}
        else:
            done[name] = {"not_found": True}
        
        if (i + 1) % 20 == 0 or i == len(todo) - 1:
            out_p.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
            el = time.time() - t0
            found = sum(1 for v in done.values() if "netease_id" in v)
            print(f"[{i+1}/{len(todo)}] found={found} el={el:.0f}s", flush=True)
        
        time.sleep(0.5)  # 限速

    # 统计
    found = sum(1 for v in done.values() if "netease_id" in v)
    pops = [v["popularity"] for v in done.values() if "netease_id" in v and "popularity" in v]
    comments = [v["comment_count"] for v in done.values() if "netease_id" in v and "comment_count" in v]
    print(f"\n完成: {found}/{len(todo)} 首")
    if pops:
        import numpy as np
        print(f"Popularity: 中位 {np.median(pops):.0f} · P25 {np.percentile(pops,25):.0f} · P75 {np.percentile(pops,75):.0f}")
    if comments:
        print(f"评论数: 中位 {np.median(comments):.0f} · 最大 {max(comments)}")
    print("SAVED", flush=True)


if __name__ == "__main__":
    main()
