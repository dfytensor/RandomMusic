# -*- coding: utf-8 -*-
"""改进歌名解析 + 重爬 NetEase 热度数据"""
import json, re, time, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path(__file__).parent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://music.163.com",
}


def smart_parse(filename):
    """智能解析歌名：处理各种格式"""
    clean = filename.replace(".mp3", "").replace(".flac", "").strip()
    
    # 常见分隔符
    seps = [" - ", " — ", " – ", " -", "- ", "—", "–", "_", "、"]
    
    artist, song = "", clean
    for sep in seps:
        if sep in clean:
            parts = clean.split(sep, 1)
            artist = parts[0].strip()
            song = parts[1].strip()
            break
    
    # 去掉括号内的备注
    song = re.sub(r"[\(（【\[].*?[\)）\]】]", "", song).strip()
    # 去掉 feat/ft
    song = re.sub(r"\s*(feat|ft)\..*$", "", song, flags=re.I).strip()
    # 去掉 LIVE/Remix 等标记
    song = re.sub(r"\s*[\(（]?\s*(Live|Remix|Acoustic|Demo)\s*[\)）]?\s*$", "", song, flags=re.I).strip()
    
    return artist, song


def search_netease(query, limit=3):
    url = "https://music.163.com/api/search/get"
    data = urllib.parse.urlencode({"s": query, "type": 1, "limit": limit}).encode()
    req = urllib.request.Request(url, data=data, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        songs = data.get("result", {}).get("songs", []) if data.get("result") else []
        results = []
        for s in songs:
            artists = ", ".join(a["name"] for a in s.get("artists", []))
            results.append(dict(
                id=s["id"], name=s.get("name", ""), artists=artists,
                popularity=s.get("popularity", 0),
                commentCount=s.get("commentCount", 0),
                fee=s.get("fee", 0),
            ))
        return results
    except Exception as e:
        return []


def get_detail(song_id):
    url = "https://music.163.com/api/song/detail"
    params = urllib.parse.urlencode({"ids": f"[{song_id}]"})
    req = urllib.request.Request(f"{url}?{params}", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        songs = data.get("songs", [])
        if songs:
            s = songs[0]
            return dict(
                commentCount=s.get("commentCount", 0),
                shareCount=s.get("shareCount", 0),
                likedCount=s.get("likedCount", 0) if "likedCount" in s else s.get("likeCount", 0),
                popularity=s.get("popularity", 0),
            )
    except:
        pass
    return {}


def best_match(song_name, artist_hint, results):
    """搜索结果选最佳匹配(严格: 歌名必须双向包含, 否则视为未命中)"""
    if not results:
        return None
    def title_ok(r):
        n = r["name"].lower(); s = song_name.lower()
        return (s in n) or (n in s) or (s[:3] and s[:3] in n)
    # 歌名匹配 + 艺人匹配优先
    for r in results:
        if title_ok(r) and artist_hint and artist_hint.lower() in r["artists"].lower():
            return r
    # 仅歌名匹配
    for r in results:
        if title_ok(r):
            return r
    return None  # 无可靠匹配 → not_found, 宁缺毋滥


def main():
    feats = json.loads((ROOT / "corpus" / "features.json").read_text(encoding="utf-8"))
    out_p = ROOT / "eval" / "netease_popularity_v2.json"
    done = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}

    # 加载之前的成功结果
    old_p = ROOT / "eval" / "netease_popularity.json"
    if old_p.exists():
        old = json.loads(old_p.read_text(encoding="utf-8"))
        for k, v in old.items():
            if "netease_id" in v:
                done[k] = v

    todo = [n for n in sorted(feats) if n not in done]
    print(f"待爬: {len(todo)} / {len(feats)}（已有 {len(done)}）", flush=True)

    import time
    t0 = time.time()
    found = 0
    for i, name in enumerate(todo):
        artist, song = smart_parse(name)
        if not song:
            song = name.replace(".mp3", "").strip()
        
        # 搜索：先搜"歌名 艺人"，不行再搜"歌名"
        results = search_netease(f"{song} {artist}")
        if not results:
            results = search_netease(song)
        
        best = best_match(song, artist, results)
        if best:
            detail = get_detail(best["id"])
            entry = dict(
                netease_id=best["id"],
                netease_name=best["name"],
                artists=best["artists"],
                search_song=song,
                search_artist=artist,
                **(detail or {}),
            )
            done[name] = entry
            found += 1
        else:
            done[name] = {"not_found": True}
        
        if (i + 1) % 25 == 0 or i == len(todo) - 1:
            out_p.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
            el = time.time() - t0
            ok = sum(1 for v in done.values() if "netease_id" in v)
            print(f"[{i+1}/{len(todo)}] found={ok} el={el:.0f}s", flush=True)
        
        time.sleep(0.4)

    out_p.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    
    # 统计
    ok_data = [v for v in done.values() if "netease_id" in v]
    pops = [v.get("popularity", 0) for v in ok_data]
    comments = [v.get("commentCount", 0) for v in ok_data]
    import numpy as np
    print(f"\n完成: {found}/{len(todo)} 首（总 {len(ok_data)}）", flush=True)
    if pops:
        print(f"Popularity: 中位 {np.median(pops):.0f} · P25 {np.percentile(pops,25):.0f} · P75 {np.percentile(pops,75):.0f}")
    if comments:
        print(f"评论数: 中位 {np.median(comments):.0f} · 最大 {max(comments)}")


if __name__ == "__main__":
    main()
