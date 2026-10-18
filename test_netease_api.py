# -*- coding: utf-8 -*-
"""网易云音乐播放量爬取（测试版）：搜歌 -> 获取播放量
NetEase Cloud Music 公开API（无需登录）
"""
import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).parent

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://music.163.com",
}


def search_song(song_name, artist="", limit=1):
    """搜索歌曲获取 ID"""
    query = f"{song_name} {artist}".strip()
    url = "https://music.163.com/api/search/get/web"
    params = urllib.parse.urlencode({"s": query, "type": 1, "limit": limit, "offset": 0})
    req = urllib.request.Request(f"{url}?{params}", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        songs = data.get("result", {}).get("songs", [])
        if songs:
            return songs[0]["id"], songs[0].get("name", "")
    except Exception as e:
        print(f"  search err: {str(e)[:60]}")
    return None, None


def get_play_count(song_id):
    """获取歌曲详情（含播放量相关数据）"""
    url = "https://music.163.com/api/song/detail"
    params = urllib.parse.urlencode({"ids": f"[{song_id}]", "ids": f"[{song_id}]"})
    req = urllib.request.Request(f"{url}?{params}", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        songs = data.get("songs", [])
        if songs:
            s = songs[0]
            # NetEase 不直接返回播放量，但返回评论数/收藏数/分享数作为热度代理
            return {
                "comment_count": s.get("commentCount", 0),
                "share_count": s.get("shareCount", 0),
                "like_count": s.get("likedCount", 0),
                "popularity": s.get("popularity", 0),
                "fee": s.get("fee", 0),
            }
    except Exception as e:
        print(f"  detail err: {str(e)[:60]}")
    return None


# ===== 测试 5 首 =====
test_songs = [
    ("林俊杰", "江南"),
    ("周杰伦", "青花瓷"),
    ("信乐团", "死了都要爱"),
    ("张韶涵", "隐形的翅膀"),
    ("黄雅莉", "蝴蝶泉边"),
]

for artist, song in test_songs:
    print(f"\n--- {artist} - {song} ---")
    sid, sname = search_song(song, artist)
    if sid:
        print(f"  ID: {sid}  Name: {sname}")
        detail = get_play_count(sid)
        if detail:
            print(f"  评论: {detail['comment_count']}  收藏: {detail['like_count']}  pop: {detail['popularity']}")
    else:
        print("  未找到")

    time.sleep(1)  # 限速
