"""把配图插进飞书实验记录文档的指定小节后面。

飞书贴图三步：建 image 块 -> 上传素材拿到 token -> PATCH 把 token 写进块。
插入位置按小节的标题块定位，插到该节第三段之后；从后往前插，前面的下标才不会错位。

用法: python3 scripts/doc_insert_images.py
"""
import json
import os
import pathlib
import sys

import requests

DOC = "KHp0dmJlcoY9ATxSqfhcyDR4nLb"
D = "https://open.feishu.cn"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# (小节标题开头, 图片, 图注)
PLAN = [
    ("评测换了一把尺子", "reports/chart_yardstick.png", "左：8 局一点的旧尺子，阴影是 95% 区间；右：40 局一点的新尺子"),
    ("叶子上补一眼吃子", "reports/chart_q1.png", "叶子吃子修正 + 240 次模拟这一轮"),
    ("策略主干从 256 加宽到 512", "reports/chart_wide.png", "加宽这一轮，五个评测点"),
    ("把策略损失拆成三块", "reports/chart_loss.png", "左：一份损失的三块构成；右：同一局面两次搜索的差距"),
    ("把搜索深度扫一遍", "reports/chart_depth.png", "左：只改模拟次数；右：棋力来自哪里"),
]


def env():
    e = {}
    p = pathlib.Path.home() / ".hermes/.env"
    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            e[k.strip()] = v.strip().strip('"').strip("'")
    return e


def token(e):
    r = requests.post(f"{D}/open-apis/auth/v3/tenant_access_token/internal",
                      json={"app_id": e["FEISHU_APP_ID"], "app_secret": e["FEISHU_APP_SECRET"]},
                      timeout=20)
    r.raise_for_status()
    return r.json()["tenant_access_token"]


def block_text(b):
    for key in ("text", "heading1", "heading2", "heading3", "heading4", "heading5"):
        if key in b:
            return "".join(x.get("text_run", {}).get("content", "")
                           for x in b[key].get("elements", []))
    return ""


def main():
    e = env()
    tok = token(e)
    H = {"Authorization": f"Bearer {tok}"}

    r = requests.get(f"{D}/open-apis/docx/v1/documents/{DOC}/blocks",
                     headers=H, params={"page_size": 500}, timeout=30).json()
    items = r["data"]["items"]
    print("blocks:", len(items))

    jobs = []
    for prefix, img, caption in PLAN:
        img_path = os.path.join(ROOT, img)
        if not os.path.exists(img_path):
            print("跳过（缺图）:", img)
            continue
        hit = [i for i, b in enumerate(items) if block_text(b).startswith(prefix)]
        if not hit:
            print("跳过（找不到小节）:", prefix)
            continue
        idx = hit[0]
        # 小节结构：标题 + 三段正文，插在第 4 个位置
        jobs.append((idx + 4, prefix, img_path, caption))
    jobs.sort(reverse=True)

    for index, prefix, img_path, caption in jobs:
        payload = {"index": index,
                   "children": [{"block_type": 27, "image": {}},
                                {"block_type": 2,
                                 "text": {"elements": [{"text_run": {
                                     "content": caption,
                                     "text_element_style": {"italic": False}}}],
                                     "style": {}}}]}
        r = requests.post(f"{D}/open-apis/docx/v1/documents/{DOC}/blocks/{DOC}/children",
                          headers=H, json=payload, timeout=30).json()
        if r.get("code") != 0:
            print("插入失败", prefix, r)
            continue
        img_block = r["data"]["children"][0]["block_id"]

        with open(img_path, "rb") as f:
            up = requests.post(f"{D}/open-apis/drive/v1/medias/upload_all",
                               headers=H,
                               data={"file_name": os.path.basename(img_path),
                                     "parent_type": "docx_image",
                                     "parent_node": img_block,
                                     "size": str(os.path.getsize(img_path))},
                               files={"file": open(img_path, "rb")},
                               timeout=120).json()
        if up.get("code") != 0:
            print("上传失败", prefix, up)
            continue
        ftoken = up["data"]["file_token"]
        pa = requests.patch(f"{D}/open-apis/docx/v1/documents/{DOC}/blocks/{img_block}",
                            headers=H, json={"replace_image": {"token": ftoken}},
                            timeout=30).json()
        if pa.get("code") != 0:
            print("设置图片失败", prefix, pa)
            continue
        print(f"OK  index={index}  {prefix}  block={img_block}  token={ftoken[:12]}…")

    print("done")


if __name__ == "__main__":
    main()
