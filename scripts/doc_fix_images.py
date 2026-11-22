"""修正配图位置：飞书的 index 是"子块下标"，不含文档标题块本身，
所以之前整体错位了一格，图落在了下一个小节标题后面。这个脚本先删掉插错的
图和图注，再按正确下标重插。

用法: python3 scripts/doc_fix_images.py
"""
import os
import pathlib

import requests

DOC = "KHp0dmJlcoY9ATxSqfhcyDR4nLb"
D = "https://open.feishu.cn"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CAPTIONS = ("左：8 局一点", "叶子吃子修正 + 240", "加宽这一轮", "左：一份损失", "左：只改模拟次数")

PLAN = [
    ("评测换了一把尺子", "reports/chart_yardstick.png", "左：8 局一点的旧尺子，阴影是 95% 区间；右：40 局一点的新尺子"),
    ("叶子上补一眼吃子", "reports/chart_q1.png", "叶子吃子修正 + 240 次模拟这一轮"),
    ("策略主干从 256 加宽到 512", "reports/chart_wide.png", "加宽这一轮，五个评测点"),
    ("把策略损失拆成三块", "reports/chart_loss.png", "左：一份损失的三块构成；右：同一局面两次搜索的差距"),
    ("把搜索深度扫一遍", "reports/chart_depth.png", "左：只改模拟次数；右：棋力来自哪里"),
]


def env():
    e = {}
    for line in (pathlib.Path.home() / ".hermes/.env").read_text(errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            e[k.strip()] = v.strip().strip('"').strip("'")
    return e


def main():
    e = env()
    tok = requests.post(f"{D}/open-apis/auth/v3/tenant_access_token/internal",
                        json={"app_id": e["FEISHU_APP_ID"], "app_secret": e["FEISHU_APP_SECRET"]},
                        timeout=20).json()["tenant_access_token"]
    H = {"Authorization": f"Bearer {tok}"}
    base = f"{D}/open-apis/docx/v1/documents/{DOC}"

    def blocks():
        return requests.get(f"{base}/blocks", headers=H, params={"page_size": 500},
                            timeout=30).json()["data"]["items"]

    def text_of(b):
        for k in ("text", "heading3"):
            if k in b:
                return "".join(x.get("text_run", {}).get("content", "") for x in b[k].get("elements", []))
        return ""

    items = blocks()
    print("起始 blocks:", len(items))

    # 1) 删掉插错的图与图注（每对是连续两块，从后往前删）
    bad = []
    for i, b in enumerate(items):
        if b.get("block_type") == 27:
            cap = text_of(items[i + 1]) if i + 1 < len(items) else ""
            if cap.startswith(CAPTIONS):
                bad.append(i)
    print("待删图片块:", bad)
    for i in sorted(bad, reverse=True):
        # 子块下标 = 列表下标 - 1（第 0 个是文档标题本身，不算子块）
        start = i - 1
        r = requests.delete(f"{base}/blocks/{DOC}/children/batch_delete", headers=H,
                            json={"start_index": start, "end_index": start + 2},
                            timeout=30).json()
        print("删除", i, "->", r.get("code"), r.get("msg"))

    items = blocks()
    print("删除后 blocks:", len(items))

    # 2) 按正确下标重插：标题位置 + 3 = 第三段之后
    jobs = []
    for prefix, img, caption in PLAN:
        path = os.path.join(ROOT, img)
        hit = [i for i, b in enumerate(items) if text_of(b).startswith(prefix)]
        if not hit or not os.path.exists(path):
            print("跳过", prefix)
            continue
        jobs.append((hit[0] + 3, prefix, path, caption))
    jobs.sort(reverse=True)

    for index, prefix, path, caption in jobs:
        payload = {"index": index,
                   "children": [{"block_type": 27, "image": {}},
                                {"block_type": 2, "text": {"elements": [
                                    {"text_run": {"content": caption,
                                                  "text_element_style": {"italic": False}}}]}}]}
        r = requests.post(f"{base}/blocks/{DOC}/children", headers=H, json=payload,
                          timeout=30).json()
        if r.get("code") != 0:
            print("插入失败", prefix, r)
            continue
        blk = r["data"]["children"][0]["block_id"]
        with open(path, "rb") as f:
            up = requests.post(f"{D}/open-apis/drive/v1/medias/upload_all", headers=H,
                               data={"file_name": os.path.basename(path),
                                     "parent_type": "docx_image", "parent_node": blk,
                                     "size": str(os.path.getsize(path))},
                               files={"file": f}, timeout=120).json()
        if up.get("code") != 0:
            print("上传失败", prefix, up)
            continue
        pa = requests.patch(f"{base}/blocks/{blk}", headers=H,
                            json={"replace_image": {"token": up["data"]["file_token"]}},
                            timeout=30).json()
        print(("OK  " if pa.get("code") == 0 else "设置失败 ") + f"index={index} {prefix}")

    # 3) 校验：每个小节标题和下一个标题之间应该有 5 块
    items = blocks()
    print("最终 blocks:", len(items))
    heads = [i for i, b in enumerate(items) if b.get("block_type") == 5]
    for a, b in zip(heads, heads[1:]):
        kinds = [items[j].get("block_type") for j in range(a + 1, b)]
        if 27 in kinds:
            print(f"  {text_of(items[a])[:16]:18s} 段内块类型 {kinds}")


if __name__ == "__main__":
    main()
