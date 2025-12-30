# -*- coding: utf-8 -*-
import json
from pathlib import Path
INPUT_FILES = [
"/home/liujuntao/test_topk_prediction_user2000_random_10.json",
"/home/liujuntao/test_topk_prediction_user2000_random_50.json",
"/home/liujuntao/test_topk_prediction_user2000_random_100.json",
"/home/liujuntao/ml-1m_user2000_inter50_.json"
]

# 输出：为空表示覆盖原文件；不为空表示输出到该目录（文件名不变）
OUTPUT_DIR = ""  # e.g. "/home/liujuntao/cleaned"

# 要删除的 uid（注意：你的示例 key 是字符串，所以这里用字符串匹配）
UIDS_TO_REMOVE = ["2271", "3230","4029","3686","3382","3953","5978"]
# ============================


def iter_input_files():
    if INPUT_FILES:
        for p in INPUT_FILES:
            yield Path(p)

def process_one_file(in_path: Path):
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)  # 读 JSON 为 Python 对象 [web:26]

    if not isinstance(data, dict):
        print(f"[SKIP] {in_path} top-level 不是 dict")
        return

    removed = 0
    for uid in UIDS_TO_REMOVE:
        if uid in data:
            data.pop(uid)  # 删除该 uid 的整条数据 [web:3]
            removed += 1

    out_path = in_path
    if OUTPUT_DIR:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / in_path.name

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)  # 保留中文并格式化 [web:26]
        f.write("\n")
    tmp_path.replace(out_path)

    print(f"[OK] {in_path} -> {out_path}, removed={removed}")


def main():
    files = list(iter_input_files())
    if not files:
        raise SystemExit("没有找到任何输入 JSON 文件，请检查 INPUT_FILES / INPUT_DIR / PATTERN。")

    for fp in files:
        process_one_file(fp)


if __name__ == "__main__":
    main()
