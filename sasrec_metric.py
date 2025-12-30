import os
import json
import math
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_item_token(x: Any) -> Optional[str]:
    """
    支持：
    - "item_880" / "880"
    - 880
    - {"id": "item_880", ...}
    返回规范化的 token: "item_xxx"；无法解析返回 None
    """
    if x is None:
        return None

    # dict 元素：优先取 id 字段
    if isinstance(x, dict):
        if "id" in x and x["id"] is not None:
            x = x["id"]
        else:
            return None

    s = str(x).strip()
    if not s:
        return None

    return s if s.startswith("item_") else f"item_{s}"


def _map_item_tok_to_iid(item_tok: str, item_token2id: Dict[str, int]) -> Optional[int]:
    s = str(item_tok)
    if s in item_token2id:
        return int(item_token2id[s])

    if s.startswith("item_") and s[5:] in item_token2id:
        return int(item_token2id[s[5:]])

    if (not s.startswith("item_")) and (f"item_{s}" in item_token2id):
        return int(item_token2id[f"item_{s}"])

    return None


def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in xs:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _save_summary_new_file(
    out_dir: str,
    summary: dict,
    prefix: str = "rerank_eval",
    extra_tag: Optional[str] = None,
) -> str:
    """
    每次新建一个文件保存 summary，避免覆盖。
    文件名：{prefix}_{timestamp}_{tag}.json
    timestamp 用 strftime 生成，包含微秒 %f，基本可避免冲突。[web:45]
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    tag = ""
    if extra_tag:
        # 简单清洗，避免路径字符
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in extra_tag)[:80]
        tag = f"_{safe}"

    file_path = out_path / f"{prefix}_{ts}{tag}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)  # 写入 JSON 文件 [web:70][web:68]
    return str(file_path)


def rerank_candidates_and_eval(
    mem_dir: str,
    candidate_json_path: str,
    ground_truth_json_path: str,
    topk_list: Tuple[int, ...] = (5, 10),
    drop_unknown_items: bool = True,
    dedup_candidates: bool = True,
    save_reranked_json: bool = False,
    reranked_out_path: Optional[str] = None,
    save_summary: bool = True,
    summary_out_dir: Optional[str] = None,
):
    """
    mem_dir: 你 precompute_scores_memmap_with_metrics 生成的目录，需包含：
      - meta.json
      - user_index.json (raw uid -> row)
      - item_token2id.json (token -> internal iid)
      - scores.f16 (shape=[n_users, n_items])

    candidate_json_path: uid -> [candidate items...]
      - candidate item 可以是 "item_123" / "123" / 123 / {"id":"item_123", ...}

    ground_truth_json_path: uid -> [hist..., target]  (默认最后一个为 target)
    """
    meta = _load_json(os.path.join(mem_dir, "meta.json"))
    user_index = _load_json(os.path.join(mem_dir, "user_index.json"))
    item_token2id = _load_json(os.path.join(mem_dir, "item_token2id.json"))

    n_users = int(meta["n_users"])
    n_items = int(meta["n_items"])
    dtype_str = str(meta.get("dtype", "float16")).lower()
    dtype = np.float16 if dtype_str in ("float16", "f16") else np.float32

    scores_path = os.path.join(mem_dir, "scores.f16")
    scores = np.memmap(scores_path, dtype=dtype, mode="r", shape=(n_users, n_items))

    cand = _load_json(candidate_json_path)
    gt = _load_json(ground_truth_json_path)

    topk_list = tuple(sorted(set(int(k) for k in topk_list)))
    k_max = int(max(topk_list))

    sums = {f"Hit@{k}": 0.0 for k in topk_list}
    sums.update({f"Recall@{k}": 0.0 for k in topk_list})
    sums.update({f"MRR@{k}": 0.0 for k in topk_list})
    sums.update({f"NDCG@{k}": 0.0 for k in topk_list})

    n_eval = 0
    reranked_dump = {}

    for uid_raw, cand_list in cand.items():
        uid_raw = str(uid_raw)

        if uid_raw not in user_index:
            continue
        if uid_raw not in gt or not gt[uid_raw]:
            continue

        row = int(user_index[uid_raw])

        # target
        target_tok = _extract_item_token(gt[uid_raw][-1])
        if target_tok is None:
            continue
        target_iid = _map_item_tok_to_iid(target_tok, item_token2id)
        if target_iid is None:
            continue

        # candidates: 支持 dict/list/str/int
        raw_list = cand_list or []
        cands_tok = []
        for x in raw_list:
            t = _extract_item_token(x)
            if t is not None:
                cands_tok.append(t)

        if dedup_candidates:
            cands_tok = _dedup_keep_order(cands_tok)

        # token -> iid & score
        cands_iid = []
        cands_score = []
        for t in cands_tok:
            iid = _map_item_tok_to_iid(t, item_token2id)
            if iid is None or not (0 <= int(iid) < n_items):
                if drop_unknown_items:
                    continue
                iid = -1
                sc = -1e9
            else:
                sc = float(scores[row, int(iid)])

            cands_iid.append(iid)
            cands_score.append(sc)

        if not cands_iid:
            continue

        # rerank by score desc
        order = np.argsort(-np.asarray(cands_score, dtype=np.float32))
        rerank_iid = [int(cands_iid[i]) for i in order.tolist() if int(cands_iid[i]) >= 0]
        rerank_tok = [cands_tok[i] for i in order.tolist() if int(cands_iid[i]) >= 0]

        rerank_iid_top = rerank_iid[:k_max]

        # target rank (1-based)
        try:
            rank = rerank_iid_top.index(int(target_iid)) + 1
        except ValueError:
            rank = 0

        n_eval += 1
        if save_reranked_json:
            reranked_dump[uid_raw] = rerank_tok[:k_max]

        for k in topk_list:
            if rank != 0 and rank <= k:
                sums[f"Hit@{k}"] += 1.0
                sums[f"Recall@{k}"] += 1.0
                sums[f"MRR@{k}"] += 1.0 / float(rank)
                sums[f"NDCG@{k}"] += 1.0 / float(math.log2(rank + 1.0))

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mem_dir": mem_dir,
        "candidate_json_path": candidate_json_path,
        "ground_truth_json_path": ground_truth_json_path,
        "topk_list": list(topk_list),
        "k_max": k_max,
        "drop_unknown_items": drop_unknown_items,
        "dedup_candidates": dedup_candidates,
        "n_eval": n_eval,
    }

    if n_eval == 0:
        summary["error"] = "No evaluable users (uid mismatch / no target / empty candidates)."
    else:
        for k in topk_list:
            summary[f"Hit@{k}"] = sums[f"Hit@{k}"] / n_eval
            summary[f"Recall@{k}"] = sums[f"Recall@{k}"] / n_eval
            summary[f"MRR@{k}"] = sums[f"MRR@{k}"] / n_eval
            summary[f"NDCG@{k}"] = sums[f"NDCG@{k}"] / n_eval

    if save_reranked_json:
        if reranked_out_path is None:
            reranked_out_path = os.path.join(mem_dir, "reranked_candidates_topk.json")
        with open(reranked_out_path, "w", encoding="utf-8") as f:
            json.dump(reranked_dump, f, ensure_ascii=False, indent=2)  # 写入 JSON 文件 [web:70][web:68]
        summary["reranked_out_path"] = reranked_out_path

    # 每次新建 summary 文件
    if save_summary:
        if summary_out_dir is None:
            summary_out_dir = os.path.join(mem_dir, "eval_logs")

        # 给文件名加个稳定 tag（候选文件名 + 简短hash），方便区分不同实验
        base = os.path.basename(candidate_json_path)
        h = hashlib.md5(candidate_json_path.encode("utf-8")).hexdigest()[:8]
        tag = f"{base}_{h}"

        summary_path = _save_summary_new_file(
            out_dir=summary_out_dir,
            summary=summary,
            prefix="rerank_eval",
            extra_tag=tag,
        )
        summary["summary_path"] = summary_path

    return summary

import os
import re
import json
from datetime import datetime

def _infer_cand_len_from_name(path: str) -> int:
    # e.g. test_topk_prediction_user2000_random_100.json -> 100
    m = re.search(r"random_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else -1

def batch_rerank_eval(
    mem_dirs: dict,                # {model_name: mem_dir}
    candidate_paths: list,          # [cand_json_10, cand_json_50, cand_json_100]
    ground_truth_json_path: str,
    topk_list=(3, 5, 10),
    drop_unknown_items=True,
    dedup_candidates=True,
    save_reranked_json=False,
    save_summary=True,
    summary_root_dir=None,          # e.g. /home/.../rerank_logs
    out_csv_path=None,              # e.g. /home/.../rerank_results.csv
    out_md_path=None,               # e.g. /home/.../rerank_results.md
):
    rows = []

    if summary_root_dir is None:
        summary_root_dir = os.path.join(os.getcwd(), "rerank_logs")

    for model_name, mem_dir in mem_dirs.items():
        for cand_path in candidate_paths:
            cand_len = _infer_cand_len_from_name(cand_path)

            # 每个模型单独放日志，方便查
            model_log_dir = os.path.join(summary_root_dir, model_name)

            s = rerank_candidates_and_eval(
                mem_dir=mem_dir,
                candidate_json_path=cand_path,
                ground_truth_json_path=ground_truth_json_path,
                topk_list=topk_list,
                drop_unknown_items=drop_unknown_items,
                dedup_candidates=dedup_candidates,
                save_reranked_json=save_reranked_json,
                reranked_out_path=None,
                save_summary=save_summary,
                summary_out_dir=model_log_dir,
            )

            row = {
                "time": s.get("time"),
                "model": model_name,
                "cand_len_tag": cand_len,
                "mem_dir": mem_dir,
                "candidate_json": cand_path,
                "n_eval": s.get("n_eval", 0),
            }

            # 展开指标列
            for k in s.get("topk_list", list(topk_list)):
                row[f"Hit@{k}"] = s.get(f"Hit@{k}")
                row[f"Recall@{k}"] = s.get(f"Recall@{k}")
                row[f"MRR@{k}"] = s.get(f"MRR@{k}")
                row[f"NDCG@{k}"] = s.get(f"NDCG@{k}")

            row["summary_path"] = s.get("summary_path", "")
            row["error"] = s.get("error", "")
            rows.append(row)

    # ---------- 输出 CSV / Markdown ----------
    if out_csv_path is None:
        out_csv_path = os.path.join(summary_root_dir, "rerank_batch_results.csv")
    if out_md_path is None:
        out_md_path = os.path.join(summary_root_dir, "rerank_batch_results.md")

    # 优先用 pandas（表格更舒服）；没有 pandas 就退化成写 jsonl/csv
    try:
        import pandas as pd

        df = pd.DataFrame(rows)

        # 排序：模型 -> 候选长度
        if "cand_len_tag" in df.columns:
            df = df.sort_values(["model", "cand_len_tag"], ascending=[True, True])

        # 保存 CSV（index=False 可避免多写一列行号）[web:1]
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        df.to_csv(out_csv_path, index=False, encoding="utf-8")

        # 保存 Markdown（to_markdown 依赖 tabulate）[web:12]
        md = df.to_markdown(index=False)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(md + "\n")

        return {"n_runs": len(rows), "out_csv_path": out_csv_path, "out_md_path": out_md_path}

    except ImportError:
        # 无 pandas：写一个 JSON Lines 也能做记录
        out_jsonl_path = os.path.join(summary_root_dir, "rerank_batch_results.jsonl")
        os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
        with open(out_jsonl_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return {"n_runs": len(rows), "out_jsonl_path": out_jsonl_path}

if __name__ == "__main__":
    mem_dirs = {
        "glintru":  "/home/liujuntao/Agent4Rec/data/ml-1m/glintru_scores_memmap",
        "gru4rec":  "/home/liujuntao/Agent4Rec/data/ml-1m/gru4rec_scores_memmap",
        "kgat":     "/home/liujuntao/Agent4Rec/data/ml-1m/kgat_scores_memmap",
        "lightgcn": "/home/liujuntao/Agent4Rec/data/ml-1m/lightgcn_scores_memmap",
        "sasrec":   "/home/liujuntao/Agent4Rec/data/ml-1m/sasrec_scores_memmap",
        "stamp":    "/home/liujuntao/Agent4Rec/data/ml-1m/stamp_scores_memmap",
    }

    candidate_paths = [
        "/home/liujuntao/test_topk_prediction_user2000_random_10.json",
        "/home/liujuntao/test_topk_prediction_user2000_random_50.json",
        "/home/liujuntao/test_topk_prediction_user2000_random_100.json",
    ]

    ground_truth_json_path = "/home/liujuntao/ml-1m_user2000_inter50_.json"

    res = batch_rerank_eval(
        mem_dirs=mem_dirs,
        candidate_paths=candidate_paths,
        ground_truth_json_path=ground_truth_json_path,
        topk_list=(3, 5, 10),
        drop_unknown_items=True,
        dedup_candidates=True,
        save_reranked_json=False,  # 如果也想存每次重排后的 topk，再打开
        save_summary=True,
        summary_root_dir="/home/liujuntao/Agent4Rec/exp_results/rerank_logs",
        out_csv_path="/home/liujuntao/Agent4Rec/exp_results/rerank_logs/rerank_batch_results.csv",
        out_md_path="/home/liujuntao/Agent4Rec/exp_results/rerank_logs/rerank_batch_results.md",
    )
    print(res)
