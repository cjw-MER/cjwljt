import json
import re
import numpy as np
from typing import List, Optional, Any, Dict
from datetime import datetime
from pathlib import Path

from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

from metric import ndcg_all, recall_all


def _map_item_token_to_inner_id(dataset, token: str) -> Optional[int]:
    iid_field = dataset.iid_field
    token2id = dataset.field2token_id[iid_field]

    s = str(token).strip()
    if s in token2id:
        return int(token2id[s])

    if s.startswith("item_"):
        pure = s[5:]
        if pure in token2id:
            return int(token2id[pure])

    return None


def _normalize_candidate_row(row, k: int = 10) -> List[str]:
    if row is None:
        tokens = []
    elif isinstance(row, list):
        tokens = []
        for x in row:
            if isinstance(x, dict) and "id" in x:
                tokens.append(str(x["id"]))
            else:
                tokens.append(str(x))
    elif isinstance(row, str):
        tokens = re.findall(r"item_\d+", row)
    else:
        tokens = [str(row)]

    if len(tokens) >= k:
        return tokens[:k]
    return tokens + ["0"] * (k - len(tokens))


def _candidates_to_topk_iids(dataset, candidate_items: List, k: int = 10) -> np.ndarray:
    topk_iids = []
    for row in candidate_items:
        tokens = _normalize_candidate_row(row, k=k)
        iids = []
        for t in tokens:
            inner = _map_item_token_to_inner_id(dataset, t)
            iids.append(0 if inner is None else inner)
        topk_iids.append(iids)

    return np.array(topk_iids, dtype=np.int64)


def _build_uid_target_from_test(test_data) -> dict:
    data = test_data.dataset.inter_feat.interaction
    uid_target = {}

    for uid, iid_his, i_len in zip(data["user_id"], data["item_id_list"], data["item_length"]):
        u = int(uid.detach().cpu()) if hasattr(uid, "detach") else int(uid)
        L = int(i_len.detach().cpu()) if hasattr(i_len, "detach") else int(i_len)

        tgt = iid_his[L - 1]
        tgt = int(tgt.detach().cpu()) if hasattr(tgt, "detach") else int(tgt)
        uid_target[u] = tgt

    return uid_target


def _save_metrics_new_file(out_dir: str, record: Dict[str, Any], prefix: str = "metrics") -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    file_path = out_path / f"{prefix}_{ts}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return str(file_path)


def metric_cal(
    candidate_items=None,
    k: int = 10,
    selected_users_path: str = "/home/liujuntao/ml-1m_user2000_inter50_.json",
    model_file: str = "/home/liujuntao/RecBole-master/saved_50/SASRec-Dec-25-2025_20-30-57.pth",
    out_dir: str = "/home/liujuntao/Agent4Rec/exp_results",
    out_prefix: str = "SASRec",
    eval_user_tokens: Optional[List[str]] = None,   # ====== NEW ======
):
    """
    candidate_items:
    - None：评测 RecBole 模型本身 full_sort_topk 的性能
    - List：评测外部候选列表性能（长度=评测用户数）

    eval_user_tokens（新增）：
    - 当 candidate_items 不是 None 时，强烈建议传入，用于保证“每一行候选”对应“哪个用户”。
    """
    print("Loading data and model...")
    print("Candidate items type:", type(candidate_items))
    if candidate_items is not None:
        print("Candidate items sample:", candidate_items[0] if len(candidate_items) > 0 else "N/A")

    with open(selected_users_path, "r", encoding="utf-8") as f:
        selected_users = json.load(f)

    all_user_tokens = list(selected_users.keys())

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_file,
    )

    uid_target = _build_uid_target_from_test(test_data)

    # ---------- case 1: 评测 RecBole 模型 ----------
    if candidate_items is None:
        uid_series = dataset.token2id(dataset.uid_field, all_user_tokens)

        topk_score, topk_iid_list = full_sort_topk(
            uid_series, model, test_data, k=k, device=config["device"]
        )

        vals = [int(uid_target[int(u)]) for u in uid_series]

        r3 = recall_all(topk_iid_list, vals, 3)
        r5 = recall_all(topk_iid_list, vals, 5)
        n3 = ndcg_all(topk_iid_list, vals, 3)
        n5 = ndcg_all(topk_iid_list, vals, 5)

        print("recall_3", r3)
        print("recall_5", r5)
        print("ndcg_3", n3)
        print("ndcg_5", n5)

        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "mode": "recbole_full_sort_topk",
            "model_file": model_file,
            "selected_users_path": selected_users_path,
            "k": int(k),
            "n_eval": int(len(uid_series)),
            "recall_3": float(r3),
            "recall_5": float(r5),
            "ndcg_3": float(n3),
            "ndcg_5": float(n5),
        }
        saved_path = _save_metrics_new_file(out_dir, record, prefix=out_prefix)
        print("Saved metrics to:", saved_path)
        return record

    # ---------- case 2: 评测外部候选列表 ----------
    n_eval = len(candidate_items)
    if eval_user_tokens is None:
        # 兼容旧行为：用 selected_users 的前 n_eval 个（注意：如果你“跳过用户”，这里会错位）
        eval_user_tokens = all_user_tokens[:n_eval]
    else:
        if len(eval_user_tokens) != n_eval:
            raise ValueError(f"len(eval_user_tokens)={len(eval_user_tokens)} != len(candidate_items)={n_eval}")

    uid_series = dataset.token2id(dataset.uid_field, eval_user_tokens)

    print(f"Evaluating {n_eval} users...")
    if len(candidate_items) > 0:
        print("Sample Data Type:", type(candidate_items[0]))
        if isinstance(candidate_items[0], list) and len(candidate_items[0]) > 0:
            print("Sample Item:", candidate_items[0][0])

    topk_iid_list = _candidates_to_topk_iids(test_data.dataset, candidate_items, k=k)
    vals = [int(uid_target[int(u)]) for u in uid_series]

    r3 = recall_all(topk_iid_list, vals, 3)
    r5 = recall_all(topk_iid_list, vals, 5)
    n3 = ndcg_all(topk_iid_list, vals, 3)
    n5 = ndcg_all(topk_iid_list, vals, 5)

    print("recall_3", r3)
    print("recall_5", r5)
    print("ndcg_3", n3)
    print("ndcg_5", n5)

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "external_candidates",
        "model_file": model_file,
        "selected_users_path": selected_users_path,
        "k": int(k),
        "n_eval": int(n_eval),
        "recall_3": float(r3),
        "recall_5": float(r5),
        "ndcg_3": float(n3),
        "ndcg_5": float(n5),
        "eval_user_tokens_head": eval_user_tokens[:10],
    }
    saved_path = _save_metrics_new_file(out_dir, record, prefix=out_prefix)
    print("Saved metrics to:", saved_path)
    return record
