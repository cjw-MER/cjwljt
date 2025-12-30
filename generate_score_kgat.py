import os
import json
import math
import numpy as np
import torch
from recbole.data.interaction import Interaction
from recbole.quick_start import load_data_and_model


def map_token_to_iid(token: str, token2id: dict):
    """把外部 item token 映射成 RecBole 内部 item id（0 ~ n_items-1）"""
    if token in token2id:
        return token2id[token]
    if token.startswith("item_") and token[5:] in token2id:
        return token2id[token[5:]]
    if (not token.startswith("item_")) and (f"item_{token}" in token2id):
        return token2id[f"item_{token}"]
    return None


def map_token_to_uid(token: str, token2id: dict):
    """把外部 user token 映射成 RecBole 内部 user id（0 ~ n_users-1）"""
    if token in token2id:
        return token2id[token]
    if token.startswith("user_") and token[5:] in token2id:
        return token2id[token[5:]]
    if (not token.startswith("user_")) and (f"user_{token}" in token2id):
        return token2id[f"user_{token}"]
    return None


def _save_token2id_json(token2id: dict, out_path: str, prefix: str):
    """
    保存 token2id 映射到 json，并补齐带 prefix 的 key（例如 item_123 / user_123）。
    prefix: "item_" or "user_"
    """
    token2id_json = {str(k): int(v) for k, v in token2id.items()}
    token2id_json_aug = dict(token2id_json)

    for k, v in token2id_json.items():
        # 若 key 是纯数字字符串，则补一个带前缀的版本
        if k.isdigit():
            token2id_json_aug[f"{prefix}{k}"] = int(v)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(token2id_json_aug, f, ensure_ascii=False)

    print("saved:", out_path)


@torch.no_grad()
def precompute_scores_memmap_and_metrics(
    model_path: str,
    inter_json_path: str,
    out_dir: str,
    last_n: int = 50,
    batch_size: int = 64,
    mask_seen: bool = True,
    use_history_exclude_last: bool = True,  # True: 最后一个作为 target，不参与输入历史
    topk_list=(5, 10, 20, 50),
    save_topk_items: bool = True,
):
    # 1) load model/dataset
    config, model, dataset, *_ = load_data_and_model(model_file=model_path)
    model.eval()
    device = torch.device(config["device"])
    model = model.to(device)

    # 输出目录
    os.makedirs(out_dir, exist_ok=True)

    # KGAT / LightGCN 等：物品数量用 model.n_items（你原注释这么写的）
    n_items = int(model.n_items)

    # 2) token<->id mapping
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    u_token2id = dataset.field2token_id[uid_field]
    i_token2id = dataset.field2token_id[iid_field]

    # ===== NEW: 保存 user/item token2id 映射 =====
    _save_token2id_json(u_token2id, os.path.join(out_dir, "user_token2id.json"), prefix="user_")
    _save_token2id_json(i_token2id, os.path.join(out_dir, "item_token2id.json"), prefix="item_")
    # ===== NEW END =====

    # 3) load user sequences
    with open(inter_json_path, "r", encoding="utf-8") as f:
        user2seq = json.load(f)

    user_ids = sorted(user2seq.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    n_users = len(user_ids)

    # 4) prepare outputs
    # scores memmap
    score_path = os.path.join(out_dir, "scores.f16")
    mmap = np.memmap(score_path, dtype=np.float16, mode="w+", shape=(n_users, n_items))

    # user index: raw uid -> row idx
    user_index = {uid: idx for idx, uid in enumerate(user_ids)}
    with open(os.path.join(out_dir, "user_index.json"), "w", encoding="utf-8") as f:
        json.dump(user_index, f, ensure_ascii=False)

    # meta
    topk_list = tuple(sorted(set(int(k) for k in topk_list)))
    k_max = int(max(topk_list))
    meta = {
        "model_path": model_path,
        "inter_json_path": inter_json_path,
        "n_users": n_users,
        "n_items": n_items,
        "dtype": "float16",
        "last_n": int(last_n),
        "batch_size": int(batch_size),
        "mask_seen": bool(mask_seen),
        "use_history_exclude_last": bool(use_history_exclude_last),
        "topk_list": list(topk_list),
        "save_topk_items": bool(save_topk_items),
        "note": "target item is last element of each user sequence (within last_n) when use_history_exclude_last=True",
        "uid_field": str(uid_field),
        "iid_field": str(iid_field),
        "user_token2id_file": "user_token2id.json",
        "item_token2id_file": "item_token2id.json",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # metrics buffers (per-user)
    per_user = {
        "target_iid": np.full((n_users,), -1, dtype=np.int32),
        "target_rank": np.full((n_users,), 0, dtype=np.int32),
    }
    for k in topk_list:
        per_user[f"Hit@{k}"] = np.zeros((n_users,), dtype=np.float32)
        per_user[f"NDCG@{k}"] = np.zeros((n_users,), dtype=np.float32)
        per_user[f"MRR@{k}"] = np.zeros((n_users,), dtype=np.float32)
        per_user[f"Recall@{k}"] = np.zeros((n_users,), dtype=np.float32)
        per_user[f"Precision@{k}"] = np.zeros((n_users,), dtype=np.float32)

    # 可选：保存每个用户的 topK 推荐物品（内部 item id）
    topk_items_path = os.path.join(out_dir, "topk_items.int32") if save_topk_items else None
    topk_items_mmap = None
    if save_topk_items:
        topk_items_mmap = np.memmap(topk_items_path, dtype=np.int32, mode="w+", shape=(n_users, k_max))

    def build_history_and_target(uid_raw: str):
        seq = user2seq.get(uid_raw, [])
        seq = seq[-last_n:]

        if use_history_exclude_last:
            if len(seq) == 0:
                return [], None
            target_raw = seq[-1]
            hist = seq[:-1]
        else:
            hist = seq
            target_raw = None

        hist_iids = []
        for x in hist:
            s = str(x)
            tok = s if s.startswith("item_") else f"item_{s}"
            mapped = map_token_to_iid(tok, i_token2id)
            if mapped is None:
                continue
            mapped = int(mapped)
            if 0 <= mapped < n_items:
                hist_iids.append(mapped)

        target_iid = None
        if target_raw is not None:
            s = str(target_raw)
            tok = s if s.startswith("item_") else f"item_{s}"
            mapped = map_token_to_iid(tok, i_token2id)
            if mapped is not None:
                mapped = int(mapped)
                if 0 <= mapped < n_items:
                    target_iid = mapped

        return hist_iids, target_iid

    # 5) batch inference
    n_batches = math.ceil(n_users / batch_size)
    for b in range(n_batches):
        lo = b * batch_size
        hi = min((b + 1) * batch_size, n_users)
        batch_uids_raw = user_ids[lo:hi]

        # map raw user token -> internal user id
        batch_uid_internal = []
        batch_hist = []
        batch_target = []
        for uid_raw in batch_uids_raw:
            tok = str(uid_raw)
            mapped_u = map_token_to_uid(tok, u_token2id)
            if mapped_u is None:
                mapped_u = map_token_to_uid(f"user_{tok}", u_token2id)
            if mapped_u is None:
                raise KeyError(f"User token not found in dataset mapping: {uid_raw}")
            batch_uid_internal.append(int(mapped_u))

            hist_iids, target_iid = build_history_and_target(uid_raw)
            batch_hist.append(hist_iids)
            batch_target.append(target_iid)

        u = torch.tensor(batch_uid_internal, dtype=torch.long, device=device)

        # KGAT full_sort_predict needs user_id field
        inter = Interaction({model.USER_ID: u}).to(device)

        # full_sort_predict returns flatten -> reshape
        scores = model.full_sort_predict(inter).view(-1, n_items)

        # mask padding id=0（如 0 是 padding id）
        scores[:, 0] = -1e9

        # mask seen
        if mask_seen:
            for i, hist_iids in enumerate(batch_hist):
                if not hist_iids:
                    continue
                seen = sorted(set([x for x in hist_iids if 0 <= x < n_items]))
                if seen:
                    idx = torch.tensor(seen, dtype=torch.long, device=device)
                    scores[i].index_fill_(0, idx, -1e9)

        # write scores memmap
        mmap[lo:hi, :] = scores.to(torch.float16).detach().cpu().numpy()

        # topk
        _, topk_items = torch.topk(scores, k_max, dim=1)

        if save_topk_items:
            topk_items_mmap[lo:hi, :] = topk_items.detach().cpu().to(torch.int32).numpy()

        # compute metrics per user (single target item)
        for i in range(hi - lo):
            gidx = lo + i
            t = batch_target[i]
            if t is None:
                continue

            per_user["target_iid"][gidx] = int(t)

            items_i = topk_items[i]
            hit_vec = (items_i == int(t))
            if bool(hit_vec.any().item()):
                rank = int(torch.nonzero(hit_vec, as_tuple=False)[0].item()) + 1
            else:
                rank = 0
            per_user["target_rank"][gidx] = rank

            for k in topk_list:
                if rank != 0 and rank <= k:
                    per_user[f"Hit@{k}"][gidx] = 1.0
                    per_user[f"Recall@{k}"][gidx] = 1.0
                    per_user[f"Precision@{k}"][gidx] = 1.0 / float(k)
                    per_user[f"MRR@{k}"][gidx] = 1.0 / float(rank)
                    per_user[f"NDCG@{k}"][gidx] = 1.0 / float(math.log2(rank + 1.0))

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            print(f"done {b+1}/{n_batches} batches")

    mmap.flush()
    if save_topk_items:
        topk_items_mmap.flush()

    # 6) save metrics
    metrics_csv = os.path.join(out_dir, "metrics_per_user.csv")
    header = (
        ["uid_raw", "target_iid", "target_rank"]
        + [f"Hit@{k}" for k in topk_list]
        + [f"NDCG@{k}" for k in topk_list]
        + [f"MRR@{k}" for k in topk_list]
        + [f"Recall@{k}" for k in topk_list]
        + [f"Precision@{k}" for k in topk_list]
    )

    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for idx, uid_raw in enumerate(user_ids):
            row = [str(uid_raw), str(int(per_user["target_iid"][idx])), str(int(per_user["target_rank"][idx]))]
            for k in topk_list:
                row.append(f"{per_user[f'Hit@{k}'][idx]:.6f}")
            for k in topk_list:
                row.append(f"{per_user[f'NDCG@{k}'][idx]:.6f}")
            for k in topk_list:
                row.append(f"{per_user[f'MRR@{k}'][idx]:.6f}")
            for k in topk_list:
                row.append(f"{per_user[f'Recall@{k}'][idx]:.6f}")
            for k in topk_list:
                row.append(f"{per_user[f'Precision@{k}'][idx]:.6f}")
            f.write(",".join(row) + "\n")

    summary = {
        "n_users": n_users,
        "n_items": n_items,
        "n_users_with_target": int(np.sum(per_user["target_iid"] >= 0)),
    }
    for k in topk_list:
        summary[f"Hit@{k}"] = float(np.mean(per_user[f"Hit@{k}"]))
        summary[f"NDCG@{k}"] = float(np.mean(per_user[f"NDCG@{k}"]))
        summary[f"MRR@{k}"] = float(np.mean(per_user[f"MRR@{k}"]))
        summary[f"Recall@{k}"] = float(np.mean(per_user[f"Recall@{k}"]))
        summary[f"Precision@{k}"] = float(np.mean(per_user[f"Precision@{k}"]))

    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("saved scores:", score_path)
    print("saved metrics:", metrics_csv)
    print("saved summary:", os.path.join(out_dir, "metrics_summary.json"))
    if save_topk_items:
        print("saved topk items:", topk_items_path)


if __name__ == "__main__":
    precompute_scores_memmap_and_metrics(
        model_path="/home/liujuntao/RecBole-master/saved_50/KGAT-Dec-26-2025_15-10-02.pth",
        inter_json_path="/home/liujuntao/Agent4Rec/data/ml-1m/ml-1m_user_inter50_all.json",
        out_dir="/home/liujuntao/Agent4Rec/data/ml-1m/kgat_scores_memmap",
        last_n=50,
        batch_size=64,
        mask_seen=True,
        use_history_exclude_last=True,
        topk_list=(5, 10, 20, 50),
        save_topk_items=True,
    )
