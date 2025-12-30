import os
import json
import math
import numpy as np
import torch
from recbole.data.interaction import Interaction
from recbole.quick_start import load_data_and_model


def map_token_to_iid(token: str, token2id: dict):
    if token in token2id:
        return token2id[token]
    if token.startswith("item_") and token[5:] in token2id:
        return token2id[token[5:]]
    # 兼容 "123" / "item_123"
    if (not token.startswith("item_")) and (f"item_{token}" in token2id):
        return token2id[f"item_{token}"]
    return None


def _infer_n_items(model, dataset):
    # 尽量兼容不同模型：有的模型有 item_embedding，有的有 n_items
    if hasattr(model, "n_items"):
        return int(getattr(model, "n_items"))
    if hasattr(model, "item_embedding") and hasattr(model.item_embedding, "num_embeddings"):
        return int(model.item_embedding.num_embeddings)
    if hasattr(dataset, "item_num"):
        return int(dataset.item_num)
    raise AttributeError("无法推断 n_items：模型无 n_items/item_embedding，dataset 也无 item_num。")


@torch.no_grad()
def precompute_scores_memmap_with_metrics(
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

    # 先创建 out_dir，后面要写 token2id.json / meta.json / memmap 文件
    os.makedirs(out_dir, exist_ok=True)

    iid_field = dataset.iid_field
    token2id = dataset.field2token_id[iid_field]

    # ===== NEW: 保存 token2id 映射（外部 token -> 内部 iid）=====
    token2id_path = os.path.join(out_dir, "item_token2id.json")

    token2id_json = {str(k): int(v) for k, v in token2id.items()}
    token2id_json_aug = dict(token2id_json)

    # 如果 key 是纯数字字符串（如 "3431"），补一个 "item_3431" -> same id，兼容你的候选 token 格式
    for k, v in token2id_json.items():
        if k.isdigit():
            token2id_json_aug[f"item_{k}"] = int(v)

    with open(token2id_path, "w", encoding="utf-8") as f:
        json.dump(token2id_json_aug, f, ensure_ascii=False)

    print("saved:", token2id_path)
    # ===== NEW END =====

    n_items = _infer_n_items(model, dataset)

    # 顺序模型常用的字段配置（RecBole 会用 LIST_SUFFIX / ITEM_LIST_LENGTH_FIELD 等生成序列字段）
    max_len = getattr(dataset, "max_item_list_len", config["MAX_ITEM_LIST_LENGTH"])
    item_list_field = getattr(dataset, "item_id_list_field", dataset.iid_field + config["LIST_SUFFIX"])
    item_len_field = getattr(dataset, "item_list_length_field", config["ITEM_LIST_LENGTH_FIELD"])

    # 2) load user sequences
    with open(inter_json_path, "r", encoding="utf-8") as f:
        user2seq = json.load(f)

    user_ids = sorted(user2seq.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    n_users = len(user_ids)

    # 3) prepare outputs
    score_path = os.path.join(out_dir, "scores.f16")
    mmap = np.memmap(score_path, dtype=np.float16, mode="w+", shape=(n_users, n_items))

    user_index = {uid: idx for idx, uid in enumerate(user_ids)}
    with open(os.path.join(out_dir, "user_index.json"), "w", encoding="utf-8") as f:
        json.dump(user_index, f, ensure_ascii=False)

    topk_list = tuple(sorted(set(int(k) for k in topk_list)))
    k_max = int(max(topk_list))

    meta = {
        "model_path": model_path,
        "inter_json_path": inter_json_path,
        "n_users": n_users,
        "n_items": n_items,
        "dtype": "float16",
        "max_len": int(max_len),
        "last_n": int(last_n),
        "batch_size": int(batch_size),
        "mask_seen": bool(mask_seen),
        "use_history_exclude_last": bool(use_history_exclude_last),
        "topk_list": list(topk_list),
        "save_topk_items": bool(save_topk_items),
        "note": "target item is last element of each user sequence (within last_n) when use_history_exclude_last=True",
        "item_token2id_file": "item_token2id.json",
        "iid_field": str(iid_field),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # per-user metrics buffers
    per_user = {
        "target_iid": np.full((n_users,), -1, dtype=np.int32),
        "target_rank": np.full((n_users,), 0, dtype=np.int32),  # 1..Kmax, 0 means not hit
    }
    for k in topk_list:
        per_user[f"Hit@{k}"] = np.zeros((n_users,), dtype=np.float32)
        per_user[f"NDCG@{k}"] = np.zeros((n_users,), dtype=np.float32)
        per_user[f"MRR@{k}"] = np.zeros((n_users,), dtype=np.float32)
        per_user[f"Recall@{k}"] = np.zeros((n_users,), dtype=np.float32)
        per_user[f"Precision@{k}"] = np.zeros((n_users,), dtype=np.float32)

    # 可选：保存 topK 推荐 item（内部 item id）
    topk_items_path = os.path.join(out_dir, "topk_items.int32") if save_topk_items else None
    topk_items_mmap = None
    if save_topk_items:
        topk_items_mmap = np.memmap(topk_items_path, dtype=np.int32, mode="w+", shape=(n_users, k_max))

    # 4) sequence builder: history + target
    def build_hist_and_target(uid: str):
        seq = user2seq[uid][-last_n:]

        target_iid = None
        if use_history_exclude_last and len(seq) > 0:
            # target = last
            t_raw = seq[-1]
            t_tok = str(t_raw)
            t_tok = t_tok if t_tok.startswith("item_") else f"item_{t_tok}"
            mapped_t = map_token_to_iid(t_tok, token2id)
            if mapped_t is not None:
                mapped_t = int(mapped_t)
                if 0 <= mapped_t < n_items:
                    target_iid = mapped_t

            seq = seq[:-1]  # history

        # map history
        hist = []
        for x in seq:
            s = str(x)
            tok = s if s.startswith("item_") else f"item_{s}"
            mapped = map_token_to_iid(tok, token2id)
            if mapped is None:
                continue
            mapped = int(mapped)
            if 0 <= mapped < n_items:
                hist.append(mapped)

        hist = hist[-max_len:]
        return hist, target_iid

    # 5) batch inference
    n_batches = math.ceil(n_users / batch_size)
    for b in range(n_batches):
        lo = b * batch_size
        hi = min((b + 1) * batch_size, n_users)
        batch_uids = user_ids[lo:hi]

        hist_list = []
        target_list = []
        for uid in batch_uids:
            h, t = build_hist_and_target(uid)
            hist_list.append(h)
            target_list.append(t)

        lens = torch.tensor([len(s) for s in hist_list], dtype=torch.long, device=device)

        item_seq = []
        for s in hist_list:
            pad_len = max_len - len(s)
            item_seq.append([0] * pad_len + s)
        item_seq = torch.tensor(item_seq, dtype=torch.long, device=device)

        inter = Interaction({item_list_field: item_seq, item_len_field: lens}).to(device)

        scores = model.full_sort_predict(inter)

        # RecBole 中 full_sort_predict 在部分实现里会返回展平的 [B*n_items]，这里做一次兼容 reshape
        if scores.dim() == 1:
            scores = scores.view(-1, n_items)

        # 不推荐 padding=0（如 0 是 padding id）
        scores[:, 0] = -1e9

        # mask seen
        if mask_seen:
            for i, s in enumerate(hist_list):
                seen = sorted(set([x for x in s if 0 < x < n_items]))
                if seen:
                    idx = torch.tensor(seen, dtype=torch.long, device=device)
                    scores[i].index_fill_(0, idx, -1e9)

        # 写入 memmap（float16）
        mmap[lo:hi, :] = scores.to(torch.float16).detach().cpu().numpy()

        # topK
        _, topk_items = torch.topk(scores, k_max, dim=1)  # (B, Kmax)

        if save_topk_items:
            topk_items_mmap[lo:hi, :] = topk_items.detach().cpu().to(torch.int32).numpy()

        # metrics (single target)
        for i in range(hi - lo):
            gidx = lo + i
            t = target_list[i]
            if t is None:
                continue

            per_user["target_iid"][gidx] = int(t)

            items_i = topk_items[i]
            hit_vec = (items_i == int(t))
            if bool(hit_vec.any().item()):
                rank = int(torch.nonzero(hit_vec, as_tuple=False)[0].item()) + 1  # 1..Kmax
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

    # 6) save metrics files
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

    print("saved:", score_path)
    print("saved:", metrics_csv)
    print("saved:", os.path.join(out_dir, "metrics_summary.json"))
    if save_topk_items:
        print("saved:", topk_items_path)


if __name__ == "__main__":
    precompute_scores_memmap_with_metrics(
        model_path="/home/liujuntao/RecBole-master/saved_50/STAMP-Dec-25-2025_20-33-48.pth",
        inter_json_path="/home/liujuntao/Agent4Rec/data/ml-1m/ml-1m_user_inter50_all.json",
        out_dir="/home/liujuntao/Agent4Rec/data/ml-1m/stamp_scores_memmap",
        last_n=50,
        batch_size=64,
        mask_seen=True,
        use_history_exclude_last=True,
        topk_list=(5, 10, 20, 50),
        save_topk_items=True,
    )
