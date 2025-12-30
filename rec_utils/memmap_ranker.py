from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .tokens import token_to_iid


@dataclass(frozen=True)
class MemmapPack:
    meta: Dict[str, Any]
    user_index: Dict[str, int]
    score_mmap: np.memmap
    item_token2id: Dict[str, int]
    user_token2id: Optional[Dict[str, int]]


class MemmapRanker:
    """
    从 memmap 目录加载分数矩阵 (n_users, n_items)，给定 user_id + candidate_tokens 返回重排结果。

    特性：
    - 内置 cache：同一个 mem_dir 只加载一次（meta/index/mmap）
    - 支持 user_token2id 可选映射
    """

    def __init__(self):
        self._cache: Dict[str, MemmapPack] = {}

    def _load_pack(self, mem_dir: str) -> MemmapPack:
        if mem_dir in self._cache:
            return self._cache[mem_dir]

        meta_path = os.path.join(mem_dir, "meta.json")
        user_index_path = os.path.join(mem_dir, "user_index.json")
        score_path = os.path.join(mem_dir, "scores.f16")
        item_token2id_path = os.path.join(mem_dir, "item_token2id.json")
        user_token2id_path = os.path.join(mem_dir, "user_token2id.json")

        for p in [meta_path, user_index_path, score_path, item_token2id_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required file not found: {p}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(user_index_path, "r", encoding="utf-8") as f:
            user_index = json.load(f)
        with open(item_token2id_path, "r", encoding="utf-8") as f:
            item_token2id = json.load(f)

        user_token2id = None
        if os.path.exists(user_token2id_path):
            with open(user_token2id_path, "r", encoding="utf-8") as f:
                user_token2id = json.load(f)

        n_users = int(meta["n_users"])
        n_items = int(meta["n_items"])
        dtype_str = str(meta.get("dtype", "float16")).lower()

        if dtype_str in ("float16", "f16", "float_16", "np.float16"):
            dtype = np.float16
        elif dtype_str in ("float32", "f32", "float_32", "np.float32"):
            dtype = np.float32
        else:
            dtype = np.float16

        score_mmap = np.memmap(score_path, dtype=dtype, mode="r", shape=(n_users, n_items))

        pack = MemmapPack(
            meta=meta,
            user_index=user_index,
            score_mmap=score_mmap,
            item_token2id=item_token2id,
            user_token2id=user_token2id,
        )
        self._cache[mem_dir] = pack
        return pack

    def _resolve_user_row(self, pack: MemmapPack, user_id: str) -> Optional[int]:
        uid_raw = str(user_id)

        if pack.user_token2id is None:
            return pack.user_index.get(uid_raw)

        u_tok = uid_raw if uid_raw.startswith("user_") else f"user_{uid_raw}"
        internal_uid = pack.user_token2id.get(uid_raw, pack.user_token2id.get(u_tok, None))
        if internal_uid is not None:
            row = pack.user_index.get(str(int(internal_uid)))
            if row is not None:
                return row

        return pack.user_index.get(uid_raw)

    def rank(
        self,
        mem_dir: str,
        user_id: str,
        candidate_tokens: List[str],
        drop_ratio: float = 0.0,
        drop_unknown: bool = True,
        min_keep: int = 5,
        unknown_score: float = -1e9,
    ) -> Tuple[List[str], List[float]]:
        """
        返回 (tokens_sorted, scores_sorted)。

        drop_ratio:
        - 0.0: 全保留但重排
        - 0.5: 保留约一半（仍然会至少保留 min_keep）
        """
        if not (0.0 <= float(drop_ratio) <= 1.0):
            raise ValueError(f"drop_ratio must be in [0, 1], got {drop_ratio}")

        pack = self._load_pack(mem_dir)
        row = self._resolve_user_row(pack, user_id)
        if row is None:
            return [], []

        n_items = int(pack.score_mmap.shape[1])

        tokens_all: List[str] = []
        scores_all: List[float] = []

        for t in candidate_tokens:
            tok = str(t)
            iid = token_to_iid(tok, pack.item_token2id)

            if iid is None or not (0 <= int(iid) < n_items):
                if drop_unknown:
                    continue
                tokens_all.append(tok)
                scores_all.append(float(unknown_score))
                continue

            tokens_all.append(tok)
            scores_all.append(float(pack.score_mmap[row, int(iid)]))

        if not tokens_all:
            return [], []

        scores_np = np.asarray(scores_all, dtype=np.float32)
        order = np.argsort(-scores_np)
        n = int(order.shape[0])

        keep_n = max(int(min_keep), int(math.floor(n * (1.0 - float(drop_ratio)))))
        keep_n = min(keep_n, n)
        if keep_n <= 0:
            return [], []

        order = order[:keep_n]
        out_tokens = [tokens_all[i] for i in order.tolist()]
        out_scores = scores_np[order].tolist()
        return out_tokens, out_scores
