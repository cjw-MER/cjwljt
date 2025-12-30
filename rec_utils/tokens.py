from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_ITEM_PAT_1 = re.compile(r"item_ID:(item_\d+)")
_ITEM_PAT_2 = re.compile(r"(item_\d+)")
_ITEM_NUM_PAT = re.compile(r"item_(\d+)$")


def normalize_candidate_tokens(candidate_items: Any) -> List[str]:
    """
    将候选 items（可能是 str/list/dict 混合结构）拍平并提取出规范 token：item_123。

    规则：
    - 支持输入中包含 'item_ID:item_123' 或直接包含 'item_123'
    - dict 如果包含 'id' 字段优先取 id
    - 去重并保持第一次出现的顺序
    """
    if candidate_items is None:
        return []

    queue = [candidate_items]
    strings: List[str] = []

    while queue:
        x = queue.pop(0)
        if x is None:
            continue
        if isinstance(x, str):
            strings.append(x)
        elif isinstance(x, list):
            queue.extend(x)
        elif isinstance(x, dict):
            if "id" in x:
                strings.append(str(x["id"]))
            else:
                strings.append(str(x))
        else:
            strings.append(str(x))

    out: List[str] = []
    seen = set()
    for s in strings:
        s = str(s)

        hits = _ITEM_PAT_1.findall(s)
        if not hits:
            hits = _ITEM_PAT_2.findall(s)

        for t in hits:
            if t not in seen:
                seen.add(t)
                out.append(t)

    return out


def token_to_iid(token: str, token2id: Optional[Dict[str, int]] = None) -> Optional[int]:
    """
    将 token 转为内部 item id（int）。

    支持三种情况：
    - token2id 映射直接命中（token 或去掉 item_ 前缀 或补 item_ 前缀）
    - fallback：解析 item_123 得到 123
    """
    token = str(token)

    if token2id is not None:
        if token in token2id:
            return int(token2id[token])

        if token.startswith("item_") and token[5:] in token2id:
            return int(token2id[token[5:]])

        if (not token.startswith("item_")) and (f"item_{token}" in token2id):
            return int(token2id[f"item_{token}"])

    m = _ITEM_NUM_PAT.match(token)
    if m:
        return int(m.group(1))
    return None


def token_to_int(token: str) -> Optional[int]:
    """仅解析 item_123 -> 123，用于 metric/ndcg 等。"""
    m = _ITEM_NUM_PAT.match(str(token))
    return int(m.group(1)) if m else None
