import json
import math
import os
import re
from collections import Counter
import tempfile
import numpy as np

def select_best_model_with_rank(sasrec, grurec, lightgcn, target, not_found_rank=999):
    """
    Returns:
        selected_tool (str): "SASRec" / "GRURec" / "LightGCN"
        min_rank (int): best (smallest) rank_position; 999 if none found
    Rank definition:
        rank_position is 1-based (top item -> 1). If not found -> 999.
    Tie-break:
        If tie includes SASRec -> SASRec; else GRURec; else LightGCN.
    """

    def rank_position(lst, t):
        # 1-based rank_position; 999 if not found
        try:
            return lst.index(t) + 1
        except ValueError:
            return not_found_rank

    ranks = {
        "SASRec": rank_position(sasrec, target),
        "GRURec": rank_position(grurec, target),
        "LightGCN": rank_position(lightgcn, target),
    }

    min_rank = min(ranks.values())

    # All not found
    if min_rank == not_found_rank:
        return "SASRec", not_found_rank

    # Tied best tools
    tied_best = [tool for tool, r in ranks.items() if r == min_rank]

    # Tie-break rule: SASRec > GRURec > LightGCN
    if "SASRec" in tied_best:
        selected = "SASRec"
    elif "GRURec" in tied_best:
        selected = "GRURec"
    else:
        selected = "LightGCN"

    return selected, min_rank



def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 直接从文件对象加载
        return data


def re_match_planner(input_text):
    text = input_text

    # ========== 步骤2：抽取 Explanation 后的完整解释内容 ==========
    # 正则匹配 "Explanation: " 之后的所有文本（直到文本结尾）
    intention_pattern_1 = r'SASRec'
    intention_pattern_2 = r'GRURec'
    intention_pattern_3 = r'LightGCN'
    explanation_pattern = r'palnner_reasoning: (.*)'
    intention = r'recommendation_intention: (.*)'
    user_summary = r'user_summary: (.*)'
    # 使用 re.DOTALL 让 . 匹配换行符（解释内容跨行吗也能完整提取）
    explanation_match = re.search(explanation_pattern, text, re.DOTALL)
    explanation_content = explanation_match.group(1).strip() if explanation_match else ""

    intention = re.search(intention, text, re.DOTALL)
    intention = intention.group(1).strip() if intention else ""

    user_summary = re.search(explanation_pattern, text, re.DOTALL)
    user_summary = user_summary.group(1).strip() if user_summary else ""

    if re.search(intention_pattern_1, text, re.DOTALL):
        intention_tool = "SASRec"
    elif re.search(intention_pattern_2, text, re.DOTALL):
        intention_tool = "GRURec"
    elif re.search(intention_pattern_3, text, re.DOTALL):
        intention_tool = "LightGCN"
    else:
        intention_tool = "SASRec"
    return intention_tool, explanation_content, intention, user_summary

def re_match_reasoner(input_text):
    text = input_text
    # 抽取Judgment后面的内容
    if re.search(r'invalid', text):
        judgment = "invalid"
    else:
        judgment = "valid"


    if re.search(r'confidence:\s*(.+)', text):
        confidence_score = re.search(r'confidence:\s*(.+)', text)
        confidence_score = confidence_score.group(1)
    else:
        confidence_score = "0.9"

    # confidence_score = re.search(r'confidence:\s*(.+)', text)
    # confidence_score = confidence_score.group(1) if confidence_score else None

    # 抽取Reasoning后面的内容
    reasoning_match = re.search(r'reasoner_reasoning:\s*(.+)', text)
    reasoning = reasoning_match.group(1) if reasoning_match else None



    return judgment, reasoning, float(confidence_score)

def re_match_reflector(input_text, candidate_items):
    text = input_text

    # 抽取Reasoning后面的内容
    tools = re.search(r'change_tool:\s*(.*)', text)
    if tools:
        if "SASRec" in tools.group(1):
            tool = "SASRec"
        elif "GRURec" in tools.group(1):
            tool = "GRURec"
        else:
            tool = "None"
    else:
        tool = "None"


    # 抽取Reasoning后面的内容
    Re_ranked_candidate_items = re.search(r're_ranked_candidate_items:\s*(.+)', text)
    Re_ranked_candidate_items = Re_ranked_candidate_items.group(1) if Re_ranked_candidate_items else candidate_items

    return tool, Re_ranked_candidate_items



### 保证生成的列表中不会出现重复，不存在的item########

def normalize_generated(ref, gen):
    ref_c = Counter(ref)
    used = Counter()
    out = []

    # 保留 gen 中合法且未超出 ref 配额的元素（尽量保留原顺序）
    for x in gen:
        if used[x] < ref_c[x]:
            out.append(x)
            used[x] += 1

    # 按 ref 顺序补齐缺失
    for x in ref:
        if used[x] < ref_c[x]:
            out.append(x)
            used[x] += 1
    return out


def find_most_similar_user_id(file_path, user_id, candidate_users):
    """
    输入一个 user_id，从 candidate_users 中选出与其 embedding 余弦相似度最高的用户 id（字符串）。

    file_path: JSON文件路径，格式 { "user_id": [embedding...], ... }
    user_id: 查询用户ID（建议传字符串；若是int请先转str）
    candidate_users: 候选用户ID列表/集合（元素建议为字符串）
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qid = user_id
    if qid not in data:
        raise ValueError(f"用户 {qid} 不存在于 embedding 文件中")

    qvec = np.array(data[qid], dtype=np.float32)
    qnorm = np.linalg.norm(qvec)
    if qnorm == 0:
        raise ValueError(f"用户 {qid} 的 embedding 是零向量，无法计算相似度")

    best_id, best_sim = None, -1e18

    # 只遍历候选集合
    for cid in candidate_users:
        oid = cid
        if oid == qid:
            continue
        if oid not in data:
            continue  # 候选不在embedding文件里就跳过

        ovec = np.array(data[oid], dtype=np.float32)
        onorm = np.linalg.norm(ovec)
        if onorm == 0:
            sim = 0.0
        else:
            sim = float(np.dot(qvec, ovec) / (qnorm * onorm))
            sim = max(-1.0, min(1.0, sim))

        if sim > best_sim:
            best_sim = sim
            best_id = oid

    if best_id is None:
        raise ValueError("candidate_users 中没有任何可用的用户（不在embedding文件、全是自己、或全是零向量）")

    return best_id


def safe_parse_json(text):
    try:
        return json.loads(text)
    except:
        pass
    
    # 方法2：查找最外层的{}
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except:
            pass
    
    # 方法3：使用简单正则匹配（无递归）
    # 匹配至少有一个有效内容的{}
    simple_pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
    matches = re.findall(simple_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except:
            continue
    
    return {}

from typing import Dict, List, Tuple, Optional
import numpy as np

def _token_to_iid(token: str, token2id: Dict[str, int]) -> Optional[int]:
    """
    支持 'item_1282' / '1282' 两种形式，映射为 RecBole 内部 iid（int）。
    """
    if token in token2id:
        return int(token2id[token])
    if token.startswith("item_"):
        pure = token[5:]
        if pure in token2id:
            return int(token2id[pure])
    return None


def rank_candidates_by_table_tokens(
    user_id: str,
    candidate_tokens: List[str],
    user_index: Dict[str, int],
    score_mmap: np.memmap,                 # (n_users, n_items)
    token2id: Dict[str, int],
    drop_ratio: float = 0.0,
    drop_unknown: bool = True,
    min_keep: int = 5,                     # 新增：至少保留这么多
) -> Tuple[List[str], List[float]]:
    if not (0.0 <= float(drop_ratio) <= 1.0):
        raise ValueError(f"drop_ratio must be in [0, 1], got {drop_ratio}")

    row = user_index[str(user_id)]
    n_items = int(score_mmap.shape[1])

    kept_tokens: List[str] = []
    kept_iids: List[int] = []

    for t in candidate_tokens:
        iid = _token_to_iid(str(t), token2id)
        if iid is None:
            if drop_unknown:
                continue
            else:
                continue
        if 0 <= int(iid) < n_items:
            kept_tokens.append(str(t))
            kept_iids.append(int(iid))

    if not kept_iids:
        return [], []

    cand_iids = np.asarray(kept_iids, dtype=np.int32)
    cand_scores = score_mmap[row, cand_iids].astype(np.float32)

    order = np.argsort(-cand_scores)  # 降序

    n = len(order)
    # 目标保留数量：max(min_keep, floor(n*(1-drop_ratio)))
    keep_n = max(int(min_keep), int(math.floor(n * (1.0 - float(drop_ratio)))))

    # 不能超过 n
    keep_n = min(keep_n, n)

    # 如果 n 本来就 < min_keep，就直接全保留
    if keep_n <= 0:
        return [], []

    order = order[:keep_n]
    top_tokens = [kept_tokens[i] for i in order.tolist()]
    top_scores = cand_scores[order].tolist()
    return top_tokens, top_scores



# def main():
#     # 1) 构造测试用 user_index / token2id
#     user_index = {"u0": 0, "u1": 1}
#     token2id = {
#         "item_0": 0, "item_1": 1, "item_2": 2, "item_3": 3, "item_4": 4,
#         # 也可混合提供无前缀形式，测试映射兼容性
#         "5": 5
#     }

#     # 2) 创建一个 memmap 打分表 (2 users x 6 items)
#     with tempfile.TemporaryDirectory() as td:
#         path = os.path.join(td, "scores.f16")
#         scores = np.memmap(path, dtype=np.float16, mode="w+", shape=(2, 6))  # 用法同官方示例 [web:71]

#         # user u0 的 item 分数：item_3 最高，其次 item_1，再 item_5
#         scores[0, :] = np.array([0.10, 0.80, 0.20, 0.90, 0.30, 0.60], dtype=np.float16)
#         # user u1 的 item 分数：item_2 最高
#         scores[1, :] = np.array([0.05, 0.10, 0.99, 0.20, 0.30, 0.40], dtype=np.float16)
#         scores.flush()

#         # 3) 准备候选集合（含未知 token）
#         candidate_tokens = ["item_1", "item_3", "item_5", "item_2", "item_999"]

#         # 4) 调用函数
#         top_tokens, top_scores = rank_candidates_by_table_tokens(
#             user_id="u0",
#             candidate_tokens=candidate_tokens,
#             user_index=user_index,
#             score_mmap=scores,
#             token2id=token2id,
#             topk=3,
#             drop_unknown=True,
#         )

#         print("Top tokens:", top_tokens)
#         print("Top scores:", top_scores)

#         # 5) 简单断言（可选）：预期 top3 是 item_3 > item_1 > item_5
#         assert top_tokens == ["item_3", "item_1", "item_5"], f"Unexpected order: {top_tokens}"
#         # 分数用 float16/float32 可能有微小误差，这里只比近似
#         assert abs(top_scores[0] - 0.9) < 1e-3
#         assert abs(top_scores[1] - 0.8) < 1e-3
#         assert abs(top_scores[2] - 0.6) < 1e-3

#         print("OK: rank_candidates_by_table_tokens works as expected.")


