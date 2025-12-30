import numpy as np

import numpy as np

# Recall@K (单个 target)
def recall(candidate_items, target_item, k=5):
    """
    candidate_items: List[Any] 排序后的候选列表（高->低）
    target_item: Any 单个目标 item
    """
    if k <= 0 or not candidate_items:
        return 0.0
    
    top_k_items = candidate_items[:k]
    return 1.0 if target_item in top_k_items else 0.0


# NDCG@K (单个 target)
def ndcg(candidate_items, target_item, k=5):
    """
    单个 target 的 NDCG@K：
    - 若 target 在 topK 的第 r(1-based) 位：DCG = 1/log2(r+1)
    - IDCG 恒为 1（理想情况下 target 在第1位）
    """
    if k <= 0 or not candidate_items:
        return 0.0
    
    top_k_items = candidate_items[:k]
    try:
        rank = top_k_items.index(target_item) + 1  # 1-based rank
    except ValueError:
        return 0.0

    dcg = 1.0 / np.log2(rank + 1)
    idcg = 1.0  # 单个相关项的理想DCG
    return dcg / idcg



# 2. 计算 Recall（适配批量用户输入）
def recall_all(candidate_items, target_item, k=5):
    """
    计算批量用户的Recall@K，返回每个用户的Recall值 + 整体平均值
    （适配每个用户仅有1个目标项的场景）
    
    参数：
    candidate_items: 嵌套列表，[[user1候选列表], [user2候选列表], ...]
    target_item: 嵌套列表/列表，[[user1目标项], [user2目标项], ...] 或 [user1目标项, user2目标项, ...]
    k: 取前K个候选项计算
    
    返回：
    recall_list: 每个用户的Recall值列表
    avg_recall: 所有用户的Recall平均值
    """
    recall_list = []
    # 统一格式：确保target_item是嵌套列表（兼容单层列表输入）
    target_item = [[t] if not isinstance(t, list) else t for t in target_item]
    
    for cand, target in zip(candidate_items, target_item):
        # 处理空目标（避免异常）
        if not target or len(target) == 0:
            recall_list.append(0.0)
            continue
        
        # 每个用户仅1个目标项，直接取第一个
        target_single = target[0]
        # 取当前用户的Top-K候选（不足K则取全部）
        top_k = cand[:k]
        
        # Recall@K = 目标项是否出现在Top-K中（0或1）
        recall_val = 1.0 if target_single in top_k else 0.0
        recall_list.append(recall_val)
    
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    # 返回列表+平均值（保持原函数返回逻辑，若需仅返回平均值可调整）
    return avg_recall

def ndcg_all(candidate_items, target_item, k=5):
    """
    计算批量用户的NDCG@K，返回每个用户的NDCG值 + 整体平均值
    （适配每个用户仅有1个目标项的场景，简化计算逻辑）
    
    参数：
    candidate_items: 嵌套列表，[[user1候选列表], [user2候选列表], ...]
    target_item: 嵌套列表/列表，[[user1目标项], [user2目标项], ...] 或 [user1目标项, user2目标项, ...]
    k: 取前K个候选项计算
    
    返回：
    ndcg_list: 每个用户的NDCG值列表
    avg_ndcg: 所有用户的NDCG平均值
    """
    ndcg_list = []
    # 统一格式：确保target_item是嵌套列表（兼容单层列表输入）
    target_item = [[t] if not isinstance(t, list) else t for t in target_item]
    
    for cand, target in zip(candidate_items, target_item):
        # 处理空目标
        if not target or len(target) == 0:
            ndcg_list.append(0.0)
            continue
        
        # 每个用户仅1个目标项
        target_single = target[0]
        # 取当前用户的Top-K候选
        top_k_cand = cand[:k]
        
        # 计算DCG：仅需判断目标项是否在Top-K中，以及其位置
        dcg = 0.0
        for i, item in enumerate(top_k_cand):
            if item == target_single:
                # 目标项出现在第i+1位（i从0开始），计算得分
                dcg = 1.0 / np.log2(i + 2)  # 找到后直接break，无需计算后续
                break
        
        # 计算IDCG：理想情况是目标项出现在第1位（得分=1/log2(2)=1）
        idcg = 1.0 / np.log2(2)  # 固定值=1.0（因为仅1个目标项，理想位置是第1位）
        
        # 计算NDCG（避免除以0）
        ndcg_val = dcg / idcg if idcg != 0 else 0.0
        ndcg_list.append(ndcg_val)
    
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    # 返回列表+平均值
    return avg_ndcg
