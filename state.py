# state.py
from dataclasses import dataclass
from typing import Dict, List, Any
from typing_extensions import TypedDict, NotRequired  # Py3.9: NotRequired/TypedDict from typing_extensions [web:156][web:158]


class RecommendationState(TypedDict):
    used_filter_tools: List[str]
    
    tool_set: Any  # 兼容 set[str] / str
    NDCG: float
    reorder_conversation: Dict[str, Any]

    reasoner_memory: List[str]
    reflector_memory: List[str]

    train_step: int
    train: bool
    confidence: float
    model_choice: str

    user_profile: Dict[str, Any]
    user_id: str

    # 关键：运行时统一规范化为 token 列表
    candidate_items: List[str]
    re_candidate_items: List[str]

    target: str
    planner_explanation: str
    planner_intention: str

    reasoner_judgment: str
    reasoner_reasoning: str

    reflection_feedback: str
    final_recommendations: List[str]
    final_explanation: str
    planner_summary: str

    iteration_count: int
    max_iterations: int

    # optional eval
    ground_truth_items: NotRequired[List[str]]
    metrics: NotRequired[Dict[str, float]]

    # filtering (new)
    need_filter: bool
    filter_reason: str
    filter_mode: str            # "tool" / "llm" / "none"
    filter_tool: str            # "sasrec/gru4rec/lightgcn/stamp/glintru/kgat"
    drop_ratio: float
    filter_plan_reason: str

    filter_round: int
    max_filter_rounds: int
    min_keep: int
    filter_log: Dict[str, Any]
    


@dataclass
class UserMemory:
    """结构化推理记忆单元"""
    target_item_inforamtion: str
    user_summary: str
    rank_conversation: List[str]
    rank_conversation_summary: List[str]
