# rec_utils/rec_helpers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils import find_most_similar_user_id


# ---------------- Tool registry ----------------
@dataclass(frozen=True)
class ToolRegistry:
    """
    负责：
    - 统一工具命名（alias）
    - tool -> memmap_dir 路径映射

    你也可以在初始化时传入自定义 mapping/alias，实现复用到别的数据集或目录结构。
    """
    tool_to_dir: Dict[str, str]
    alias: Dict[str, str]

    @classmethod
    def default_ml1m(cls) -> "ToolRegistry":
        alias = {
            "sasrec": "sasrec",
            "grurec": "gru4rec",
            "gru4rec": "gru4rec",
            "lightgcn": "lightgcn",
            "stamp": "stamp",
            "glintru": "glintru",
            "kgat": "kgat",
        }
        tool_to_dir = {
            "glintru": "/home/liujuntao/Agent4Rec/data/ml-1m/glintru_scores_memmap",
            "gru4rec": "/home/liujuntao/Agent4Rec/data/ml-1m/gru4rec_scores_memmap",
            "kgat": "/home/liujuntao/Agent4Rec/data/ml-1m/kgat_scores_memmap",
            "lightgcn": "/home/liujuntao/Agent4Rec/data/ml-1m/lightgcn_scores_memmap",
            "sasrec": "/home/liujuntao/Agent4Rec/data/ml-1m/sasrec_scores_memmap",
            "stamp": "/home/liujuntao/Agent4Rec/data/ml-1m/stamp_scores_memmap",
        }
        return cls(tool_to_dir=tool_to_dir, alias=alias)

    def normalize(self, tool_name: str) -> str:
        x = str(tool_name).strip().lower()
        return self.alias.get(x, x)

    def memmap_dir(self, tool_name: str) -> str:
        key = self.normalize(tool_name)
        if key in self.tool_to_dir:
            return self.tool_to_dir[key]
        return self.tool_to_dir.get("glintru", "")


# ---------------- Item formatting ----------------
def format_candidates_with_meta(candidate_tokens: List[str], item_meta: Dict[str, Any]) -> str:
    """
    把 token 列表格式化成可读文本，供 LLM reasoner/reflector 使用。
    """
    lines = []
    for tok in candidate_tokens:
        meta = item_meta.get(tok, {}) if isinstance(item_meta, dict) else {}
        title = meta.get("movie_title", "Unknown Title")
        year = meta.get("release_year", "N/A")
        genre = meta.get("genre", "Unknown Genre")
        lines.append(f"{tok}: {title} ({year}) [{genre}]")
    return "\n".join(lines)


# ---------------- Similar memory retrieval ----------------
@dataclass
class SimilarMemoryRetriever:
    """
    给定 reasoning_memory（一个 dict: user_id -> memory），
    根据 embedding/profile 文件在候选用户中检索最相似 user 的 memory。
    """
    reasoning_memory: Dict[str, Any]

    def get(self, user_emb_profile_path: str, user_id: str) -> Optional[Any]:
        if not self.reasoning_memory:
            return None
        can_users = list(self.reasoning_memory.keys())
        similar_user_id = find_most_similar_user_id(
            file_path=user_emb_profile_path,
            user_id=user_id,
            candidate_users=can_users,
        )
        return self.reasoning_memory.get(similar_user_id)
