# agent.py
import os
import json
import math
import re
import sys
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from langgraph.graph import StateGraph, END

from prompts import planner, reasoner, reflector
from utils import load_json_file, safe_parse_json, find_most_similar_user_id
from metric import ndcg
from state import RecommendationState
from tools import retrieval_topk_load, stdout_retrived_items_load
from api import ZHI


# ============================================================
# 0) Candidate normalization (CRITICAL FIX)
# ============================================================
def normalize_candidate_tokens(candidate_items: Any) -> List[str]:
    """
    目标输出：["item_123", "item_456", ...]
    兼容输入：
    - ["item_1", "item_2"]
    - [{"id": "item_1", "title":...}, {"id": "item_2"...}]
    - "item_ID:item_1, title:...\nitem_ID:item_2, title:..."
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
        hits = re.findall(r"item_ID:(item_\d+)", s)
        if not hits:
            hits = re.findall(r"(item_\d+)", s)

        for t in hits:
            if t not in seen:
                seen.add(t)
                out.append(t)

    return out


# ============================================================
# 1) Filtering helpers
# ============================================================
def _token_to_iid(token: str, token2id: Optional[Dict[str, int]] = None) -> Optional[int]:
    """
    修复版：不早退，保证 fallback 可达。
    """
    token = str(token)
    if token2id is not None:
        if token in token2id:
            return int(token2id[token])
        if token.startswith("item_") and token[5:] in token2id:
            return int(token2id[token[5:]])
        if (not token.startswith("item_")) and (f"item_{token}" in token2id):
            return int(token2id[f"item_{token}"])

    m = re.match(r"item_(\d+)$", token)
    if m:
        return int(m.group(1))
    return None


def rank_candidates_by_memdir(
    mem_dir: str,
    user_id: str,
    candidate_tokens: List[str],
    drop_ratio: float = 0.0,
    drop_unknown: bool = True,
    min_keep: int = 5,
    unknown_score: float = -1e9,
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[str], List[float]]:
    if not (0.0 <= float(drop_ratio) <= 1.0):
        raise ValueError(f"drop_ratio must be in [0, 1], got {drop_ratio}")

    if cache is None:
        cache = {}

    # ---------- 0) lazy load pack ----------
    if mem_dir not in cache:
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

        cache[mem_dir] = {
            "meta": meta,
            "user_index": user_index,
            "score_mmap": score_mmap,
            "item_token2id": item_token2id,
            "user_token2id": user_token2id,
        }

    pack = cache[mem_dir]
    user_index: Dict[str, int] = pack["user_index"]
    score_mmap: np.memmap = pack["score_mmap"]
    item_token2id: Dict[str, int] = pack["item_token2id"]
    user_token2id: Optional[Dict[str, int]] = pack.get("user_token2id")

    # ---------- 1) resolve row index ----------
    uid_raw = str(user_id)
    row = None

    if user_token2id is None:
        row = user_index.get(uid_raw)
    else:
        u_tok = uid_raw if uid_raw.startswith("user_") else f"user_{uid_raw}"
        internal_uid = user_token2id.get(uid_raw, user_token2id.get(u_tok, None))
        if internal_uid is not None:
            row = user_index.get(str(int(internal_uid)))
        if row is None:
            row = user_index.get(uid_raw)

    if row is None:
        return [], []

    n_items = int(score_mmap.shape[1])

    # ---------- 2) map items -> iid + read scores ----------
    tokens_all: List[str] = []
    scores_all: List[float] = []

    for t in candidate_tokens:
        tok = str(t)
        iid = _token_to_iid(tok, item_token2id)

        if iid is None or not (0 <= int(iid) < n_items):
            if drop_unknown:
                continue
            tokens_all.append(tok)
            scores_all.append(float(unknown_score))
            continue

        tokens_all.append(tok)
        scores_all.append(float(score_mmap[row, int(iid)]))

    if not tokens_all:
        return [], []

    scores_np = np.asarray(scores_all, dtype=np.float32)

    # ---------- 3) sort desc + drop tail ----------
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


# ============================================================
# 3) Agent
# ============================================================
class RecommendationAgent:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

        self.workflow = self._build_workflow()

        self.tool_memory = load_json_file(args.tool_memory_file)
        self.reasoning_memory = load_json_file(args.reasoning_memory_file)

        meta_path = getattr(args, "item_meta_file", "/home/liujuntao/Agent4Rec/data/ml-1m/ml-1m.item.json")
        if os.path.exists(meta_path):
            self.item_meta = load_json_file(meta_path)
        else:
            print(f"Warning: Item meta file not found at {meta_path}. Reasoning will lack metadata.")
            self.item_meta = {}

        self.api_planner = ZHI()
        self.api_reasoner = ZHI()
        self.api_reflector = ZHI()
        self.api_summray = ZHI()

        self.mem_cache: Dict[str, Dict[str, Any]] = {}

        # logging dirs
        self.enable_cand_log = bool(getattr(args, "enable_cand_log", True))
        self.cand_log_dir = getattr(args, "cand_log_dir", "./candidate_change_logs")
        self.final_log_dir = getattr(args, "final_log_dir", "./final_recommendations_logs")

    # ============================================================
    # Logging helpers (two dirs)
    # ============================================================
    def _user_jsonl_path(self, base_dir: str, user_id: str) -> str:
        uid = str(user_id)
        return os.path.join(base_dir, f"user_{uid}.jsonl")

    def _append_jsonl(self, path: str, obj: dict):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _log_change_to_dir(
        self,
        base_dir: str,
        state: RecommendationState,
        stage: str,
        before: Any,
        after: Any,
        extra: Optional[Dict[str, Any]] = None,
    ):
        if not self.enable_cand_log:
            return

        before_norm = normalize_candidate_tokens(before)
        after_norm = normalize_candidate_tokens(after)
        if before_norm == after_norm:
            return

        payload = {
            "ts": time.time(),
            "stage": stage,
            "user_id": str(state.get("user_id", "")),
            "iteration_count": int(state.get("iteration_count", 0)),
            "filter_round": int(state.get("filter_round", 0)),
            "need_filter": bool(state.get("need_filter", False)),
            "filter_mode": str(state.get("filter_mode", "")),
            "filter_tool": str(state.get("filter_tool", "")),
            "drop_ratio": float(state.get("drop_ratio", 0.0)),
            "before_len": len(before_norm),
            "after_len": len(after_norm),
            "before": before_norm,
            "after": after_norm,
        }
        if extra:
            payload["extra"] = extra

        path = self._user_jsonl_path(base_dir, str(state.get("user_id", "")))
        self._append_jsonl(path, payload)

    def _log_candidate_change(self, state, stage, before, after, extra=None):
        self._log_change_to_dir(self.cand_log_dir, state, stage, before, after, extra)

    def _log_final_change(self, state, stage, before, after, extra=None):
        self._log_change_to_dir(self.final_log_dir, state, stage, before, after, extra)

    # ============================================================
    # Internal helpers
    # ============================================================
    def _normalize_tool_name(self, tool_name: str) -> str:
        x = str(tool_name).strip().lower()
        alias = {
            "sasrec": "sasrec",
            "grurec": "gru4rec",
            "gru4rec": "gru4rec",
            "lightgcn": "lightgcn",
            "stamp": "stamp",
            "glintru": "glintru",
            "kgat": "kgat",
        }
        return alias.get(x, x)

    def _tool_to_memmap_dir(self, tool_name: str) -> str:
        key = self._normalize_tool_name(tool_name)
        mapping = {
            "glintru": "/home/liujuntao/Agent4Rec/data/ml-1m/glintru_scores_memmap",
            "gru4rec": "/home/liujuntao/Agent4Rec/data/ml-1m/gru4rec_scores_memmap",
            "kgat": "/home/liujuntao/Agent4Rec/data/ml-1m/kgat_scores_memmap",
            "lightgcn": "/home/liujuntao/Agent4Rec/data/ml-1m/lightgcn_scores_memmap",
            "sasrec": "/home/liujuntao/Agent4Rec/data/ml-1m/sasrec_scores_memmap",
            "stamp": "/home/liujuntao/Agent4Rec/data/ml-1m/stamp_scores_memmap",
        }
        if key not in mapping:
            return mapping["glintru"]
        return mapping[key]

    def _format_candidates_with_meta(self, candidate_tokens: List[str]) -> str:
        lines = []
        for tok in candidate_tokens:
            meta = self.item_meta.get(tok, {})
            title = meta.get("movie_title", "Unknown Title")
            year = meta.get("release_year", "N/A")
            genre = meta.get("genre", "Unknown Genre")
            lines.append(f"{tok}: {title} ({year}) [{genre}]")
        return "\n".join(lines)

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(RecommendationState)

        workflow.add_edge("__start__", "planner")
        workflow.add_node("planner", self.planner_agent)
        workflow.add_node("tool_node", self.tool_node)
        workflow.add_node("reasoner", self.reasoner_agent)
        workflow.add_node("reflector", self.reflector_agent)

        workflow.set_entry_point("planner")

        workflow.add_edge("planner", "tool_node")
        workflow.add_edge("tool_node", "reasoner")
        workflow.add_edge("reasoner", "reflector")

        workflow.add_conditional_edges(
            "reflector",
            self.should_continue_or_finish,
            {"continue": "tool_node", "finish": END},
        )
        return workflow.compile()

    def retrive_memory(self, path, user_id):
        can_users = list(self.reasoning_memory.keys())
        similar_user_id = find_most_similar_user_id(
            file_path=path, user_id=user_id, candidate_users=can_users
        )
        return self.reasoning_memory[similar_user_id]

    # ============================================================
    # Nodes
    # ============================================================
    def planner_agent(self, state: RecommendationState) -> RecommendationState:
        print("Planner working")
        user_profile = state["user_profile"]
        user_id = state["user_id"]

        if getattr(self.args, "planner_memory_use", False):
            similar_memory = self.retrive_memory(self.args.user_emb_profile, user_id)
            planner_prompt = planner(user_profile, similar_memory)
        else:
            planner_prompt = planner(user_profile)

        system_planner_prompt = """Output a JSON object strictly following this format:
        {
        "tool_selected": "SASRec" or "GRURec" or "LightGCN",
        "planner_intention": "1-2 movie genres",
        "planner_reasoning": "Brief explanation. Max 60 words."
        }
        """
        messages = [
            {"role": "system", "content": system_planner_prompt},
            {"role": "user", "content": planner_prompt},
        ]
        response = self.api_planner.chat(messages)
        data = safe_parse_json(response)

        state["planner_intention"] = data.get("planner_intention", "")
        state["planner_explanation"] = data.get("planner_reasoning", "")
        state["model_choice"] = data.get("tool_selected", "SASRec")
        state["planner_summary"] = ""
        return state

    def tool_node(self, state: RecommendationState) -> RecommendationState:
        """
        tool_node 只负责“拉取/传递候选集”，不做 tool 排序/过滤。
        """
        print("tool working")
        user_id = state["user_id"]

        before_cand = normalize_candidate_tokens(state.get("candidate_items", []))

        if state.get("re_candidate_items"):
            cand_raw = state["re_candidate_items"]
        else:
            if state["iteration_count"] == 0:
                tool_set = state.get("tool_set", {"SASRec", "GRURec", "LightGCN"})
                if isinstance(tool_set, str):
                    tool_set = set([x.strip() for x in tool_set.split(",")])

                if state["model_choice"] in tool_set:
                    external_item_list = retrieval_topk_load(
                        model_file="/home/liujuntao/test_topk_prediction_user2000_random_100.json",
                        user_id=user_id,
                    )
                    cand_raw = stdout_retrived_items_load(external_item_list)
                else:
                    cand_raw = state.get("candidate_items", [])
            else:
                cand_raw = state.get("candidate_items", [])

        cand = normalize_candidate_tokens(cand_raw)
        state["candidate_items"] = cand

        self._log_candidate_change(
            state=state,
            stage="tool_node.fetch_candidates",
            before=before_cand,
            after=cand,
            extra={"source": "re_candidate_items" if state.get("re_candidate_items") else "retrieval_or_state"},
        )
        return state

    def reasoner_agent(self, state: RecommendationState) -> RecommendationState:
        print("Reasoner working")

        state["candidate_items"] = normalize_candidate_tokens(state.get("candidate_items", []))
        candidate_items = state["candidate_items"]
        user_id = state["user_id"]

        formatted_candidates = self._format_candidates_with_meta(candidate_items)

        input_memory = state.get("reasoner_memory", [])
        planner_intention = state["planner_intention"]
        planner_reasoning = state["planner_explanation"]

        similar_memory = None
        if getattr(self.args, "reasoner_memory_use", False):
            similar_memory = self.retrive_memory(self.args.user_emb_profile, user_id)

        prompt = reasoner(
            formatted_candidates,
            planner_intention,
            planner_reasoning,
            input_memory,
            state["user_profile"],
            similar_memory,
        )

        system_prompt = """Output a JSON object strictly following this format:
        {
        "judgment": "valid" or "invalid",
        "confidence": 0.0-1.0,
        "reasoner_reasoning": "Brief reasoning",
        "need_filter": true or false,
        "filter_reason": "If need_filter=true, explain why filtering is needed (max 40 words)."
        }
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = self.api_reasoner.chat(messages)
        data = safe_parse_json(response)
        print("Reasoner response:", data)

        state["reasoner_judgment"] = data.get("judgment", "valid")
        state["confidence"] = float(data.get("confidence", 0.0))
        state["reasoner_reasoning"] = data.get("reasoner_reasoning", "")

        state["need_filter"] = bool(data.get("need_filter", False))
        state["filter_reason"] = str(data.get("filter_reason", ""))

        state["reasoner_memory"].append(
            f"round={state['iteration_count']} judgment={state['reasoner_judgment']} "
            f"conf={state['confidence']} reason={state['reasoner_reasoning']}"
        )
        return state

    def reflector_agent(self, state: RecommendationState) -> RecommendationState:
        print("Reflector working")

        # normalize
        state["candidate_items"] = normalize_candidate_tokens(state.get("candidate_items", []))
        state["re_candidate_items"] = normalize_candidate_tokens(state.get("re_candidate_items", []))
        state["final_recommendations"] = normalize_candidate_tokens(state.get("final_recommendations", []))

        candidate_items = state["candidate_items"]
        re_candidate_items = state.get("re_candidate_items", [])
        user_id = state["user_id"]

        before_re = re_candidate_items if re_candidate_items else candidate_items
        before_cand = list(candidate_items)
        before_final = list(state.get("final_recommendations", []))

        formatted_candidates = self._format_candidates_with_meta(candidate_items)
        formatted_re_candidates = self._format_candidates_with_meta(re_candidate_items)

        similar_memory = (
            self.retrive_memory(self.args.user_emb_profile, user_id)
            if getattr(self.args, "reflector_memory_use", False)
            else None
        )

        used_filter_tools = state.get("used_filter_tools", [])
        prompt = reflector(
            state["reasoner_reasoning"],
            formatted_candidates,
            formatted_re_candidates,
            state["user_profile"],
            memory=similar_memory,
            need_filter=state.get("need_filter", False),
            filter_reason=state.get("filter_reason", ""),
            min_keep=int(state.get("min_keep", 5)),
            used_filter_tools=used_filter_tools,
        )

        system_prompt = """Output a JSON object strictly following this format:
        {
        "filter_mode": "tool" or "llm" or "none",
        "filter_tool": "sasrec" or "gru4rec" or "lightgcn" or "stamp" or "glintru" or "kgat",
        "drop_ratio": 0.0-1.0,
        "filter_plan_reason": "why this mode/tool and drop_ratio (max 40 words)",
        "re_ranked_candidate_items": []
        }
        """

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        try:
            response = self.api_reflector.chat(messages)
            data = safe_parse_json(response)
            print("Reflector response:", data)
        except Exception as e:
            target_dbg = state.get("target", None)
            # 打到 stderr + 强制 flush，确保你一定能看到 uid [web:121][web:126]
            print(f"[Reflector ERROR] uid={user_id} target={target_dbg} err={e}",
                file=sys.stderr, flush=True)
            raise  # 中断流程

        filter_mode = str(data.get("filter_mode", "none")).lower()
        filter_tool = self._normalize_tool_name(
            data.get("filter_tool", state.get("model_choice", "sasrec"))
        )
        drop_ratio = float(data.get("drop_ratio", 0.0))
        filter_plan_reason = str(data.get("filter_plan_reason", ""))

        # ---- 强制：若从未 tool 过滤过，则必须先按 reflector 指定 tool 做一次过滤 ----
        has_tool_filtered = bool(state.get("has_tool_filtered", False))
        force_tool_filter_first = not has_tool_filtered

        min_keep = int(state.get("min_keep", 5))
        max_filter_rounds = int(state.get("max_filter_rounds", 3))
        filter_round = int(state.get("filter_round", 0))

        kept_scores_head: List[float] = []
        current_candidates = list(candidate_items)

        def _apply_tool_filter(tokens_in: List[str],dr: float) -> Tuple[List[str], List[float], str]:
            mem_dir_local = self._tool_to_memmap_dir(filter_tool)
            out_tokens, out_scores = rank_candidates_by_memdir(
                mem_dir=mem_dir_local,
                user_id=str(user_id),
                candidate_tokens=tokens_in,
                drop_ratio=dr,
                drop_unknown=True,
                min_keep=min_keep,
                cache=self.mem_cache,
            )
            return out_tokens, out_scores, mem_dir_local

        # 1) 第一步：必要时先 tool 过滤（无视 reasoner need_filter）
        tool_filtered = current_candidates
        mem_dir_used = ""
        if force_tool_filter_first:
            tool_filtered, tool_scores, mem_dir_used = _apply_tool_filter(current_candidates,0.0)
            if tool_scores:
                kept_scores_head = tool_scores[:10]
            if filter_round < max_filter_rounds and len(current_candidates) > min_keep:
                state["filter_round"] = filter_round + 1
            state["has_tool_filtered"] = True
            state["filter_log"] = {
                "mode": "tool_forced_first",
                "tool": filter_tool,
                "mem_dir": mem_dir_used,
                "drop_ratio": drop_ratio,
                "after": len(tool_filtered),
                "where": "reflector_agent",
            }
            used = list(state.get("used_filter_tools", []))
            if filter_tool not in used:
                used.append(filter_tool)
            state["used_filter_tools"] = used

        else:
            # 未强制时也可能根据 filter_mode 走 tool
            if filter_mode == "tool":
                tool_filtered, tool_scores, mem_dir_used = _apply_tool_filter(current_candidates,drop_ratio)
                if tool_scores:
                    kept_scores_head = tool_scores[:10]
                if filter_round < max_filter_rounds and len(current_candidates) > min_keep:
                    state["filter_round"] = filter_round + 1
                state["has_tool_filtered"] = True
                state["filter_log"] = {
                    "mode": "tool",
                    "tool": filter_tool,
                    "mem_dir": mem_dir_used,
                    "drop_ratio": drop_ratio,
                    "after": len(tool_filtered),
                    "where": "reflector_agent",
                }
                used = list(state.get("used_filter_tools", []))
                if filter_tool not in used:
                    used.append(filter_tool)
                state["used_filter_tools"] = used


        # 2) 第二步：根据 filter_mode 决定是否再做 LLM 重排/none
        base_for_rerank = tool_filtered

        if (not force_tool_filter_first) and filter_mode == "none":
            final_re_ranked = base_for_rerank  # 原样
        elif filter_mode == "llm":
            re_ranked_raw = data.get("re_ranked_candidate_items", [])
            re_ranked = normalize_candidate_tokens(re_ranked_raw)

            if not re_ranked:
                merged = base_for_rerank
            else:
                cand_set = set(base_for_rerank)
                valid_ranked = [x for x in re_ranked if x in cand_set]
                missing = [x for x in base_for_rerank if x not in set(valid_ranked)]
                merged = valid_ranked + missing

            # LLM 模式下也允许用 drop_ratio 截断
            keep_n = max(min_keep, int(math.floor(len(merged) * (1.0 - float(drop_ratio)))))
            keep_n = min(keep_n, len(merged))
            final_re_ranked = merged[:keep_n]

            if filter_round < max_filter_rounds and len(base_for_rerank) > min_keep:
                state["filter_round"] = int(state.get("filter_round", 0)) + 1

            state["filter_log"] = {
                "mode": "llm",
                "drop_ratio": drop_ratio,
                "after": len(final_re_ranked),
                "where": "reflector_agent",
            }
        else:
            # tool / forced_first：最终就是 tool_filtered（已经按分数排序并截断）
            final_re_ranked = base_for_rerank

        # 3) 记录 reflector 决策（候选变化日志：cand_log_dir）
        self._log_candidate_change(
            state=state,
            stage="reflector.rerank_or_plan",
            before=before_re,
            after=final_re_ranked,
            extra={
                "filter_mode": filter_mode,
                "filter_tool": filter_tool,
                "drop_ratio": drop_ratio,
                "filter_plan_reason": filter_plan_reason,
                "kept_scores_head": kept_scores_head,
                "mem_dir": mem_dir_used,
                "forced_first_tool_filter": force_tool_filter_first,
            },
        )

        # 4) 先更新 candidate_items，再更新 final_recommendations
        state["candidate_items"] = normalize_candidate_tokens(final_re_ranked)
        self._log_candidate_change(
            state=state,
            stage="reflector.update_candidate_items",
            before=before_cand,
            after=state["candidate_items"],
        )

        state["re_candidate_items"] = normalize_candidate_tokens(final_re_ranked)
        state["final_recommendations"] = normalize_candidate_tokens(final_re_ranked)

        # 5) final_recommendations 单独目录记录（final_log_dir）
        self._log_final_change(
            state=state,
            stage="final_recommendations.update",
            before=before_final,
            after=state["final_recommendations"],
            extra={
                "filter_mode": filter_mode,
                "filter_tool": filter_tool,
                "drop_ratio": drop_ratio,
                "forced_first_tool_filter": force_tool_filter_first,
            },
        )

        # 6) 更新 filter 参数
        state["filter_mode"] = filter_mode
        state["filter_tool"] = filter_tool
        state["drop_ratio"] = drop_ratio
        state["filter_plan_reason"] = filter_plan_reason

        # 7) NDCG（监控）
        def tok2int(t: str) -> Optional[int]:
            m = re.match(r"item_(\d+)$", str(t))
            return int(m.group(1)) if m else None

        item_list = [tok2int(x) for x in before_cand]
        item_list = [x for x in item_list if x is not None]

        item_list_re = [tok2int(x) for x in state["final_recommendations"]]
        item_list_re = [x for x in item_list_re if x is not None]

        target = int(state["target"]) if state.get("target") not in (None, "") else None
        if target is None or not item_list or not item_list_re:
            ndcg_before, ndcg_after = 0.0, 0.0
        else:
            ndcg_before = ndcg(item_list, target, len(item_list))
            ndcg_after = ndcg(item_list_re, target, len(item_list_re))

        state["NDCG"] = 0.0 if ndcg_before == 0 else 1.0

        state["reflector_memory"].append(
            f"round={state['iteration_count']} mode={filter_mode} ndcg {ndcg_before}->{ndcg_after}"
        )

        state["iteration_count"] = state["iteration_count"] + 1
        return state

    def should_continue_or_finish(self, state: RecommendationState) -> str:
        print("Decides whether to continue iterating or finish")

        if float(state.get("NDCG", 0.0)) == 0.0:
            return "finish"

        need_filter = bool(state.get("need_filter", False))
        filter_round = int(state.get("filter_round", 0))
        max_filter_rounds = int(state.get("max_filter_rounds", 5))
        min_keep = int(state.get("min_keep", 5))
        cand_len = len(normalize_candidate_tokens(state.get("candidate_items", [])))

        if need_filter and (filter_round < max_filter_rounds) and (cand_len > min_keep):
            return "continue"

        judgment = str(state.get("reasoner_judgment", "valid"))
        confidence = float(state.get("confidence", 0.85))
        iteration_count = int(state.get("iteration_count", 0))
        max_iterations = int(state.get("max_iterations", 1))

        if ("invalid" in judgment) or (confidence < 0.85):
            if iteration_count < max_iterations:
                return "continue"
        return "finish"

    # ============================================================
    # Run
    # ============================================================
    def run(self, user_profile, user_id, target=None, max_iterations=0, train_step=0, train=False) -> List[Dict[str, Any]]:
        init: RecommendationState = {
            "tool_set": {"SASRec", "GRURec", "LightGCN", "STAMP", "GLINTRU", "KGAT"},
            "NDCG": 0.0,
            "reorder_conversation": {},
            "reasoner_memory": [],
            "reflector_memory": [],
            "train_step": int(train_step or 0),
            "train": bool(train),
            "confidence": 0.0,
            "model_choice": "",
            "user_profile": user_profile,
            "user_id": str(user_id),
            "candidate_items": [],
            "re_candidate_items": [],
            "target": str(target) if target is not None else "",
            "planner_explanation": "",
            "planner_intention": "",
            "reasoner_judgment": "",
            "reasoner_reasoning": "",
            "reflection_feedback": "",
            "final_recommendations": [],
            "final_explanation": "",
            "planner_summary": "",
            "iteration_count": 0,
            "max_iterations": int(max_iterations),
            "need_filter": False,
            "filter_reason": "",
            "filter_mode": "none",
            "filter_tool": "",
            "drop_ratio": 0.0,
            "filter_plan_reason": "",
            "filter_round": 0,
            "max_filter_rounds": int(getattr(self.args, "max_filter_rounds", 3)),
            "min_keep": int(getattr(self.args, "min_keep", 5)),
            "filter_log": {},
            "has_tool_filtered": False,  # [NEW] 是否已经做过 tool 过滤（强制首轮用）
        }

        result = self.workflow.invoke(init)

        final_tokens = normalize_candidate_tokens(result.get("final_recommendations", []))
        print(f"!!!!!!!!!!!!!!!!!!!!User {user_id} finished after {result['iteration_count']} iterations.!!!!!!!!!!!!!!!!!!")
        # print(f"Final recommended items for user {user_id}: {final_tokens}")
        enriched_results = []
        for tok in final_tokens:
            meta = self.item_meta.get(tok, {})
            item_obj = {
                "id": tok,
                "movie_title": meta.get("movie_title"),
                "release_year": meta.get("release_year"),
                "genre": meta.get("genre"),
            }
            enriched_results.append(item_obj)

        return enriched_results
