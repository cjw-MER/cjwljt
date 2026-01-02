# agent.py
import os
import math
from typing import Dict, List, Any

from langgraph.graph import StateGraph, END

from prompts import planner, reasoner, reflector
from utils import load_json_file, safe_parse_json

# 外部工具函数：过滤
from utils import apply_tool_filter
from utils import find_most_similar_user_id

from state import RecommendationState
from tools import retrieval_topk_load, stdout_retrived_items_load
from api import ZHI

from rec_utils.tokens import normalize_candidate_tokens
from rec_utils.memmap_ranker import MemmapRanker
from rec_utils.rec_helpers import ToolRegistry, format_candidates_with_meta


class RecommendationAgent:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

        # 固定丢弃率（不再从 reflector 读取）
        self.fixed_drop_ratio = float(getattr(args, "fixed_drop_ratio", 0.2))

        # memories
        self.reasoning_memory = load_json_file(args.reasoning_memory_file)

        # item meta
        meta_path = getattr(args, "item_meta_file", "/home/liujuntao/Agent4Rec/data/ml-1m/ml-1m.item.json")
        self.item_meta = load_json_file(meta_path) if os.path.exists(meta_path) else {}

        # apis
        self.api_planner = ZHI()
        self.api_reasoner = ZHI()
        self.api_reflector = ZHI()

        # tools
        self.mem_ranker = MemmapRanker()
        self.tool_registry = ToolRegistry.default_ml1m()

        # workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(RecommendationState)
        workflow.set_entry_point("planner")

        workflow.add_node("planner", self.planner_agent)
        workflow.add_node("tool_node", self.tool_node)
        workflow.add_node("reasoner", self.reasoner_agent)
        workflow.add_node("reflector", self.reflector_agent)

        workflow.add_edge("planner", "tool_node")
        workflow.add_edge("tool_node", "reasoner")
        workflow.add_edge("reasoner", "reflector")
        workflow.add_conditional_edges(
            "reflector",
            self.should_continue_or_finish,
            {"continue": "tool_node", "finish": END},
        )
        return workflow.compile()

    # ---------------- nodes ----------------
    def planner_agent(self, state: RecommendationState) -> RecommendationState:
        user_profile = state["user_profile"]
        user_id = state["user_id"]

        similar_memory = None
        if getattr(self.args, "planner_memory_use", False) and self.reasoning_memory:
            can_users = list(self.reasoning_memory.keys())
            similar_user_id = find_most_similar_user_id(
                file_path=self.args.user_emb_profile,
                user_id=str(user_id),
                candidate_users=can_users,
            )
            similar_memory = self.reasoning_memory.get(similar_user_id)

        planner_prompt = planner(user_profile, similar_memory) if similar_memory else planner(user_profile)

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
        data = safe_parse_json(self.api_planner.chat(messages))

        state["planner_intention"] = data.get("planner_intention", "")
        state["planner_explanation"] = data.get("planner_reasoning", "")
        state["model_choice"] = data.get("tool_selected", "SASRec")
        return state

    def tool_node(self, state: RecommendationState) -> RecommendationState:
        user_id = state["user_id"]

        # 1) 第 0 轮：外部召回；后续轮次：沿用 candidate_items
        if int(state.get("iteration_count", 0)) == 0 and not state.get("candidate_items"):
            tool_set = state.get("tool_set", {"SASRec", "GRURec", "LightGCN"})
            if isinstance(tool_set, str):
                tool_set = set([x.strip() for x in tool_set.split(",") if x.strip()])

            if state.get("model_choice") in tool_set:
                external_item_list = retrieval_topk_load(
                    model_file="/home/liujuntao/test_topk_prediction_user2000_random_100.json",
                    user_id=user_id,
                )
                cand_raw = stdout_retrived_items_load(external_item_list)
            else:
                cand_raw = state.get("candidate_items", [])
            tokens_in = normalize_candidate_tokens(cand_raw)
        else:
            tokens_in = normalize_candidate_tokens(state.get("candidate_items", []))

        if not tokens_in:
            state["candidate_items"] = []
            return state

        tokens_before_filter = list(tokens_in)

        # 2) 工具过滤/重排：用 iteration_count 控制过滤轮次
        min_keep = int(state.get("min_keep", getattr(self.args, "min_keep", 5)))
        max_filter_rounds = int(state.get("max_filter_rounds", getattr(self.args, "max_filter_rounds", 3)))
        filter_mode = str(state.get("filter_mode", "tool")).lower()
        iteration_count = int(state.get("iteration_count", 0))

        if (
            filter_mode == "tool"
            and iteration_count < max_filter_rounds
            and len(tokens_in) > min_keep
        ):
            filter_tool = self.tool_registry.normalize(state.get("filter_tool") or "glintru")
            mem_dir = self.tool_registry.memmap_dir(filter_tool)

            tokens_in = apply_tool_filter(
                mem_ranker=self.mem_ranker,
                mem_dir=mem_dir,
                user_id=str(user_id),
                candidate_tokens=tokens_in,
                drop_ratio=self.fixed_drop_ratio,
                min_keep=min_keep,
                drop_unknown=True,
            )

            used = state.get("used_filter_tools", [])
            if not isinstance(used, list):
                used = []
            used.append(filter_tool)
            state["used_filter_tools"] = used

        if len(tokens_in) < min_keep and len(tokens_before_filter) >= min_keep:
            seen = set(tokens_in)
            for x in tokens_before_filter:
                if x not in seen:
                    tokens_in.append(x)
                    seen.add(x)
                if len(tokens_in) >= min_keep:
                    break

        state["candidate_items"] = normalize_candidate_tokens(tokens_in)
        return state

    def reasoner_agent(self, state: RecommendationState) -> RecommendationState:
        state["candidate_items"] = normalize_candidate_tokens(state.get("candidate_items", []))
        candidate_items = state["candidate_items"]

        formatted_candidates = format_candidates_with_meta(candidate_items, self.item_meta)

        input_memory = state.get("reasoner_memory", [])
        planner_intention = state.get("planner_intention", "")
        planner_reasoning = state.get("planner_explanation", "")

        similar_memory = None
        if getattr(self.args, "reasoner_memory_use", False) and self.reasoning_memory:
            can_users = list(self.reasoning_memory.keys())
            similar_user_id = find_most_similar_user_id(
                file_path=self.args.user_emb_profile,
                user_id=str(state["user_id"]),
                candidate_users=can_users,
            )
            similar_memory = self.reasoning_memory.get(similar_user_id)

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
        data = safe_parse_json(self.api_reasoner.chat(messages))

        state["reasoner_judgment"] = data.get("judgment", "valid")
        state["confidence"] = float(data.get("confidence", 0.0))
        state["reasoner_reasoning"] = data.get("reasoner_reasoning", "")
        state["need_filter"] = bool(data.get("need_filter", False))
        state["filter_reason"] = str(data.get("filter_reason", ""))

        mem = state.get("reasoner_memory", [])
        if not isinstance(mem, list):
            mem = []
        mem.append(
            f"round={state.get('iteration_count', 0)} judgment={state['reasoner_judgment']} "
            f"conf={state['confidence']} reason={state['reasoner_reasoning']}"
        )
        state["reasoner_memory"] = mem
        return state

    def reflector_agent(self, state: RecommendationState) -> RecommendationState:
        state["candidate_items"] = normalize_candidate_tokens(state.get("candidate_items", []))
        candidate_items_before = list(state["candidate_items"])

        formatted_candidates = format_candidates_with_meta(candidate_items_before, self.item_meta)

        similar_memory = None
        if getattr(self.args, "reflector_memory_use", False) and self.reasoning_memory:
            can_users = list(self.reasoning_memory.keys())
            similar_user_id = find_most_similar_user_id(
                file_path=self.args.user_emb_profile,
                user_id=str(state["user_id"]),
                candidate_users=can_users,
            )
            similar_memory = self.reasoning_memory.get(similar_user_id)

        used_filter_tools = state.get("used_filter_tools", [])
        prompt = reflector(
            state.get("reasoner_reasoning", ""),
            formatted_candidates,
            formatted_candidates,
            state["user_profile"],
            memory=similar_memory,
            need_filter=state.get("need_filter", False),
            filter_reason=state.get("filter_reason", ""),
            min_keep=int(state.get("min_keep", getattr(self.args, "min_keep", 5))),
            used_filter_tools=used_filter_tools,
        )

        system_prompt = """Output a JSON object strictly following this format:
        {
          "filter_mode": "tool" or "llm" or "none",
          "filter_tool": "sasrec" or "gru4rec" or "stamp" or "glintru",
          "filter_plan_reason": "max 40 words",
          "re_ranked_candidate_items": []
        }
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        data = safe_parse_json(self.api_reflector.chat(messages))

        state["filter_mode"] = str(data.get("filter_mode", "tool")).lower()
        state["filter_tool"] = self.tool_registry.normalize(data.get("filter_tool") or "glintru")
        state["filter_plan_reason"] = str(data.get("filter_plan_reason", ""))

        # llm 模式：重排 + 固定丢弃率裁剪
        if state["filter_mode"] == "llm":
            min_keep = int(state.get("min_keep", getattr(self.args, "min_keep", 5)))
            re_ranked = normalize_candidate_tokens(data.get("re_ranked_candidate_items", []))

            if re_ranked:
                base_set = set(candidate_items_before)
                valid_ranked = [x for x in re_ranked if x in base_set]
                merged = valid_ranked
            else:
                merged = candidate_items_before

            keep_n = max(min_keep, int(math.floor(len(merged) * (1.0 - self.fixed_drop_ratio))))
            keep_n = min(keep_n, len(merged))
            state["candidate_items"] = normalize_candidate_tokens(merged[:keep_n])
        else:
            state["candidate_items"] = normalize_candidate_tokens(candidate_items_before)

        state["iteration_count"] = int(state.get("iteration_count", 0)) + 1
        return state

    def should_continue_or_finish(self, state: RecommendationState) -> str:
        need_filter = bool(state.get("need_filter", False))

        iteration_count = int(state.get("iteration_count", 0))
        max_filter_rounds = int(state.get("max_filter_rounds", getattr(self.args, "max_filter_rounds", 3)))

        min_keep = int(state.get("min_keep", getattr(self.args, "min_keep", 5)))
        cand_len = len(normalize_candidate_tokens(state.get("candidate_items", [])))

        if need_filter and (iteration_count < max_filter_rounds) and (cand_len > min_keep):
            return "continue"

        judgment = str(state.get("reasoner_judgment", "valid"))
        confidence = float(state.get("confidence", 1.0))
        max_iterations = int(state.get("max_iterations", 0))

        if ("invalid" in judgment or confidence < 0.85) and iteration_count < max_iterations:
            return "continue"

        return "finish"

    def run(self, user_profile, user_id, target=None, max_iterations=0, train_step=0, train=False) -> List[Dict[str, Any]]:
        init: RecommendationState = {
            "tool_set": {"SASRec", "GRURec", "STAMP", "GLINTRU"},
            "user_profile": user_profile,
            "user_id": str(user_id),
            "candidate_items": [],
            "target": str(target) if target is not None else "",

            "planner_explanation": "",
            "planner_intention": "",
            "model_choice": "",

            "reasoner_judgment": "",
            "reasoner_reasoning": "",
            "confidence": 0.0,
            "need_filter": False,
            "filter_reason": "",

            "filter_mode": "tool",
            "filter_tool": "glintru",
            "filter_plan_reason": "",

            "iteration_count": 0,
            "max_iterations": int(max_iterations),

            # 用 iteration_count 控制过滤轮次
            "max_filter_rounds": int(getattr(self.args, "max_filter_rounds", 3)),
            "min_keep": int(getattr(self.args, "min_keep", 5)),

            "has_tool_filtered": False,
            "used_filter_tools": [],

            "reasoner_memory": [],
            "reflector_memory": [],

            "train_step": int(train_step or 0),
            "train": bool(train),
        }

        result = self.workflow.invoke(init)

        final_tokens = normalize_candidate_tokens(result.get("candidate_items", []))
        enriched_results = []
        for tok in final_tokens:
            meta = self.item_meta.get(tok, {})
            enriched_results.append(
                {
                    "id": tok,
                    "movie_title": meta.get("movie_title"),
                    "release_year": meta.get("release_year"),
                    "genre": meta.get("genre"),
                }
            )
        return enriched_results
