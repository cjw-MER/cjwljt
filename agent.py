import os
import math
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Tuple

from langgraph.graph import StateGraph, END

from prompts import planner, reasoner, reflector
from utils import load_json_file, safe_parse_json
from metric import ndcg
from state import RecommendationState
from tools import retrieval_topk_load, stdout_retrived_items_load
from api import ZHI

from rec_utils.tokens import normalize_candidate_tokens, token_to_int
from rec_utils.memmap_ranker import MemmapRanker
from rec_utils.logging_utils import ConversationLogger, ToolFilterLogger, RunMetaManager
from rec_utils.rec_helpers import ToolRegistry, format_candidates_with_meta, SimilarMemoryRetriever


class RecommendationAgent:
    """
    仅保留 state["candidate_items"] 作为候选集单一真源。
    去掉：
    - state["re_candidate_items"]
    - state["final_recommendations"]
    """

    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

        self.run_dir = getattr(args, "run_dir", None)
        if not self.run_dir:
            raise ValueError("args.run_dir is required (created by driver).")
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_id = os.path.basename(os.path.abspath(self.run_dir))

        self.git_commit = None
        try:
            self.git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            self.git_commit = None

        # logging dirs
        self.enable_tool_log = bool(getattr(args, "enable_tool_log", True))
        self.enable_conversation_log = bool(getattr(args, "enable_conversation_log", True))

        self.tool_log_dir = os.path.join(self.run_dir, "tool_calls_logs")
        self.conv_log_dir = os.path.join(self.run_dir, "conversations")
        os.makedirs(self.tool_log_dir, exist_ok=True)
        os.makedirs(self.conv_log_dir, exist_ok=True)

        # run meta
        self.run_meta = RunMetaManager(
            run_dir=self.run_dir,
            run_id=self.run_id,
            git_commit=self.git_commit,
            args=self.args,
        )
        self.run_meta.write(started_at=datetime.now())

        # loggers
        self.conv_logger = ConversationLogger(
            conv_log_dir=self.conv_log_dir,
            run_id=self.run_id,
            enabled=self.enable_conversation_log,
            max_chars=int(getattr(self.args, "conv_max_chars", 100000)),
        )
        self.tool_logger = ToolFilterLogger(
            tool_log_dir=self.tool_log_dir,
            run_id=self.run_id,
            enabled=self.enable_tool_log,
        )

        # memories
        self.tool_memory = load_json_file(args.tool_memory_file)
        self.reasoning_memory = load_json_file(args.reasoning_memory_file)
        self.sim_mem = SimilarMemoryRetriever(self.reasoning_memory)

        # item meta
        meta_path = getattr(args, "item_meta_file", "/home/liujuntao/Agent4Rec/data/ml-1m/ml-1m.item.json")
        if os.path.exists(meta_path):
            self.item_meta = load_json_file(meta_path)
        else:
            print(f"Warning: Item meta file not found at {meta_path}. Reasoning will lack metadata.")
            self.item_meta = {}

        # apis
        self.api_planner = ZHI()
        self.api_reasoner = ZHI()
        self.api_reflector = ZHI()

        # tools
        self.mem_ranker = MemmapRanker()
        self.tool_registry = ToolRegistry.default_ml1m()

        # workflow
        self.workflow = self._build_workflow()

    def finalize_run(self, total_users: int, ok_users: int, skipped_users: int, elapsed_seconds: float):
        self.run_meta.write(finished_at=datetime.now())
        self.run_meta.update_tool_filter_stats(
            tool_log_dir=self.tool_log_dir,
            total_users=total_users,
            ok_users=ok_users,
            skipped_users=skipped_users,
            elapsed_seconds=elapsed_seconds,
        )

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

    # ---------------- nodes ----------------
    def planner_agent(self, state: RecommendationState) -> RecommendationState:
        user_profile = state["user_profile"]
        user_id = state["user_id"]

        if getattr(self.args, "planner_memory_use", False):
            similar_memory = self.sim_mem.get(self.args.user_emb_profile, user_id)
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

        self.conv_logger.log_turn(state, stage="planner", messages=messages, response=response, parsed=data)

        state["planner_intention"] = data.get("planner_intention", "")
        state["planner_explanation"] = data.get("planner_reasoning", "")
        state["model_choice"] = data.get("tool_selected", "SASRec")
        state["planner_summary"] = ""
        return state

    def tool_node(self, state: RecommendationState) -> RecommendationState:
        user_id = state["user_id"]

        # 1) 拿到候选集（第 0 轮：外部 retrieval；后续轮次：沿用 state["candidate_items"]）
        if state["iteration_count"] == 0:
            tool_set = state.get("tool_set", {"SASRec", "GRURec", "LightGCN"})
            if isinstance(tool_set, str):
                tool_set = set([x.strip() for x in tool_set.split(",")])

            if state.get("model_choice") in tool_set:
                external_item_list = retrieval_topk_load(
                    model_file="/home/liujuntao/test_topk_prediction_user2000_random_100.json",
                    user_id=user_id,
                )
                cand_raw = stdout_retrived_items_load(external_item_list)
            else:
                cand_raw = state.get("candidate_items", [])
        else:
            cand_raw = state.get("candidate_items", [])

        tokens_in = normalize_candidate_tokens(cand_raw)

        # 2) 在 tool_node 执行工具过滤/重排
        min_keep = int(state.get("min_keep", 5))
        max_filter_rounds = int(state.get("max_filter_rounds", 5))
        filter_round = int(state.get("filter_round", 0))

        filter_mode = str(state.get("filter_mode", "none")).lower()
        drop_ratio = float(state.get("drop_ratio", 0.0))

        has_tool_filtered = bool(state.get("has_tool_filtered", False))
        force_tool_filter_first = not has_tool_filtered  # 第一次强制工具重排，确保候选集从一开始就是“过滤过”的

        # reflector 可能还没给 filter_tool；这里给个兜底
        filter_tool = self.tool_registry.normalize(state.get("filter_tool") or "glintru")
        print("tool_node: filter_tool=", filter_tool)
        def _apply_tool_filter(tokens: List[str], dr: float) -> List[str]:
            mem_dir_local = self.tool_registry.memmap_dir(filter_tool)

            self.tool_logger.log_filter_call(
                state=state,
                filter_tool=filter_tool,
                mem_dir=mem_dir_local,
                drop_ratio=dr,
                min_keep=min_keep,
                tokens_in_len=len(tokens),
            )

            out_tokens, _ = self.mem_ranker.rank(
                mem_dir=mem_dir_local,
                user_id=str(user_id),
                candidate_tokens=tokens,
                drop_ratio=dr,
                drop_unknown=True,
                min_keep=min_keep,
            )

            # 保底：别因为 user row 不存在导致候选集被清空
            return out_tokens if out_tokens else tokens

        did_tool_filter = False
        if tokens_in:
            if force_tool_filter_first:
                tokens_in = _apply_tool_filter(tokens_in, 0.0)
                state["has_tool_filtered"] = True
                did_tool_filter = True

                if filter_round < max_filter_rounds and len(tokens_in) > min_keep:
                    state["filter_round"] = filter_round + 1

            elif filter_mode == "tool":
                if filter_round < max_filter_rounds and len(tokens_in) > min_keep:
                    tokens_in = _apply_tool_filter(tokens_in, drop_ratio)
                    state["has_tool_filtered"] = True
                    did_tool_filter = True
                    state["filter_round"] = filter_round + 1

        if did_tool_filter:
            used = state.get("used_filter_tools", [])
            if not isinstance(used, list):
                used = []
            used.append(filter_tool)
            state["used_filter_tools"] = used
        # print("tool_node:before. ",state["candidate_items"])
        # 3) 写回：只保留 candidate_items
        state["candidate_items"] = normalize_candidate_tokens(tokens_in)
        # print("tool_node:after. ",state["candidate_items"])
        return state

    def reasoner_agent(self, state: RecommendationState) -> RecommendationState:
        state["candidate_items"] = normalize_candidate_tokens(state.get("candidate_items", []))
        candidate_items = state["candidate_items"]
        user_id = state["user_id"]

        formatted_candidates = format_candidates_with_meta(candidate_items, self.item_meta)

        input_memory = state.get("reasoner_memory", [])
        planner_intention = state["planner_intention"]
        planner_reasoning = state["planner_explanation"]

        similar_memory = None
        if getattr(self.args, "reasoner_memory_use", False):
            similar_memory = self.sim_mem.get(self.args.user_emb_profile, user_id)

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

        self.conv_logger.log_turn(state, stage="reasoner", messages=messages, response=response, parsed=data)

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
        state["candidate_items"] = normalize_candidate_tokens(state.get("candidate_items", []))
        candidate_items_before = list(state["candidate_items"])
        print(f"Reflector: candidate_items_before len={len(candidate_items_before)}")
        user_id = state["user_id"]

        formatted_candidates = format_candidates_with_meta(candidate_items_before, self.item_meta)
        print(f"Reflector: formatted_candidates len={len(formatted_candidates)}")
        similar_memory = (
            self.sim_mem.get(self.args.user_emb_profile, user_id)
            if getattr(self.args, "reflector_memory_use", False)
            else None
        )

        used_filter_tools = state.get("used_filter_tools", [])
        prompt = reflector(
            state["reasoner_reasoning"],
            formatted_candidates,
            formatted_candidates,  # reflector 目前没用上 re_candidate_items，先传一样的
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
        "filter_tool": "gru4rec" or "lightgcn" or "stamp" or "glintru" or "kgat",
        "drop_ratio": 0.0-1.0,
        "filter_plan_reason": "why this mode/tool and drop_ratio (max 40 words)",
        "re_ranked_candidate_items": []
        }
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        response = self.api_reflector.chat(messages)
        data = safe_parse_json(response)

        self.conv_logger.log_turn(state, stage="reflector", messages=messages, response=response, parsed=data)

        # 1) reflector 只写计划，不执行 tool filter（tool filter 在 tool_node）
        state["filter_mode"] = str(data.get("filter_mode", "none")).lower()
        print("Reflector: filter_mode=", state["filter_mode"])
        state["filter_tool"] = self.tool_registry.normalize(data.get("filter_tool"))
        state["drop_ratio"] = float(data.get("drop_ratio", 0.0))
        print("Reflector: drop_ratio=", state["drop_ratio"])
        state["filter_plan_reason"] = str(data.get("filter_plan_reason", ""))

        filter_mode = state["filter_mode"]
        drop_ratio = state["drop_ratio"]
        min_keep = int(state.get("min_keep", 5))

        # 2) llm 模式：只改 candidate_items
        if filter_mode == "llm":
            re_ranked = normalize_candidate_tokens(data.get("re_ranked_candidate_items", []))
            base_for_rerank = candidate_items_before

            if not re_ranked:
                merged = base_for_rerank
            else:
                cand_set = set(base_for_rerank)
                valid_ranked = [x for x in re_ranked if x in cand_set]
                missing = [x for x in base_for_rerank if x not in set(valid_ranked)]
                merged = valid_ranked + missing

            keep_n = max(min_keep, int(math.floor(len(merged) * (1.0 - float(drop_ratio)))))
            keep_n = min(keep_n, len(merged))
            state["candidate_items"] = normalize_candidate_tokens(merged[:keep_n])
        else:
            state["candidate_items"] = normalize_candidate_tokens(candidate_items_before)

        # 3) NDCG monitor（保持你原“只看 before 是否为 0”的行为）
        item_list = [token_to_int(x) for x in candidate_items_before]
        item_list = [x for x in item_list if x is not None]
        print("Reflector: item_list before NDCG calculation =", item_list)
        target = int(state["target"]) if state.get("target") not in (None, "") else None
        print("Reflector: target =", target)
        if target is None or not item_list:
            ndcg_before = 0.0
        else:
            ndcg_before = ndcg(item_list, target, len(item_list))
        print(f"Reflector: NDCG before={ndcg_before}")
        state["NDCG"] = 0.0 if ndcg_before == 0 else 1.0
        state["iteration_count"] = state["iteration_count"] + 1
        return state

    def should_continue_or_finish(self, state: RecommendationState) -> str:
        
        if float(state.get("NDCG", 0.0)) == 0.0:
            print("NDCG=0.0, finish.")
            return "finish"

        need_filter = bool(state.get("need_filter", False))
        print("need_filter =", need_filter)
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
            "target": str(target) if target is not None else "",
            "planner_explanation": "",
            "planner_intention": "",
            "reasoner_judgment": "",
            "reasoner_reasoning": "",
            "reflection_feedback": "",
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
            "has_tool_filtered": False,
            "used_filter_tools": [],
        }

        result = self.workflow.invoke(init)

        # 最终输出：直接从 candidate_items 取
        final_tokens = normalize_candidate_tokens(result.get("candidate_items", []))
        print(f"Final recommended items count: {len(final_tokens)}")
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
