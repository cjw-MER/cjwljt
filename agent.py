# agent.py
import json
import re
import os
import math
from typing import Dict, List, Any
import torch
from langgraph.graph import StateGraph, END

from prompts import planner, reasoner, reflector, summary
from utils import load_json_file, safe_parse_json

# 外部工具函数：过滤
from utils import apply_tool_filter
from utils import find_most_similar_user_id, MemmapRanker, ToolDir, format_candidates_with_meta, normalize_candidate_tokens
from metric import ndcg

from state import RecommendationState
from tools import retrieval_topk_load, stdout_retrived_items_load
from api import ZHI, DEEPSEEK, Qwen

from datasets import Dataset



class RecommendationAgent:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

        # 固定丢弃率（不再从 reflector 读取）
        self.fixed_drop_ratio = float(getattr(args, "fixed_drop_ratio", 0.2))

        # memories
        self.reasoning_memory = load_json_file(args.reasoning_memory_file)
        self.tool_memory = load_json_file(args.tool_memory_file)
        self.save_memory = {}

        # id_map
        with open('/home/chengjiawei/Agent4Rec/data/yelp/token2id_item.json', 'r') as f:
            self.token2id_item = json.load(f)
            self.id2token_item = {v: k for k, v in self.token2id_item.items()}
        with open('/home/chengjiawei/Agent4Rec/data/yelp/token2id_user.json', 'r') as f:
            self.token2id_user = json.load(f)
            self.id2token_user = {v: k for k, v in self.token2id_user.items()}

        # item meta
        meta_path = getattr(args, "item_meta_file", "/home/chengjiawei/Agent4Rec/data/yelp/item_info.json")
        self.item_meta = load_json_file(meta_path) if os.path.exists(meta_path) else {}

        # apis
        self.api_planner = ZHI()
        self.api_reasoner = ZHI()
        self.api_reflector = ZHI()
        self.api_summary = ZHI()

        # tools
        self.mem_ranker = MemmapRanker()
        self.tool_registry = ToolDir.default(self.args.dataset)

        # workflow
        self.workflow = self._build_workflow()

        # log save
        self.work_log = {}
        self.current_ndcg = 0.0
        self.tool_match = {}

        #

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
        similar_memory = []
        if getattr(self.args, "planner_memory_use", False) and self.tool_memory:
            can_users = list(self.tool_memory.keys())
            similar_user_id = find_most_similar_user_id(
                file_path=self.args.user_emb_profile,
                user_id=user_id,
                candidate_users=can_users,
                id_map = self.id2token_user
            )
            for i in similar_user_id:
                temp_similar_memory = self.tool_memory.get(str(self.token2id_user[i[1]])) # yelp需要将用户token转化为内部数字id,并且还得转化为str形式的数字id
                similar_memory.append(temp_similar_memory)
        # print("看看planner的相似信息:",temp_similar_memory)
        planner_prompt = planner(user_profile, self.args.dataset, similar_memory) if similar_memory else planner(user_profile, self.args.dataset)

        messages = [{"role": "user", "content": planner_prompt}]
        print("看看planner的输出：", self.api_planner.chat(messages))
        data = safe_parse_json(self.api_planner.chat(messages))
        
        state["planner_intention"] = data.get("planner_intention", "")
        state["planner_explanation"] = data.get("planner_reasoning", "")
        state["planner_choice"] = data.get("tool_selected", "sasrec")
        state["user_pattern"] = data.get("user_pattern", "")
        return state

    def tool_node(self, state: RecommendationState) -> RecommendationState:
        user_id = state["user_id"]

        # 1) 第 0 轮：外部召回；后续轮次：沿用 candidate_items
        if int(state.get("iteration_count", 0)) == 0 and not state.get("candidate_items"):
            tool_set = state["tool_set"]

            if state.get("planner_choice") in tool_set:
                external_item_list = retrieval_topk_load(model_file=self.args.candidate_item, user_id=user_id)
                cand_raw = stdout_retrived_items_load(external_item_list)
                tokens_in = normalize_candidate_tokens(cand_raw)
                print("看看未排序的候选列表:",tokens_in)
                # 根据planner的工具调用进行初次过滤
                mem_dir = self.tool_registry.memmap_dir(state.get("planner_choice", "sasrec"))
                tokens_before_filter = list(tokens_in)
                tokens_in = apply_tool_filter(
                    mem_ranker=self.mem_ranker,
                    mem_dir=mem_dir,
                    user_id=str(user_id),
                    candidate_tokens=tokens_in,
                    drop_ratio=self.fixed_drop_ratio,
                    min_keep=self.args.min_keep,
                    drop_unknown=True,
                )

                if len(tokens_in) < self.args.min_keep and len(tokens_before_filter) >= self.args.min_keep:
                    seen = set(tokens_in)
                    for x in tokens_before_filter:
                        if x not in seen:
                            tokens_in.append(x)
                            seen.add(x)
                        if len(tokens_in) >= self.args.min_keep:
                            break
                print(tokens_in)
                item_list = [int(item_id[5:]) for item_id in tokens_in]
                target = int(state["target"])
                ndcg_10 = ndcg(item_list, target, len(item_list))
                print("看看初次过滤的效果:",ndcg_10),exit(0)
                state["candidate_items"] = tokens_in
                return state
            else:
                # 返回最初的候选列表
                external_item_list = retrieval_topk_load(model_file=self.args.candidate_item, user_id=user_id)
                cand_raw = stdout_retrived_items_load(external_item_list)
                tokens_in = normalize_candidate_tokens(cand_raw)
                state["candidate_items"] = tokens_in
                return state
        else:
            # 大于 0 轮：处理上一轮的候选列表
            tokens_in = normalize_candidate_tokens(state.get("candidate_items", []))
            tokens_before_filter = list(tokens_in)

            # 2) 工具过滤：用 iteration_count 控制过滤轮次
            max_filter_rounds = self.args.max_filter_rounds
            filter_mode = str(state.get("filter_mode", "tool")).lower()
            iteration_count = int(state.get("iteration_count", 0))

            # if filter_mode == "tool": # 判断过滤条件
            mem_dir = self.tool_registry.memmap_dir(state.get("filter_tool", "sasrec"))

            tokens_in = apply_tool_filter(
                mem_ranker=self.mem_ranker,
                mem_dir=mem_dir,
                user_id=str(user_id),
                candidate_tokens=tokens_in,
                drop_ratio=self.fixed_drop_ratio,
                min_keep=self.args.min_keep,
                drop_unknown=True,
            )

            if len(tokens_in) < self.args.min_keep and len(tokens_before_filter) >= self.args.min_keep:
                seen = set(tokens_in)
                for x in tokens_before_filter:
                    if x not in seen:
                        tokens_in.append(x)
                        seen.add(x)
                    if len(tokens_in) >= self.args.min_keep:
                        break
            if self.args.dataset == 'ml-1m':
                item_list = [int(item_id[5:]) for item_id in tokens_in]
                target = int(state["target"])
                ndcg_10 = ndcg(item_list, target, len(item_list))
            else:
                item_list = [int(self.token2id_item[item_id[5:]]) for item_id in tokens_in]
                target = int(self.token2id_item[state["target"]])
                ndcg_10 = ndcg(item_list, target, len(item_list))


            print("看看使用工具过滤的效果：",ndcg_10)
            self.current_ndcg = ndcg_10
            state["candidate_items"] = normalize_candidate_tokens(tokens_in)

            # log save
            iteration = {}
            iteration[state["iteration_count"]] = f""" planner's tool_selecet:{state["planner_choice"]}, reasoner's reasoning:{state["reasoner_reasoning"]}, reflector's tool_select:{state["filter_tool"]}, NDCG:{ndcg_10}, filtered_list:{normalize_candidate_tokens(tokens_in)}"""
            state["iteration_ls"].append(iteration)

            return state

    def reasoner_agent(self, state: RecommendationState) -> RecommendationState:
        user_id = state["user_id"]
    
        candidate_items = state["candidate_items"]
        formatted_candidates = format_candidates_with_meta(candidate_items, self.item_meta, self.args.dataset)
        input_memory = state.get("reasoner_memory", [])
        planner_intention = state.get("planner_intention", "")
        planner_reasoning = state.get("planner_explanation", "")
        user_behavior_pattern = state.get("user_pattern", "")

        similar_memory = []
        if getattr(self.args, "reasoner_memory_use", False) and self.tool_memory:
            can_users = list(self.tool_memory.keys())
            similar_user_id = find_most_similar_user_id(
                file_path=self.args.user_emb_profile,
                user_id=user_id,
                candidate_users=can_users,
                id_map = self.id2token_user
            )
            for i in similar_user_id:
                temp_similar_memory = self.tool_memory.get(str(self.token2id_user[i[1]]))
                similar_memory.append(temp_similar_memory)
        # print("看看reasoner的相似信息:",temp_similar_memory)
        prompt = reasoner(
            formatted_candidates,
            planner_intention,
            planner_reasoning,
            user_behavior_pattern,
            similar_memory,
            self.args.min_keep
        )


        messages = [{"role": "user", "content": prompt}]
        print("看看reasoner的输出：", self.api_reasoner.chat(messages))
        data = safe_parse_json(self.api_reasoner.chat(messages))

        state["reasoner_judgment"] = data.get("judgment", "valid")
        state["confidence"] = float(data.get("confidence", 0.0))
        state["reasoner_reasoning"] = data.get("reasoner_reasoning", "")
        state["need_filter"] = bool(data.get("need_filter", False))
        state["tool_characteristics"] = data.get("tool_characteristics","")


        # 将reasoner的记忆保存下来
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
        user_id = state["user_id"]

        candidate_items_before = list(state["candidate_items"])
        tool_character = state["tool_characteristics"]
        reasoenr_reaosning = state.get("reasoner_reasoning", "")
        formatted_candidates = format_candidates_with_meta(candidate_items_before, self.item_meta, self.args.dataset)

        similar_memory = []
        if getattr(self.args, "reflector_memory_use", False) and self.tool_memory:
            can_users = list(self.tool_memory.keys())
            similar_user_id = find_most_similar_user_id(
                file_path=self.args.user_emb_profile,
                user_id=user_id,
                candidate_users=can_users,
                id_map = self.id2token_user
            )
            for i in similar_user_id:
                temp_similar_memory = self.tool_memory.get(str(self.token2id_user[i[1]]))
                similar_memory.append(temp_similar_memory)
        # print("看看reflector的相似信息:",temp_similar_memory)
        prompt = reflector(tool_character, reasoenr_reaosning, memory=similar_memory)

        messages = [{"role": "user", "content": prompt}]

        # api 调用
        #print("看看reflector的输出：", self.api_reflector.chat(messages))
        data = safe_parse_json(self.api_reflector.chat(messages))

        # text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        # with torch.inference_mode():
        #     outputs = self.model.generate(
        #         **inputs,
        #         max_new_tokens=512,
        #         temperature=0.0001
        #     )
        # prompt_len = inputs["input_ids"].shape[1]
        # gen_ids = outputs[0, prompt_len:]
        # response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # data = safe_parse_json(response)
        # print("看看reflector的输出：",response)

        # # 检查显存使用情况
        # del inputs, outputs, text
        # torch.cuda.empty_cache()

        
        state["filter_mode"] = str(data.get("filter_mode", "tool")).lower()
        state["filter_tool"] = self.tool_registry.normalize(data.get("filter_tool") or "glintru")
        state["filter_plan_reason"] = str(data.get("filter_plan_reason", ""))
        state["candidate_items"] = normalize_candidate_tokens(candidate_items_before)
        state["iteration_count"] = int(state.get("iteration_count", 0)) + 1

        return state

    def should_continue_or_finish(self, state: RecommendationState) -> str:

        need_filter = state.get("need_filter", False)
        iteration_count = state.get("iteration_count", 0)
        cand_len = len(normalize_candidate_tokens(state.get("candidate_items", [])))
        judgment = state.get("reasoner_judgment", "valid")
        confidence = state.get("confidence", 1.0)

        if need_filter and (iteration_count < self.args.max_filter_rounds) and (cand_len > self.args.min_keep):
            return "continue"

        if ("invalid" in judgment or confidence < 0.85) and iteration_count < self.args.max_filter_rounds:
            return "continue"

        return "finish"

    def run(self, user_profile, user_id, target=None, max_iterations=0, train_step=0, train=False) -> List[Dict[str, Any]]:
        init: RecommendationState = {
            "tool_set": {"sasrec", "grurec", "lightgcn", "glintru", "kgat", "stamp", "fmlprec", "difsr"},
            "user_profile": user_profile,
            "user_id": str(user_id),
            "candidate_items": [],
            "target": str(target) if target is not None else "",

            "user_pattern": "",
            "tool_characteristics": "",
            "planner_explanation": "",
            "planner_intention": "",
            "planner_choice": "",

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
            "best_tool": "",
            "best_ndcg": "",

            "reasoner_memory": [],
            "iteration_ls": [],

            "train_step": int(train_step or 0),
            "train": bool(train),
        }

        result = self.workflow.invoke(init)

        out_tokens = normalize_candidate_tokens(result.get("candidate_items", []))

        if self.args.dataset == 'ml-1m':
            return out_tokens
        
        elif self.args.dataset == 'yelp':
            final_tokens = [ids[5:] for ids in out_tokens]
            return final_tokens
        else:
            return out_tokens
