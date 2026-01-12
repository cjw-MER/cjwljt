# agent.py
import re
import os
import math
from typing import Dict, List, Any
import torch
from langgraph.graph import StateGraph, END

from prompts import planner, reasoner, reflector
from utils import load_json_file, safe_parse_json

# 外部工具函数：过滤
from utils import apply_tool_filter
from utils import find_most_similar_user_id
from metric import ndcg

from state import RecommendationState
from tools import retrieval_topk_load, stdout_retrived_items_load
from api import ZHI, DEEPSEEK, Qwen

from rec_utils.tokens import normalize_candidate_tokens
from rec_utils.memmap_ranker import MemmapRanker
from rec_utils.rec_helpers import ToolRegistry, format_candidates_with_meta

from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig




def tool_accuracy_reward(completions, **kwargs):
    """
    completions: 展平后的 completion 列表（长度 = batch_size * num_generations）
    best_tool/user_id: 会按同样的展平策略对齐（通常是每个 prompt 复制 num_generations 次）
    """
    # with open('/home/chengjiawei/Agent4Rec/训练版本/ml-1m_user_reward.json', 'r') as f:
    #     reward_table = json.load(f)
    print("看看refletor在GRPOTrainer上的响应内容：")
    print(completions[0])
    rewards = []
    for comp in completions:

        # 1) 主奖励：工具选对
        r = 1.0 


        rewards.append(float(r))

    return rewards

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

        # item meta
        meta_path = getattr(args, "item_meta_file", "/home/chengjiawei/Agent4Rec/data/ml-1m/ml-1m.item.json")
        self.item_meta = load_json_file(meta_path) if os.path.exists(meta_path) else {}

        # apis
        self.api_planner = ZHI()
        self.api_reasoner = ZHI()
        self.api_reflector = ZHI()
        # self.api_summary = Qwen()

        # tools
        self.mem_ranker = MemmapRanker()
        self.tool_registry = ToolRegistry.default_ml1m()

        # workflow
        self.workflow = self._build_workflow()

        # log save
        self.work_log = {}
        self.current_ndcg = 0.0

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
                user_id=str(user_id),
                candidate_users=can_users,
            )

            for i in similar_user_id:
                temp_similar_memory = self.tool_memory.get(i[1])
                similar_memory.append(temp_similar_memory)

                # # 提取相似用户所使用的tool
                # m = re.search(r"best_tool\s*:\s*([A-Za-z0-9_]+)", similar_memory)
                # best_tool = m.group(1) if m else None
                # users_tool.append(best_tool)
        planner_prompt = planner(user_profile, similar_memory) if similar_memory else planner(user_profile)

        system_planner_prompt = """Output a JSON object strictly following this format:
        {
          "tool_selected": "sasrec" | "grurec" | "lightgcn" | "stamp" | "glintru" | "kgat",
          "planner_intention": "1-2 movie genres",
          "user_pattern": "Concise description of user behavior pattern",
          "planner_reasoning": "Brief explanation. Max 60 words."
        }
        """
        messages = [
            {"role": "system", "content": system_planner_prompt},
            {"role": "user", "content": planner_prompt},
        ]
        print("看看planner的输出：", self.api_planner.chat(messages))
        data = safe_parse_json(self.api_planner.chat(messages))


        # 本地模型调用
        # text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        # output = self.model.generate(text, sampling_params)
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

        
        #print("看看planner的输出：",data)
        state["planner_intention"] = data.get("planner_intention", "")
        state["planner_explanation"] = data.get("planner_reasoning", "")
        state["planner_choice"] = data.get("tool_selected", "SASRec")
        state["user_pattern"] = data.get("user_pattern", "")
        return state

    def tool_node(self, state: RecommendationState) -> RecommendationState:
        user_id = state["user_id"]

        # 1) 第 0 轮：外部召回；后续轮次：沿用 candidate_items
        if int(state.get("iteration_count", 0)) == 0 and not state.get("candidate_items"):
            tool_set = state.get("tool_set", {"sasrec", "grurec", "lightgcn", "glintru", "kgat", "stamp"})
            if isinstance(tool_set, str):
                tool_set = set([x.strip() for x in tool_set.split(",") if x.strip()])

            if state.get("planner_choice") in tool_set:
                external_item_list = retrieval_topk_load(
                    model_file=self.args.candidate_item,
                    user_id=user_id,
                )
                cand_raw = stdout_retrived_items_load(external_item_list)
                tokens_in = normalize_candidate_tokens(cand_raw)
                item_list = [int(item_id[5:]) for item_id in tokens_in]
                target = int(state["target"])
                ndcg_10 = ndcg(item_list, target, len(item_list))

                # 下面的代码主要用于计算当前user的最优tool选择
                
                tool_1 = apply_tool_filter(
                    mem_ranker=self.mem_ranker,
                    mem_dir="/home/chengjiawei/Agent4Rec/juntao/data/ml-1m/glintru_scores_memmap",
                    user_id=str(user_id),
                    candidate_tokens=tokens_in,
                    drop_ratio=self.fixed_drop_ratio,
                    min_keep=5,
                    drop_unknown=True,
                )
                item_list_1 = [int(item_id[5:]) for item_id in tool_1]
                target = int(state["target"])
                ndcg_1 = ndcg(item_list_1, target, 10)

                tool_2 = apply_tool_filter(
                    mem_ranker=self.mem_ranker,
                    mem_dir="/home/chengjiawei/Agent4Rec/juntao/data/ml-1m/gru4rec_scores_memmap",
                    user_id=str(user_id),
                    candidate_tokens=tokens_in,
                    drop_ratio=self.fixed_drop_ratio,
                    min_keep=5,
                    drop_unknown=True,
                )
                item_list_2 = [int(item_id[5:]) for item_id in tool_2]
                target = int(state["target"])
                ndcg_2 = ndcg(item_list_2, target, 10)

                tool_3 = apply_tool_filter(
                    mem_ranker=self.mem_ranker,
                    mem_dir="/home/chengjiawei/Agent4Rec/juntao/data/ml-1m/kgat_scores_memmap",
                    user_id=str(user_id),
                    candidate_tokens=tokens_in,
                    drop_ratio=self.fixed_drop_ratio,
                    min_keep=5,
                    drop_unknown=True,
                )
                item_list_3 = [int(item_id[5:]) for item_id in tool_3]
                target = int(state["target"])
                ndcg_3 = ndcg(item_list_3, target, 10)

                tool_4 = apply_tool_filter(
                    mem_ranker=self.mem_ranker,
                    mem_dir="/home/chengjiawei/Agent4Rec/juntao/data/ml-1m/sasrec_scores_memmap",
                    user_id=str(user_id),
                    candidate_tokens=tokens_in,
                    drop_ratio=self.fixed_drop_ratio,
                    min_keep=5,
                    drop_unknown=True,
                )
                item_list_4 = [int(item_id[5:]) for item_id in tool_4]
                target = int(state["target"])
                ndcg_4 = ndcg(item_list_4, target, 10)

                tool_5 = apply_tool_filter(
                    mem_ranker=self.mem_ranker,
                    mem_dir="/home/chengjiawei/Agent4Rec/juntao/data/ml-1m/stamp_scores_memmap",
                    user_id=str(user_id),
                    candidate_tokens=tokens_in,
                    drop_ratio=self.fixed_drop_ratio,
                    min_keep=5,
                    drop_unknown=True,
                )
                item_list_5 = [int(item_id[5:]) for item_id in tool_5]
                target = int(state["target"])
                ndcg_5 = ndcg(item_list_5, target, 10)

                tool_6 = apply_tool_filter(
                    mem_ranker=self.mem_ranker,
                    mem_dir="/home/chengjiawei/Agent4Rec/juntao/data/ml-1m/lightgcn_scores_memmap",
                    user_id=str(user_id),
                    candidate_tokens=tokens_in,
                    drop_ratio=self.fixed_drop_ratio,
                    min_keep=5,
                    drop_unknown=True,
                )
                item_list_6 = [int(item_id[5:]) for item_id in tool_6]
                target = int(state["target"])
                ndcg_6 = ndcg(item_list_6, target, 10)

                # 你给的6个tool对应的名字（和 mem_dir 一一对应）
                tool_scores = {
                    "GLINTRU": ndcg_1,
                    "GRU4REC": ndcg_2,
                    "KGAT": ndcg_3,
                    "SASREC": ndcg_4,
                    "STAMP": ndcg_5,
                    "LIGHTGCN": ndcg_6,
                }

                best_tool_name, best_tool_score = max(tool_scores.items(), key=lambda x: x[1])

                # 如果你还想看所有tool按分数从高到低排序：
                ranked_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
                state["best_tool"] = best_tool_name
                state["best_ndcg"] = best_tool_score
                #print("看看最初候选列表的效果：",ndcg_10)
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
            #print("查看当前第{}轮的filter_tool:{}".format(state["iteration_count"], filter_tool))
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

        print(state["target"])
        print(state["filter_tool"])
        print(tokens_in)
        print(tokens_before_filter)

        if len(tokens_in) < min_keep and len(tokens_before_filter) >= min_keep:
            seen = set(tokens_in)
            for x in tokens_before_filter:
                if x not in seen:
                    tokens_in.append(x)
                    seen.add(x)
                if len(tokens_in) >= min_keep:
                    break
        
        item_list = [int(item_id[5:]) for item_id in tokens_in]
        target = int(state["target"])
        ndcg_10 = ndcg(item_list, target, 10)


        #print("看看使用工具过滤的效果：",ndcg_10)
        self.current_ndcg = ndcg_10
        state["candidate_items"] = normalize_candidate_tokens(tokens_in)

        # log save
        iteration = {}
        iteration[state["iteration_count"]] = f""" planner's tool_selecet:{state["planner_choice"]}, reasoner's reasoning:{state["reasoner_reasoning"]}, reflector's tool_select:{state["filter_tool"]}, NDCG:{ndcg_10}, filtered_list:{normalize_candidate_tokens(tokens_in)}"""
        state["iteration_ls"].append(iteration)

        return state

    def reasoner_agent(self, state: RecommendationState) -> RecommendationState:
        state["candidate_items"] = normalize_candidate_tokens(state.get("candidate_items", []))
        candidate_items = state["candidate_items"]

        formatted_candidates = format_candidates_with_meta(candidate_items, self.item_meta)

        input_memory = state.get("reasoner_memory", [])
        planner_intention = state.get("planner_intention", "")
        planner_reasoning = state.get("planner_explanation", "")
        user_behavior_pattern = state.get("user_pattern", "")

        similar_memory = []
        if getattr(self.args, "reasoner_memory_use", False) and self.tool_memory:
            can_users = list(self.tool_memory.keys())
            similar_user_id = find_most_similar_user_id(
                file_path=self.args.user_emb_profile,
                user_id=str(state["user_id"]),
                candidate_users=can_users,
            )
            for i in similar_user_id:
                temp_similar_memory = self.tool_memory.get(i[1])
                similar_memory.append(temp_similar_memory)
        prompt = reasoner(
            formatted_candidates,
            planner_intention,
            planner_reasoning,
            user_behavior_pattern,
            similar_memory,
            self.args.min_keep
        )

        system_prompt = """Output a JSON object strictly following this format:
        {
        "judgment": "valid" or "invalid",
        "confidence": 0.0-1.0,
        "need_filter": true or false,
        "reasoner_reasoning": "If need_filter=true: max 40 words; else empty string",
        "tool_characteristics": "If need_filter=true: max 40 words; else empty string"
        }
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        print("看看easoner的输出：", self.api_reasoner.chat(messages))
        data = safe_parse_json(self.api_reasoner.chat(messages))

       # 本地模型调用
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
        # print("看看reasoner的输出：",response)
        # # 检查显存使用情况
        # del inputs, outputs, text
        # torch.cuda.empty_cache()

  
        state["reasoner_judgment"] = data.get("judgment", "valid")
        state["confidence"] = float(data.get("confidence", 0.0))
        state["reasoner_reasoning"] = data.get("reasoner_reasoning", "")
        state["need_filter"] = bool(data.get("need_filter", False))
        state["filter_reason"] = str(data.get("filter_reason", ""))
        state["tool_characteristics"] = data.get("tool_characteristics","")

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
        tool_character = state["tool_characteristics"]

        formatted_candidates = format_candidates_with_meta(candidate_items_before, self.item_meta)

        similar_memory = []
        if getattr(self.args, "reflector_memory_use", False) and self.tool_memory:
            can_users = list(self.tool_memory.keys())
            similar_user_id = find_most_similar_user_id(
                file_path=self.args.user_emb_profile,
                user_id=str(state["user_id"]),
                candidate_users=can_users,
            )
            for i in similar_user_id:
                temp_similar_memory = self.tool_memory.get(i[1])
                similar_memory.append(temp_similar_memory)

        prompt = reflector(
            tool_character,
            state.get("reasoner_reasoning", ""),
            formatted_candidates,
            state["user_profile"],
            memory=similar_memory,
            need_filter=state.get("need_filter", False),
            filter_reason=state.get("filter_reason", ""),
            min_keep=int(state.get("min_keep", getattr(self.args, "min_keep", 5))),
            tools=self.args.tools
        )

        if self.args.tools == 6:
            system_prompt = """Output a JSON object strictly following this format:
            {
            "filter_mode": "tool" or "none",
            "filter_tool": "sasrec" or "gru4rec" or "lightgcn" or "stamp" or "glintru" or "kgat",
            "filter_plan_reason": "max 40 words"
            }
            """
        else:
            system_prompt = """Output a JSON object strictly following this format:
            {
            "filter_mode": "tool" or "llm" or "none",
            "filter_tool": "sasrec" or "gru4rec" or "lightgcn",
            "filter_plan_reason": "max 40 words",
            "re_ranked_candidate_items": []
            }
            """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        test_prompt = f"""You are a Tool Selection Expert.

            Available tools (choose exactly ONE):
            - glintru: strong general sequential filtering when there is a clear step-by-step browsing path
            - sasrec: temporal / recent-sequence patterns; latest actions matter most
            - grurec: noisy but order-informative sequences
            - stamp: short-session / bursty intent
            - lightgcn: stable long-term preferences; weak sequential signals
            - kgat: knowledge/relation-aware signals are required

            Return:
            {{"tool_selected":"..."}} </final>
            """
        prompts = {"prompt":prompt}
        train_dataset = Dataset.from_dict(prompts)
        # print("看看reflector的输出：",self.api_reasoner.chat(messages))
        # data = safe_parse_json(self.api_reflector.chat(messages))

       # 本地模型调用
        grpo_args = GRPOConfig(
            output_dir="./tool_selector_grpo",

            # ===== 训练参数 =====
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=1e-5,

            # ===== GRPO 关键参数 =====
            num_generations=4,              # 强烈建议 >=2
            max_completion_length=128,
            temperature=0.5,
            # use_vllm=True,
            # ===== 其它 =====
            remove_unused_columns=False,
            bf16=True,
            beta=0.1,
    )

        # --- Trainer ---
        trainer = GRPOTrainer(
            model=self.model,                 # 直接传模型名/路径
            args=grpo_args,
            train_dataset=train_dataset,
            reward_funcs=tool_accuracy_reward,
        )
        trainer.train()
        # 测试结束
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0001
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0, prompt_len:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        data = safe_parse_json(response)
        print("看看reflector的输出：",response)

        # 检查显存使用情况
        del inputs, outputs, text
        torch.cuda.empty_cache()

        
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

        if ("invalid" in judgment or confidence < 0.85) and iteration_count < max_filter_rounds:
            return "continue"


        # save log
        log = {}
        can_users = list(self.tool_memory.keys())
        similar_user_id = find_most_similar_user_id(
        file_path=self.args.user_emb_profile,
        user_id=str(state["user_id"]),
        candidate_users=can_users,
        )

        similar_memory = self.tool_memory.get(similar_user_id[0][1])
        log["retrive_memory"] = similar_memory
        log["iteration"] = state["iteration_ls"]
        self.work_log[state["user_id"]] = log

        user_id = state["user_id"]
        last_tool = state["filter_tool"]
        best_tool = state["best_tool"]
        best_ndcg10 = state["best_ndcg"]
        if last_tool.lower() == best_tool.lower():
            print(f"current user_{user_id} selects the best tool for filtering")
        print(f"user_id:{user_id}, last_tool:{last_tool}, last_ndcg@10:{self.current_ndcg}, best_tool:{best_tool}, best_ndcg@10: {best_ndcg10}")

        return "finish"

    def run(self, user_profile, user_id, target=None, max_iterations=0, train_step=0, train=False) -> List[Dict[str, Any]]:
        init: RecommendationState = {
            "tool_set": {"sasrec", "grurec", "lightgcn", "glintru", "kgat", "stamp"},
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
