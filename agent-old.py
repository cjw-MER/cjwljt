import json
import numpy as np
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
import operator
from utils import *
from metric import *
from prompts import *
from state import RecommendationState, UserMemory
from tools import retrieval_topk, stdout_retrived_items, retrieval_topk_load, stdout_retrived_items_load
import torch
import re
from api import LLMAPI, ZHI, DEEPSEEK
import ast

class RecommendationAgent:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.workflow = self._build_workflow()
        self.tool_memory = load_json_file(args.tool_memory_file)
        self.reasoning_memory = load_json_file(args.reasoning_memory_file)
        self.item_inforamtion = None
        self.api_planner = ZHI()
        self.api_reasoner = ZHI()
        self.api_reflector = ZHI()
        self.api_summray = ZHI()

    def _build_workflow(self, train=False) -> StateGraph:
        """Builds the recommendation workflow"""
        workflow = StateGraph(RecommendationState)
        
        # Add nodes
        workflow.add_edge("__start__", "planner")
        workflow.add_node("planner", self.planner_agent)
        workflow.add_node("reasoner", self.reasoner_agent)
        workflow.add_node("reflector", self.reflector_agent)
        workflow.add_node("tool_node", self.tool_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add edges
        workflow.add_edge("planner", "tool_node")
        workflow.add_edge("tool_node", "reasoner")
        workflow.add_edge("reasoner", "reflector")
        # workflow.add_edge("reflector", "tool_node")
    
        # Add conditional edges
        # workflow.add_conditional_edges(
        #     "reasoner",
        #     self.should_continue_or_finish,
        #     {
        #         "continue": "reflector",
        #         "finish": END
        #     }
        # )
        
        workflow.add_conditional_edges(
            "reflector",
            self.should_continue_or_finish,
            {
                "continue": "tool_node",
                "finish": END
            }
        )
        
        return workflow.compile()

    def retrive_memory(self, path, user_id):
        can_users = list(self.reasoning_memory.keys())
        similar_user_id = find_most_similar_user_id(file_path=path, user_id=user_id, candidate_users=can_users)
        similar_memory = self.reasoning_memory[similar_user_id]
        return similar_memory
    
    def planner_agent(self, state: RecommendationState) -> RecommendationState:
        print(""" Planner working""")
        user_profile = state["user_profile"]
        iteration = state.get("iteration_count", 0)
        user_id = state["user_id"]
        tool_log_user = {}

        if self.args.train:
            print("............训练memory.......")
            external_item_list_sasrec = retrieval_topk_load(model_file="/home/liujuntao/Agent4Rec/data/ml-1m/test_topk_prediction_random_10.json", user_id=user_id)
            retrived_items_sasrec = stdout_retrived_items_load(external_item_list_sasrec)
            external_item_list_grurec = retrieval_topk_load(model_file="/home/liujuntao/Agent4Rec/data/ml-1m/test_topk_prediction_random_10.json", user_id=user_id)
            retrived_items_grurec = stdout_retrived_items_load(external_item_list_grurec)
            external_item_list_lightgcn = retrieval_topk_load(model_file="/home/liujuntao/Agent4Rec/data/ml-1m/test_topk_prediction_random_10.json", user_id=user_id)
            retrived_items_lightgcn = stdout_retrived_items_load(external_item_list_lightgcn)
            target = state["target"]
            target = '{}'.format(target)
            tool_selected, rank = select_best_model_with_rank(external_item_list_sasrec[0], external_item_list_grurec[0], external_item_list_lightgcn[0], target)
            tool_log_user[user_id] = (tool_selected, rank)
            if self.args.planner_memory_use:
                similar_memory = retrive_memory(self.args.user_emb_profile, user_id)
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

            messages = [{"role": "system", "content": system_planner_prompt},
                        {"role": "user", "content": planner_prompt}]

            # API 调用
            response = self.api_planner.chat(messages)

            # 本地调用
            # text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            # inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            # with torch.inference_mode():
            #     outputs = self.model.generate(
            #         **inputs,
            #         max_new_tokens=512,
            #         temperature=0.01
            #     )
            # prompt_len = inputs["input_ids"].shape[1]
            # gen_ids = outputs[0, prompt_len:]
            # response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            # del inputs, outputs, text
            # torch.cuda.empty_cache()
        else:
            if self.args.planner_memory_use:
                similar_memory = self.retrive_memory(self.args.user_emb_profile, user_id)
                planner_prompt = planner(user_profile, similar_memory)
            else:
                planner_prompt = planner(user_profile)

            system_planner_prompt = """Output a JSON object strictly"""

            messages = [{"role": "system", "content": system_planner_prompt},
                        {"role": "user", "content": planner_prompt}]

            # API 调用
            response = self.api_planner.chat(messages)

            # 本地调用
            # text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            # inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            # with torch.inference_mode():
            #     outputs = self.model.generate(
            #         **inputs,
            #         max_new_tokens=512,
            #         temperature=0.01
            #     )
            # prompt_len = inputs["input_ids"].shape[1]
            # gen_ids = outputs[0, prompt_len:]
            # response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            # del inputs, outputs, text
            # torch.cuda.empty_cache()

        print("看看planner的输出：",response)
        planner_data = safe_parse_json(response)
        intention, tool_selected, planner_reasoning = planner_data["planner_intention"], planner_data["tool_selected"], planner_data["planner_reasoning"]
        
        state["planner_intention"] = intention
        state["planner_explanation"] = planner_reasoning
        state["model_choice"] = tool_selected
        state["planner_summary"] = None
        return state
    
    def tool_node(self, state: RecommendationState) -> RecommendationState:
        print(""" tool working""")
        user_id = state["user_id"]
        reorder_items = state["re_candidate_items"]

        if len(reorder_items) > 0:
            print("用重排序的set替换.....")
            retrived_items = reorder_items
        else:
            if state["iteration_count"] == 0:
                if state["model_choice"] in state["tool_set"]:
                    tool_name = state["model_choice"].lower()
                    print(f"*********目前使用{tool_name}********")
                    external_item_list = retrieval_topk_load(model_file=f"/home/liujuntao/Agent4Rec/data/ml-1m/test_topk_prediction_random_10.json", user_id=user_id)
                    retrived_items = stdout_retrived_items_load(external_item_list)
                else:
                    print("大模型生成出问题，拿前一次的list作为大模型重排的结果")
                    retrived_items = state["candidate_items"]

        state["candidate_items"] = retrived_items
        return state

    def reasoner_agent(self, state: RecommendationState) -> RecommendationState:
        print(""" Reasoner working""")
        candidate_items = state["candidate_items"]
        user_profile = state["user_profile"]
        input_memory = state.get("reasoner_memory", [])
        planner_intention = state["planner_intention"]
        planner_reasoning = state["planner_explanation"]
        user_id = state["user_id"]
        target = state["target"]

        if self.args.reasoner_memory_use:
            similar_memory = self.retrive_memory(self.args.user_emb_profile, user_id)
            reasoner_prompt = reasoner(candidate_items, planner_intention, planner_reasoning, input_memory, similar_memory)
        else:
            reasoner_prompt = reasoner(candidate_items, planner_intention, planner_reasoning, input_memory)

        system_reasoner_prompt = """Output a JSON object strictly"""

        messages = [{"role": "system", "content": system_reasoner_prompt},
                    {"role": "user", "content": reasoner_prompt}]

        # 花钱的调用
        response = self.api_reasoner.chat(messages)

        #本地大模型调用
        # text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        # with torch.inference_mode():
        #     outputs = self.model.generate(
        #         **inputs,
        #         max_new_tokens=512,
        #         temperature=0.01
        #     )
        # prompt_len = inputs["input_ids"].shape[1]
        # gen_ids = outputs[0, prompt_len:]
        # response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # # 检查显存使用情况
        # del inputs, outputs, text
        # torch.cuda.empty_cache()

        print("看看reasoner的输出:", response)
        reasoner_data = safe_parse_json(response)
        judgment, confidence_score, current_reasoning = reasoner_data["judgment"], reasoner_data["confidence"], reasoner_data["reasoner_reasoning"]
        print(current_reasoning)
        state["reasoner_judgment"] = judgment
        state["reasoner_reasoning"] = current_reasoning
        state["confidence"] = confidence_score

        reasoner_memory = "In round {},  the reasonableness of the candidate list is {}; the reason is {}".format(state["iteration_count"], judgment, current_reasoning)
        state["reasoner_memory"].append(reasoner_memory)

        return state
    
    def reflector_agent(self, state: RecommendationState) -> RecommendationState:
        print(""" Reflector working""")
        current_tool = state["model_choice"]
        reasoner_reasoning = state["reasoner_reasoning"]
        candidate_items = state["candidate_items"]
        re_candidate_items = state["re_candidate_items"]
        user_profile = state["user_profile"]
        judgment = state["reasoner_judgment"]
        user_id = state["user_id"]

        if self.args.reflector_memory_use:
            similar_memory = self.retrive_memory(self.args.user_emb_profile, user_id)
            reflector_prompt = reflector(reasoner_reasoning, candidate_items, re_candidate_items, similar_memory)
        else:
            reflector_prompt = reflector(reasoner_reasoning, candidate_items, re_candidate_items)

        system_reflector_prompt = """Output a JSON object strictly following this format:
            {
                re_ranked_candidate_items: ["same items as candidate_items, reordered, each exactly once"]  
            }
            """

        messages = [{"role": "system", "content": system_reflector_prompt},
                    {"role": "user", "content": reflector_prompt}]

        # 花钱的调用
        response = self.api_reflector.chat(messages)

        #不花钱的调用
        # text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        # with torch.inference_mode():
        #     outputs = self.model.generate(
        #         **inputs,
        #         max_new_tokens=512,
        #         temperature=0.01
        #     )
        # prompt_len = inputs["input_ids"].shape[1]
        # gen_ids = outputs[0, prompt_len:]
        # response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 检查显存使用情况
        # del inputs, outputs, text
        # torch.cuda.empty_cache()

        print("看看reflector的输出是什么:", response)
        reflector_data = safe_parse_json(response)
        re_candidate_items = reflector_data["re_ranked_candidate_items"]
        print("看看抽取出来的重新排序的内容:", re_candidate_items)

        # Extract item_id list
        if type(candidate_items) == list:
            text = ""
            for i in candidate_items:
                text = text + i
            item_ids = re.findall(r"item_ID:item_(\d+)", text)
        else:
            item_ids = re.findall(r"item_ID:item_(\d+)", candidate_items)

        if type(re_candidate_items) == list:
            text = ""
            for i in re_candidate_items:
                text = text + i
            item_ids_re = re.findall(r"item_ID:item_(\d+)", text)
        else:
            item_ids_re = re.findall(r"item_ID:item_(\d+)", re_candidate_items)

       #计算当前重新排序前后的推荐指标，topk根据当前实验的候选集来定，10->10, 50->30, 100->50
        item_list = [int(item_id) for item_id in item_ids]
        target = int(state["target"])
        ndcg_10 = ndcg(item_list, target, len(item_list))

        item_list_re = [int(item_id) for item_id in item_ids_re]
        target = int(state["target"])
        ndcg_10_re = ndcg(item_list_re, target, len(item_list_re))
        print("排序前的list:",item_list)
        print("排序前推荐的指标为：", ndcg_10)
        print("排序后的list:",item_list_re)
        print("排序后的推荐指标为:", ndcg_10_re)

        
        # 训练时候使用的memory存储内容
        if ndcg_10 == 0:
            state["NDCG"]=0.0
            reflector_memory = """In round {}, the the NDCG metric is 0, user's target is not present in the current candidate list, ending reflection.""".format(state["iteration_count"])
        elif ndcg_10 > ndcg_10_re:
            state["NDCG"]=1.0
            reflector_memory = """In round {},  Re-ranking' reasoning is {}. After sorting according to this reasoning, the NDCG metric dropped from {} to {}, indicating that the reasoning this round is incorrect.""".format(state["iteration_count"], reasoner_reasoning, ndcg_10, ndcg_10_re)
            state["reflector_memory"].append(reflector_memory)
        elif ndcg_10 < ndcg_10_re: 
            state["NDCG"]=1.0
            reflector_memory = """In round {},  Re-ranking' reasoning is {}. After sorting according to this reason, the NDCG metric increased from {} to {}, indicating that the reasoning this round is correct.""".format(state["iteration_count"], reasoner_reasoning, ndcg_10, ndcg_10_re)
            state["reflector_memory"].append(reflector_memory)        
        else:
            state["NDCG"]=1.0
            reflector_memory = """In round {},  Re-ranking' reasoning is {}. After sorting according to this reason, the NDCG metric remains unchanged, indicating that the reasoning this round may be correct.""".format(state["iteration_count"], reasoner_reasoning, ndcg_10_re, ndcg_10)
            state["reflector_memory"].append(reflector_memory)

        state["final_recommendations"] = item_ids_re
        state["re_candidate_items"] = re_candidate_items
        state["iteration_count"] = state["iteration_count"] + 1

        return state
    
    def should_continue_or_finish(self, state: RecommendationState) -> str:
        print(""" Decides whether to continue iterating or finish""")
        judgment_result = state["reasoner_judgment"]
        iteration_count = state["iteration_count"]
        max_iterations = state.get("max_iterations", 1)
        confidence = state.get("confidence", 0.85)
        reorder_conversation = state["reorder_conversation"]
        NDCG = state["NDCG"]
        
        if NDCG == 0.0:
            print("########不需要反思,target不在当前的候选列表中########")
            return "finish"
        else:
            if "invalid" in judgment_result:
                if iteration_count < max_iterations:
                    print("########需要反思########")
                    return "continue"
                else:
                    state["iteration_count"] = 0
                    print("########不需要反思,达到最大反思次数########")
                    return "finish"
            elif float(confidence) < 0.85:
                if iteration_count < max_iterations:
                    print("########需要反思, 置信度低于0.85########")
                    return "continue"
                else:
                    state["iteration_count"] = 0
                    print("########不需要反思,达到最大反思次数########")
                    return "finish"
            else:
                print("########不需要反思,候选列表合理########")
                return "finish"

    def should_train_or_test(self, state: RecommendationState) -> str:
        print(""" Decides whether to train or test""")
        train_step = state["train_step"]
        print("训练步长：",train_step)
        if train_step >= 1000:
            print("########测试########")
            state["train"] = False
            return "test"

        else:
            print("########训练步长:{}########".format(train_step))
            state["train"] = True
            return "train"
    
    
    def run(self, user_profile, user_id, target=None, max_iterations=0, train_step=None, train=False) -> Dict[str, Any]:
        """Runs the recommendation workflow"""
        
        initial_state = {
            "tool_set": "SASRec, GRURec, LightGCN",
            "NDCG": 0.0,
            "reorder_conversation": {}, # 记录当前用户的每一次推理的原因和结果
            "reasoner_memory": [],
            "reflector_memory": [],
            "train_step": train_step,
            "train": train,
            "confidence": 0.0,
            "model_choice": "" ,
            "user_profile": user_profile,
            "user_id": user_id,
            "planner_summary": "",
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "candidate_items": [],
            "re_candidate_items": [],
            "target": target,
            "planner_explanation": "",
            "planner_intention": "",
            "reasoner_judgment": "",
            "reasoner_reasoning": "",
            "reflection_result": "",
            "final_recommendations": [],
            "final_explanation": ""
        }
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        return result["final_recommendations"]
        
