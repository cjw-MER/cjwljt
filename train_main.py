import asyncio
import os
import json
import time
import random
from datetime import datetime
from typing import Any, Dict, List

from tqdm import tqdm
import torch
from agent import RecommendationAgent
from utils import load_json_file
from baseline import metric_cal
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import BitsAndBytesConfig

from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from vllm import LLM, SamplingParams
APIRequestFailedError = None


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--item_database_file",
        type=str,
        default="/home/chengjiawei/Agent4Rec/data/ml-1m/ml-1m.item.json",
    )
    parser.add_argument(
        "--test_user_inter50_file",
        type=str,
        default="/home/chengjiawei/Agent4Rec/data/ml-1m/ml-1m_user2000_inter50.json",
    )
    parser.add_argument(
        "--train_user_inter50_file",
        type=str,
        default="/home/liujuntao/Agent4Rec/data/ml-1m/user_train_inter50_1.json",
    )

    parser.add_argument("--planner_memory_use", action='store_true', default=False)
    parser.add_argument("--reasoner_memory_use", action='store_true', default=False)
    parser.add_argument("--reflector_memory_use", action='store_true', default=False)


    parser.add_argument(
        "--user_emb_profile",
        type=str,
        default="/home/chengjiawei/Agent4Rec/data/ml-1m/user_profile_sasrec_bottom10.json",
    )
    parser.add_argument(
        "--tool_memory_file",
        type=str,
        default="/home/chengjiawei/Agent4Rec/juntao/memory结构化版本/data/agent_memory_tool.json",
    )
    parser.add_argument(
        "--reasoning_memory_file",
        type=str,
        default="/home/chengjiawei/Agent4Rec/juntao/memory结构化版本/data/agent_memory_tool.json",
    )
    parser.add_argument(
        "--candidate_item",
        type=str,
        default="/home/chengjiawei/Agent4Rec/data/ml-1m/test_topk_prediction_user2000_random_30.json",
    )

    # 新版 agent 建议显式加这些参数（与你之前“固定丢弃率/保底 min_keep”一致）
    parser.add_argument("--fixed_drop_ratio", type=float, default=0.3)
    parser.add_argument("--max_filter_rounds", type=int, default=3)
    parser.add_argument("--min_keep", type=int, default=5)

    parser.add_argument("--train", type=bool, default=False)

    parser.add_argument("--max_concurrency", type=int, default=1)
    parser.add_argument("--max_users", type=int, default=20)
    parser.add_argument("--tools", type=int, default=6)

    parser.add_argument("--output_root", type=str, default="./runs")
    return parser.parse_args()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def make_run_dir(output_root: str) -> str:
    ts = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"{ts}_pid{os.getpid()}")
    ensure_dir(run_dir)
    return run_dir


def dump_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_content_filter_1301(err: Exception) -> bool:
    s = str(err)
    return ('"code":"1301"' in s) or ("contentFilter" in s and "1301" in s) or ("1301" in s and "敏感" in s)


async def process_user(
    agent: RecommendationAgent,
    user_profile: str,
    user_id: str,
    target: str,
    i: int,
    semaphore: asyncio.Semaphore,
    train_step: int,
    train: bool,
    done_counter: Dict[str, int],
    done_lock: asyncio.Lock,
    total_users: int,
) -> Dict[str, Any]:
    async with semaphore:
        try:
            # agent.run 是同步函数：用 asyncio.to_thread 放到线程里跑，避免阻塞 event loop
            result = await asyncio.to_thread(
                agent.run,
                user_profile,
                user_id,
                target,
                max_iterations=2,
                train_step=train_step,
                train=train,
            )
            out = {"ok": True, "skipped": False, "user_id": str(user_id), "target": str(target), "result": result}
        except Exception as e:
            if (APIRequestFailedError is not None and isinstance(e, APIRequestFailedError) and is_content_filter_1301(e)) \
               or (APIRequestFailedError is None and is_content_filter_1301(e)):
                out = {"ok": False, "skipped": True, "user_id": str(user_id), "target": str(target), "error": str(e)}
            else:
                raise

        async with done_lock:
            done_counter["n"] += 1
            done_n = done_counter["n"]

        return out


async def run_batch_recommendation(args, model, tokenizer):
    run_dir = make_run_dir(args.output_root)

    item_database = load_json_file(args.item_database_file)
    all_inter = load_json_file(args.test_user_inter50_file)

    users = list(all_inter.keys())

    max_users = len(users)

    agent = RecommendationAgent(args, model, tokenizer)
    semaphore = asyncio.Semaphore(int(args.max_concurrency))
    done_counter = {"n": 0}
    done_lock = asyncio.Lock()

    start = time.perf_counter()

    tasks = []
    for i in tqdm(range(max_users), desc="Building tasks", unit="user"):
        user = users[i]
        inters = all_inter[user]
        target = inters[-1]
        # print(f"Preparing user {i+1}/{max_users}: user_id={user}, target={target}", flush=True)

        history = {f"item_{j}": item_database[f"item_{j}"] for j in inters[-20:-1]}
        user_profile = f"""user_{user}'s interaction history:{history}"""
        user_id = user

        tasks.append(
            process_user(
                agent=agent,
                user_profile=user_profile,
                user_id=user_id,
                target=str(target),
                i=i,
                semaphore=semaphore,
                train_step=2000,
                train=False,
                done_counter=done_counter,
                done_lock=done_lock,
                total_users=max_users,
            )
        )

    batch_results: List[Dict[str, Any]] = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start

    # save log
    with open("work_log_2000.json", "w", encoding="utf-8") as f:
        json.dump(agent.work_log, f, ensure_ascii=False, indent=2)

    ok_rows = [r for r in batch_results if r.get("ok") is True]
    skipped_rows = [r for r in batch_results if r.get("skipped") is True]
    skipped_uid_set = sorted(set(r["user_id"] for r in skipped_rows))

    ok_user_ids = [r["user_id"] for r in ok_rows]
    results = [r["result"] for r in ok_rows]
    ground_truth = [int(r["target"]) for r in ok_rows]

    results_top10 = []
    for recs in results:
        if isinstance(recs, list) and len(recs) >= 10:
            results_top10.append(recs[:10])
        else:
            results_top10.append(recs)

    with open(os.path.join(run_dir, "skipped_uids.txt"), "w", encoding="utf-8") as f:
        for uid in skipped_uid_set:
            f.write(str(uid) + "\n")

    dump_json(
        os.path.join(run_dir, "summary.json"),
        {
            "run_dir": run_dir,
            "time": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed,
            "total_users": len(batch_results),
            "ok_users": len(ok_rows),
            "skipped_users": len(skipped_uid_set),
            "max_concurrency": int(args.max_concurrency),
            "max_users": int(max_users),
            "fixed_drop_ratio": float(getattr(args, "fixed_drop_ratio", 0.2)),
            "min_keep": int(getattr(args, "min_keep", 5)),
            "max_filter_rounds": int(getattr(args, "max_filter_rounds", 3)),
        },
    )
    dump_json(os.path.join(run_dir, "ok_user_ids.json"), ok_user_ids)
    dump_json(os.path.join(run_dir, "ground_truth_ok.json"), ground_truth)
    dump_json(os.path.join(run_dir, "results_ok_top10.json"), results_top10)
    dump_json(os.path.join(run_dir, "skipped_rows.json"), skipped_rows)

    print(f"[DONE] run_dir={run_dir}", flush=True)
    print(
        f"[DONE] total={len(batch_results)} ok={len(ok_rows)} skipped={len(skipped_uid_set)} elapsed={elapsed:.3f}s",
        flush=True,
    )


    metric_cal(
        candidate_items=results_top10,
        k=10,
        selected_users_path=args.test_user_inter50_file,
        out_dir=run_dir,
        out_prefix="Agent4Rec",
        eval_user_tokens=ok_user_ids,
    )

    return run_dir


if __name__ == "__main__":
    random.seed(42)
    args = get_args()
    print(args)

    # 初始化tokenizer（保持不变）
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "/home/chengjiawei/Agent4Rec/Qwen/Qwen3-4B",
    #     trust_remote_code=True
    # )
    
    # # 用vLLM替换原来的模型加载
    # model = LLM(
    #     model="/home/chengjiawei/Agent4Rec/Qwen/Qwen3-4B",
    #     tokenizer="/home/chengjiawei/Agent4Rec/Qwen/Qwen3-4B",
    #     trust_remote_code=True,
    #     tensor_parallel_size=1,
    #     gpu_memory_utilization=0.9,
    #     max_model_len=8192,
    #     dtype="float16",
    # )

    # tokenizer = AutoTokenizer.from_pretrained("/home/chengjiawei/Agent4Rec/Qwen/Qwen3-4B")
    # model = AutoModelForCausalLM.from_pretrained(
    #     "/home/chengjiawei/Agent4Rec/Qwen/Qwen3-4B",
    #     torch_dtype=torch.float16,
    #     device_map="cuda:1", # balanced
    # ).eval()
    MODEL_NAME = "/home/chengjiawei/Qwen3-4B" 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,  # 建议配合 CUDA_VISIBLE_DEVICES 使用
    ).to("cuda:0")
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()

    asyncio.run(run_batch_recommendation(args, model=base_model, tokenizer=tokenizer))