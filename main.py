import asyncio
import os
import json
import time
import random
from datetime import datetime
from typing import Any, Dict, List

from tqdm import tqdm

from agent import RecommendationAgent
from utils import load_json_file
from baseline import metric_cal

APIRequestFailedError = None


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--item_database_file', type=str,
                        default='/home/liujuntao/Agent4Rec/data/ml-1m/ml-1m.item.json')
    parser.add_argument('--test_user_inter50_file', type=str,
                        default='/home/liujuntao/ml-1m_user2000_inter50_.json')
    parser.add_argument('--train_user_inter50_file', type=str,
                        default='/home/liujuntao/Agent4Rec/data/ml-1m/user_train_inter50_1.json')

    parser.add_argument('--planner_memory_use', type=bool, default=True)
    parser.add_argument('--reasoner_memory_use', type=bool, default=True)
    parser.add_argument('--reflector_memory_use', type=bool, default=False)

    parser.add_argument('--user_emb_profile', type=str,
                        default='/home/liujuntao/Agent4Rec/user_profile_sasrec_bottom10.json')
    parser.add_argument('--tool_memory_file', type=str,
                        default='/home/liujuntao/Agent4Rec/memory结构化版本/data/tool_memory_merge.json')
    parser.add_argument('--reasoning_memory_file', type=str,
                        default='/home/liujuntao/Agent4Rec/memory结构化版本/data/reasoning_memory_struct_1000.json')

    parser.add_argument('--train', type=bool, default=False)

    parser.add_argument('--max_concurrency', type=int, default=100)
    parser.add_argument('--max_users', type=int, default=1988)

    parser.add_argument('--output_root', type=str, default='./runs')
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
            result = await asyncio.to_thread(
                agent.run,
                user_profile,
                user_id,
                target,
                max_iterations=5,
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

        print(f"[DONE] done={done_n}/{total_users} (i={i+1}) uid={user_id} skipped={out.get('skipped', False)}",
              flush=True)
        return out


async def run_batch_recommendation(args, model, tokenizer):
    run_dir = make_run_dir(args.output_root)

    item_database = load_json_file(args.item_database_file)
    all_inter = load_json_file(args.test_user_inter50_file)

    users = list(all_inter.keys())
    max_users = min(int(args.max_users), len(users))

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

    ok_rows = [r for r in batch_results if r.get("ok") is True]
    skipped_rows = [r for r in batch_results if r.get("skipped") is True]
    skipped_uid_set = sorted(set(r["user_id"] for r in skipped_rows))

    # 有效用户（用于评测）
    ok_user_ids = [r["user_id"] for r in ok_rows]
    results = [r["result"] for r in ok_rows]
    ground_truth = [int(r["target"]) for r in ok_rows]

    # top10 pad
    results_top10 = []
    for recs in results:
        if isinstance(recs, list) and len(recs) >= 10:
            results_top10.append(recs[:10])
        else:
            results_top10.append(recs)

    # 保存 run 目录内所有文件
    with open(os.path.join(run_dir, "skipped_uids.txt"), "w", encoding="utf-8") as f:
        for uid in skipped_uid_set:
            f.write(str(uid) + "\n")

    dump_json(os.path.join(run_dir, "summary.json"), {
        "run_dir": run_dir,
        "time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "total_users": len(batch_results),
        "ok_users": len(ok_rows),
        "skipped_users": len(skipped_uid_set),
        "max_concurrency": int(args.max_concurrency),
        "max_users": int(max_users),
    })
    dump_json(os.path.join(run_dir, "ok_user_ids.json"), ok_user_ids)
    dump_json(os.path.join(run_dir, "ground_truth_ok.json"), ground_truth)
    dump_json(os.path.join(run_dir, "results_ok_top10.json"), results_top10)
    dump_json(os.path.join(run_dir, "skipped_rows.json"), skipped_rows)

    print(f"[DONE] run_dir={run_dir}", flush=True)
    print(f"[DONE] total={len(batch_results)} ok={len(ok_rows)} skipped={len(skipped_uid_set)} elapsed={elapsed:.3f}s",
          flush=True)

    # ===== 修复：显式传 ok_user_ids，确保评测用户与结果对齐 =====
    metric_cal(
        candidate_items=results_top10,
        k=10,
        selected_users_path=args.test_user_inter50_file,  # 仍保留原文件路径用于读取/映射
        out_dir=run_dir,                                  # 指标存到本次 run 文件夹
        out_prefix="Agent4Rec",
        eval_user_tokens=ok_user_ids,                     # 关键：避免跳过用户导致错位
    )

    return run_dir


if __name__ == "__main__":
    random.seed(42)
    args = get_args()

    tokenizer = None
    model = None

    asyncio.run(run_batch_recommendation(args, model, tokenizer))
