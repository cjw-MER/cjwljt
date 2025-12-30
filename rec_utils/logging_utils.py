from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


def safe_jsonify(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [safe_jsonify(v) for v in x]
    if isinstance(x, dict):
        return {str(k): safe_jsonify(v) for k, v in x.items()}
    return str(x)


def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def clip_text(x: Any, max_chars: int) -> str:
    s = "" if x is None else str(x)
    return s if len(s) <= max_chars else s[:max_chars] + f"...[clipped:{len(s)}]"


@dataclass
class RunMetaManager:
    run_dir: str
    run_id: str
    git_commit: Optional[str]
    args: Any

    def path(self) -> str:
        return os.path.join(self.run_dir, "run_meta.json")

    def read(self) -> Dict[str, Any]:
        p = self.path()
        if not os.path.exists(p):
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def write(self, started_at: Optional[datetime] = None, finished_at: Optional[datetime] = None) -> None:
        meta = self.read()
        if started_at is not None:
            meta["started_at"] = started_at.isoformat()
        if finished_at is not None:
            meta["finished_at"] = finished_at.isoformat()

        meta.update(
            {
                "run_id": self.run_id,
                "run_dir": self.run_dir,
                "pid": os.getpid(),
                "git_commit": self.git_commit,
                "args": safe_jsonify(vars(self.args) if hasattr(self.args, "__dict__") else {}),
            }
        )
        with open(self.path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def update_tool_filter_stats(
        self,
        tool_log_dir: str,
        total_users: int,
        ok_users: int,
        skipped_users: int,
        elapsed_seconds: float,
        tool_name: str = "rank_candidates_by_memdir",
        action: str = "tool_filter",
        model_field: str = "filter_tool",
    ) -> None:
        meta = self.read()
        meta["total_users"] = int(total_users)
        meta["ok_users"] = int(ok_users)
        meta["skipped_users"] = int(skipped_users)
        meta["elapsed_seconds"] = float(elapsed_seconds)

        files = sorted(glob.glob(os.path.join(tool_log_dir, "user_*.jsonl")))
        meta["users_logged"] = int(len(files))
        meta["tool_filter_calls_total"] = 0
        meta["tool_filter_calls_by_model"] = {}

        if files:
            dfs = []
            for fp in files:
                try:
                    dfs.append(pd.read_json(fp, lines=True))
                except Exception:
                    continue

            if dfs:
                all_df = pd.concat(dfs, ignore_index=True)
                need_cols = {"tool_name", "action", "inputs"}
                if need_cols.issubset(set(all_df.columns)):
                    df_tf = all_df[(all_df["tool_name"] == tool_name) & (all_df["action"] == action)].copy()
                    meta["tool_filter_calls_total"] = int(len(df_tf))
                    if len(df_tf) > 0:
                        df_tf[model_field] = df_tf["inputs"].apply(
                            lambda x: (x or {}).get(model_field, None) if isinstance(x, dict) else None
                        )
                        vc = df_tf[model_field].value_counts(dropna=False)
                        meta["tool_filter_calls_by_model"] = {str(k): int(v) for k, v in vc.items()}

        with open(self.path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


class ConversationLogger:
    def __init__(self, conv_log_dir: str, run_id: str, enabled: bool = True, max_chars: int = 100000):
        self.conv_log_dir = conv_log_dir
        self.run_id = run_id
        self.enabled = bool(enabled)
        self.max_chars = int(max_chars)

    def _path(self, user_id: str) -> str:
        return os.path.join(self.conv_log_dir, f"user_{str(user_id)}.jsonl")

    def log_turn(
        self,
        state: Dict[str, Any],
        stage: str,
        messages: List[Dict[str, str]],
        response: Optional[str] = None,
        parsed: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        safe_messages = [
            {"role": str(m.get("role", "")), "content": clip_text(m.get("content", ""), self.max_chars)}
            for m in (messages or [])
        ]

        payload = {
            "ts": time.time(),
            "run_id": self.run_id,
            "user_id": str(state.get("user_id", "")),
            "iteration_count": int(state.get("iteration_count", 0)),
            "filter_round": int(state.get("filter_round", 0)),
            "stage": str(stage),
            "messages": safe_messages,
        }
        if response is not None:
            payload["response_raw"] = clip_text(response, self.max_chars)
        if parsed is not None:
            payload["response_parsed"] = parsed
        if extra:
            payload["extra"] = extra

        append_jsonl(self._path(state.get("user_id", "")), payload)


class ToolFilterLogger:
    def __init__(self, tool_log_dir: str, run_id: str, enabled: bool = True):
        self.tool_log_dir = tool_log_dir
        self.run_id = run_id
        self.enabled = bool(enabled)

    def _path(self, user_id: str) -> str:
        return os.path.join(self.tool_log_dir, f"user_{str(user_id)}.jsonl")

    def log_filter_call(
        self,
        state: Dict[str, Any],
        filter_tool: str,
        mem_dir: str,
        drop_ratio: float,
        min_keep: int,
        tokens_in_len: int,
        tool_name: str = "rank_candidates_by_memdir",
    ) -> None:
        if not self.enabled:
            return

        payload = {
            "ts": time.time(),
            "run_id": self.run_id,
            "user_id": str(state.get("user_id", "")),
            "iteration_count": int(state.get("iteration_count", 0)),
            "filter_round": int(state.get("filter_round", 0)),
            "tool_name": tool_name,
            "action": "tool_filter",
            "inputs": {
                "filter_tool": str(filter_tool),
                "mem_dir": str(mem_dir),
                "drop_ratio": float(drop_ratio),
                "min_keep": int(min_keep),
                "tokens_in_len": int(tokens_in_len),
            },
            "outputs": {},
        }
        append_jsonl(self._path(state.get("user_id", "")), payload)
