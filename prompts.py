# prompts.py

def planner(user_profile, memory=None):
    planner_prompt = f"""Your task is to analyze the provided user interaction history and an external similar-user hint, then plan the recommendation workflow.

                          INPUT:
                          - user_history: {user_profile}
                          - similar_user_hint: {memory}

                          You MUST:
                          1. Examine the user interaction history for viewing patterns and preferences.
                          2. Use similar_user_hint as auxiliary evidence (NOT ground truth) when tool choice is ambiguous.
                          3. Choose ONE recommendation tool based on the actual USER BEHAVIOR CHARACTERISTICS.
                          4. Analyze and summarize the user's behavior pattern explicitly.
                          5. Output must be a valid JSON object containing exactly FOUR keys:
                            "tool_selected", "planner_intention", "user_behavior_pattern", "planner_reasoning".

                          IMPORTANT:
                          - Tool choice MUST be based on FUNCTION, NOT list order.
                          - intention should predict composite genres the user is likely to watch next.
                          - user_behavior_pattern should describe HOW the user consumes content
                            (e.g., short-term/session-driven, recency-sensitive, long-term stable, exploratory vs. focused).
                          - Do NOT output step-by-step analysis; keep reasoning concise.

                          Available tools:
                          - GLINTRU: strong general sequential filtering.
                          - SASRec: temporal / recent-sequence patterns.
                          - GRURec: noisy but order-informative sequences.
                          - STAMP: short-session / bursty intent.
                          - LightGCN: stable collaborative filtering.
                          - KGAT: relation/knowledge-aware filtering.

                          OUTPUT PROTOCOL:
                          You MUST output ONLY a valid JSON object:
                          {{
                            "tool_selected":  "glintru" | "sasrec" | "gru4rec" | "stamp" | "lightgcn" |"kgat",
                            "planner_intention": "1-2 movie genres",
                            "user_behavior_pattern": "Concise description of user behavior pattern",
                            "planner_reasoning": "Brief explanation of tool choice and intention. Max 60 words."
                          }}
                          """

    return planner_prompt
def reasoner(formatted_candidates, planner_intention, planner_reasoning, user_behavior_pattern, reasoner_memory,
             similar_memory=None, min_keep: int = 10):
    n_candidates = len(formatted_candidates.strip().split('\n')) if formatted_candidates else 0

    base_prompt = f"""You are a Recommendation Judge.

              GOAL:
              Keep filtering until the candidate list is small enough and better aligned with user intent and behavior.

              INPUT:
              - planner_intention: {planner_intention}          # 1-2 movie genres
              - user_behavior_pattern: {user_behavior_pattern}  # concise behavior description
              - planner_reasoning: {planner_reasoning}
              - similar_user_memory: {similar_memory}           # optional auxiliary info
              - n_candidates: {n_candidates}
              - min_keep: {min_keep}                            # maximum allowed size to stop filtering

              HARD RULE (MUST FOLLOW):
              - If n_candidates > min_keep:
                - judgment MUST be "invalid"
                - need_filter MUST be true
              - If n_candidates <= min_keep:
                - judgment MUST be "valid"
                - need_filter MUST be false

              ANALYSIS (ONLY if need_filter=true):
              - Using planner_intention + user_behavior_pattern + planner_reasoning,
                decide whether the candidate set contains off-intent or weakly aligned items.
              - Describe set-level issues (e.g., genre drift, mixed styles, off-intent tail),
                citing concrete Titles/Genres.

              FILTER TOOL GUIDANCE (ONLY if need_filter=true):
              - Based on user_behavior_pattern, describe the desired filtering tool behavior
                (e.g., session-driven, recency-sensitive, long-term stable, collaborative).

              MEMORY USAGE:
              - similar_user_memory is optional auxiliary context and must not override the hard rule.

              OUTPUT (STRICT JSON ONLY):
              {{
                "judgment": "valid" or "invalid",
                "confidence": 0.0-1.0,
                "need_filter": true or false,
                "reasoner_reasoning": "If need_filter=true: max 40 words; else empty string",
                "filter_tool_characteristics": "If need_filter=true: max 40 words; else empty string"
              }}
              """


    if len(reasoner_memory) > 0:
        base_prompt += f"\n- Reasoner Memory: {reasoner_memory}"

    return base_prompt

def reflector(
    filter_tool_characteristics,
    reasoner_reasoning,
    candidate_items_text,  
    user_profile,
    memory=None,
    need_filter=False,
    filter_reason="",
    min_keep=5,
    tools=6
):
    """
    - Count > 10: MUST tool, MUST output empty re_ranked_candidate_items []
    - Count <= 10: MAY llm/none
    - Output JSON does NOT include drop_ratio (drop ratio is fixed in code)
    - Encourage tool diversity
    """
    current_count = len(candidate_items_text.strip().split("\n")) if candidate_items_text else 0
    if tools == 6:
      return f"""You are a Recommendation Refiner.

            GOAL:
            - Produce a high-quality short list for the next stage.
            - Prefer tool diversity over repeatedly selecting the same tool, unless strongly justified.

            INPUT:
            - judge_reasoning: {reasoner_reasoning}
            - need_filter: {need_filter} ({filter_reason})
            - min_keep: {min_keep}

            - filter_tool_characteristics: {filter_tool_characteristics}

            - similar_user_memory: {memory}
              (Retrieved from the most similar user; may include:
              target item hints, user behavior/preferences summary, and best-performing tool.)

            TASK:
            1) Check candidate count ({current_count}).

            2) Decide filtering mode:

              [ITEM COUNT RULE – MUST FOLLOW]
              - If Count >= 10: filter_mode MUST be "tool".
                Do NOT output a re-ranked list.
              - If Count < 10: filter_mode can be "none".

            3) If filter_mode="tool":
              - Tool selection priority (STRICT):
                1. filter_tool_characteristics (PRIMARY signal: desired tool behavior).
                2. filter_reason (SECONDARY signal: what kind of misalignment exists).
                3. similar_user_memory (AUXILIARY only):
                  - user_summary → check behavior consistency.
                  - target_item_information → preserve genre/theme if relevant.
                  - best_tool → tie-breaker ONLY when multiple tools equally match.
              - Tool diversity applies only after the above priorities.

            4) If filter_mode="none" (Count <= 10):
              - There is no need to filter

            TOOL HEURISTICS:
            - GLINTRU: strong general sequential filtering.
            - SASRec: temporal / recent-sequence patterns.
            - GRURec: noisy but order-informative sequences.
            - STAMP: short-session / bursty intent.
            - LightGCN: stable collaborative filtering.
            - KGAT: relation/knowledge-aware filtering.
            

            OUTPUT (STRICT JSON):
            {{
              "filter_mode": "tool" | "llm" | "none",
              "filter_tool": "sasrec" | "gru4rec" | "lightgcn" | "stamp" | "glintru" | "kgat",
              "filter_plan_reason": "Explain briefly how filter_tool_characteristics guided the tool choice, how filter_reason supported it, and whether similar_user_memory affected the final decision.",
            }}
            """
    else:
      return f"""You are a Recommendation Refiner.

            GOAL:
            - Produce a high-quality short list for the next stage.
            - Select the most appropriate tool to filter candidate items.
            - Prefer tool diversity over repeatedly selecting the same tool, unless strongly justified.

            INPUT:
            - judge_reasoning: {reasoner_reasoning}
            - need_filter: {need_filter} ({filter_reason})
            - min_keep: {min_keep}

            - filter_tool_characteristics: {filter_tool_characteristics}

            - similar_user_memory: {memory}
              (Retrieved from the most similar user; may include:
              target item hints, user behavior/preferences summary, and best-performing tool.)

            - CANDIDATE ITEMS (Count: {current_count}):
            {candidate_items_text}

            TASK:
            1) Check candidate count ({current_count}).

            2) Decide filtering mode:

              [ITEM COUNT RULE – MUST FOLLOW]
              - If Count >= 10: filter_mode MUST be "tool".
                Do NOT output a re-ranked list.
              - If Count < 10: filter_mode can be "llm" or "none".

            3) If filter_mode="tool":
              - Set re_ranked_candidate_items to [].
              - Tool selection priority (STRICT):
                1. filter_tool_characteristics (PRIMARY signal: desired tool behavior).
                2. filter_reason (SECONDARY signal: what kind of misalignment exists).
                3. similar_user_memory (AUXILIARY only):
                  - user_summary → check behavior consistency.
                  - target_item_information → preserve genre/theme if relevant.
                  - best_tool → tie-breaker ONLY when multiple tools equally match.

            4) If filter_mode="llm" or "none" (Count <= 10):
              - Re-rank items by relevance.
              - similar_user_memory may be used only as weak tie-breaking guidance.

            TOOL HEURISTICS:
            - SASRec: temporal / recent-sequence patterns.
            - GRURec: noisy but order-informative sequences.
            - LightGCN: stable collaborative filtering.
            

            OUTPUT (STRICT JSON):
            {{
              "filter_mode": "tool" | "llm" | "none",
              "filter_tool": "sasrec" | "gru4rec" | "lightgcn",
              "filter_plan_reason": "Explain briefly how filter_tool_characteristics guided the tool choice, how filter_reason supported it, and whether similar_user_memory affected the final decision.",
              "re_ranked_candidate_items": [
                // ONLY if filter_mode is "llm" or "none" (AND Count <= 10)
                {{"id": "item_X", "movie_title": "Title", "release_year": 1999, "genre": "Genre"}}
              ]
            }}
            """

