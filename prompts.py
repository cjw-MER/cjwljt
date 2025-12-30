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
4. Output must be a valid JSON object containing exactly three keys: "tool_selected", "planner_intention", "planner_reasoning".

IMPORTANT:
- Tool choice MUST be based on FUNCTION, NOT list order.
- intention should predict composite genres the user is likely to watch next.
- Do NOT output step-by-step analysis; keep reasoning concise (max 60 words).

Available tools:
- SASRec: better for users with stable long-term preferences and clear viewing patterns
- GRURec: better for users with recent preference shifts or short-term focused behavior
- LightGCN: better for collaborative signals (optional)

OUTPUT PROTOCOL:
You MUST output ONLY a valid JSON object:
{{
"tool_selected": "SASRec" or "GRURec" or "LightGCN",
"planner_intention": "1-2 movie genres",
"planner_reasoning": "Brief explanation. Max 60 words."
}}
"""
    return planner_prompt
def reasoner(candidate_items_text, recommendation_intent, reasoning, reasoner_memory, user_profile,
             similar_memory=None, min_keep: int = 5):
    n_candidates = len(candidate_items_text.strip().split('\n')) if candidate_items_text else 0

    base_prompt = f"""You are a Recommendation Judge.

GOAL:
We want to keep filtering until the candidate list is small enough.

INPUT:
- recommendation_intent: {recommendation_intent}
- planner_reasoning: {reasoning}
- similar_user_memory: {similar_memory}
- n_candidates: {n_candidates}
- min_keep: {min_keep}

- CANDIDATE ITEMS (Ranked with Metadata):
{candidate_items_text}

HARD DECISION RULE (MUST FOLLOW):
- If n_candidates > min_keep:
  - You MUST output need_filter=true.
  - judgment MUST be "invalid" (because the list is not finalized yet).
- If n_candidates <= min_keep:
  - You MUST output need_filter=false.
  - judgment MUST be "valid".

FILTER REASON REQUIREMENT:
- When need_filter=true, `filter_reason` MUST describe the characteristics of the CURRENT candidate set
  (e.g., genre drift head->tail, mixed unrelated genres, scattered years, off-intent tail items),
  and cite concrete evidence using specific Titles/Genres from the list.
- `filter_reason` max 40 words.

OUTPUT (STRICT JSON):
{{
  "judgment": "valid" or "invalid",
  "confidence": 0.0-1.0,
  "reasoner_reasoning": "Brief reasoning referencing specific movies/genres",
  "need_filter": true or false,
  "filter_reason": "If need_filter=true, max 40 words, must describe candidate set characteristics."
}}
"""
    if len(reasoner_memory) > 0:
        base_prompt += f"\n- Reasoner Memory: {reasoner_memory}"

    return base_prompt


# def reasoner(candidate_items_text, recommendation_intent, reasoning, reasoner_memory, user_profile, similar_memory=None):
#     """
#     candidate_items_text: formatted string containing ID, Title, Year, Genre (one item per line).
#     """
#     n_candidates = len(candidate_items_text.strip().split('\n')) if candidate_items_text else 0

#     # 控制倾向：候选越长越容易混入噪声，所以 >=10 直接要求偏向过滤
#     length_bias_note = ""
#     if n_candidates >= 10:
#         length_bias_note = (
#             "Length-Bias Rule: n_candidates >= 10. "
#             "When judgment is uncertain, prefer need_filter=true. "
#             "Only set need_filter=false if the list is strongly coherent in genre/topic and matches intent throughout.\n"
#         )

#     base_prompt = f"""You are a Recommendation Judge.

# Task:
# 1) Analyze the candidate list content (Titles, Genres, Years) against the User History and Intent.
# 2) Judge whether the CURRENT ranked candidate list is semantically valid (relevant to user) or needs reordering.
# 3) Decide if the list contains "noise" (irrelevant genres, inconsistent topics) that should be filtered.

# INPUT:

# - recommendation_intent: {recommendation_intent}
# - planner_reasoning: {reasoning}
# - similar_user_memory: {similar_memory}
# - n_candidates: {n_candidates}

# - CANDIDATE ITEMS (Ranked with Metadata):
# {candidate_items_text}

# RULES (IMPORTANT):
# - Look at the GENRES and TITLES. Do they match the `recommendation_intent`?
# - If the tail items (bottom of the list) drift into unrelated genres compared to the head, set need_filter=true.
# - {length_bias_note.strip()}

# FILTERING DECISION GUIDELINES:
# - need_filter=true if you observe any of:
#   - Genre/topic drift from head -> tail.
#   - Mixed unrelated genres that conflict with intent.
#   - Many items look generic, off-topic, or inconsistent in theme.
#   - Year distribution is inconsistent with intent (e.g., intent wants classics but list is mostly recent, or vice versa).
# - If n_candidates >= 10, be conservative: filtering is preferred unless coherence is very strong.

# FILTER REASON REQUIREMENT (VERY IMPORTANT):
# - If need_filter=true, `filter_reason` MUST describe the *characteristics of the current candidate set* (not generic advice),
#   such as: "tail drifts from Action/Sci-Fi into Romance/Drama", "genres are highly mixed", "theme inconsistency",
#   "year range is scattered and mismatched with intent", etc.
# - Mention concrete evidence: cite specific Movie Titles and/or Genres found in the list.
# - `filter_reason` max 40 words.

# OUTPUT (STRICT JSON):
# {{
#   "judgment": "valid" or "invalid",
#   "confidence": 0.0-1.0,
#   "reasoner_reasoning": "Brief reasoning referencing specific movies/genres",
#   "need_filter": true or false,
#   "filter_reason": "If need_filter=true, why filtering is needed (max 40 words, must describe candidate set characteristics)."
# }}
# """

#     if len(reasoner_memory) > 0:
#         base_prompt += f"\n- Reasoner Memory: {reasoner_memory}"

#     return base_prompt

# def reflector(reasoner_reasoning, candidate_items_text, re_candidate_items_text, user_profile, memory=None, need_filter=False, filter_reason="", min_keep=5,used_filter_tools=None):
#     """
#     [Modified] Imposes a strict limit (<= 10 items) for LLM manual re-ranking.
#     Encourages diverse tool usage when filter_mode="tool".
#     """

#     current_count = len(candidate_items_text.strip().split('\n')) if candidate_items_text else 0

#     return f"""You are a Recommendation Refiner.

# GOAL:
# - Produce a high-quality short list for the next stage.
# - When using tools, prefer *tool diversity* over repeatedly selecting the same tool, unless there is a strong reason.

# INPUT:

# - judge_reasoning: {reasoner_reasoning}
# - similar_user_memory: {memory}
# - need_filter: {need_filter} ({filter_reason})
# - min_keep: {min_keep}

# - CANDIDATE ITEMS (Count: {current_count}):
# {candidate_items_text}

# - PREVIOUS RE-RANKED (if any):
# {re_candidate_items_text}
# - used_filter_tools (do NOT choose these if possible): {used_filter_tools}

# TASK:
# 1) Check the Candidate Items count ({current_count}).

# 2) Decide on a filtering strategy (mode) based on the count:

#    [CRITICAL RULE: ITEM COUNT CONSTRAINT]
#    - IF Count > 10: You MUST select filter_mode="tool".
#      The list is too long for manual re-ranking. Do NOT output a re-ranked list.
#    - IF Count <= 10: You MAY select filter_mode="llm" or "none".
#      You can manually re-rank items by relevance and fit.

# 3) If mode="tool": Choose the tool and drop_ratio (e.g., 0.1~0.5), choose filter_tool NOT IN used_filter_tools whenever possible.
#    If mode="llm": Re-rank the items to put the best ones at the top.

# TOOL DIVERSITY POLICY (IMPORTANT):
# - Do NOT default to the same tool every time.
# - Prefer diverse tool selection across calls/sessions when multiple tools are plausible.
# - If similar_user_memory / previous outputs suggest a recently used tool, prefer a different tool unless that tool is clearly best for the current situation.
# - When two tools appear equally suitable, break ties by selecting the less recently used (or a different one than you picked last time), and state this explicitly in filter_plan_reason.

# TOOL SELECTION HEURISTICS (use these + judge_reasoning):
# - GLINTRU: As a newly released sequential recommendation model, it delivers strong sequential modeling and recommendation capabilities.
# - SASRec: Strong for sequential/temporal consumption patterns (recent actions matter most).
# - GRURec: Also sequence-heavy; good when session order is informative and noisy histories exist.
# - STAMP: Session-based intent; good when user behavior is short-burst or “current session” dominates.
# - LightGCN: Strong general-purpose collaborative filtering on user-item graph; use when you want stable global CF.
# - KGAT: Best when item knowledge graph / rich relations matter (genre/actors/franchise links, explainable relations).


# DROP RATIO GUIDANCE (keep min_keep in mind):
# - If Count is moderately above 10 (e.g., 11~20): drop_ratio around 0.1~0.3.
# - If Count is large: drop_ratio around 0.3~0.5.
# - Ensure the remaining items can still cover user intent; do not over-drop niche but relevant items.

# OUTPUT (STRICT JSON):
# {{
#     "filter_mode": "tool" or "llm" or "none",
#     "filter_tool": "SASRec" or "GRURec" or "LightGCN" or "STAMP" or "GLINTRU" or "KGAT" (only if mode=tool),
#     "drop_ratio": 0.0-1.0,
#     "filter_plan_reason": "Explain why, mentioning: (1) item count constraint, (2) why this tool fits judge_reasoning/user_history, (3) how tool diversity affected your choice (e.g., tie-break / rotation).",
#     "re_ranked_candidate_items": [
#         // ONLY output this list if filter_mode is "llm" or "none" (AND Count <= 10).
#         // If filter_mode is "tool", leave this array EMPTY [].
#         {{"id": "item_X", "movie_title": "Title", "release_year": 1999, "genre": "Genre"}}
#     ]
# }}
# """


#这是调用三个工具的版本
def reflector(reasoner_reasoning, candidate_items_text, re_candidate_items_text, user_profile, memory=None, need_filter=False, filter_reason="", min_keep=5,used_filter_tools=None):
    """
    [Modified] Imposes a strict limit (<= 10 items) for LLM manual re-ranking.
    Encourages diverse tool usage when filter_mode="tool".
    """

    current_count = len(candidate_items_text.strip().split('\n')) if candidate_items_text else 0

    return f"""You are a Recommendation Refiner.

GOAL:
- Produce a high-quality short list for the next stage.
- When using tools, prefer *tool diversity* over repeatedly selecting the same tool, unless there is a strong reason.

INPUT:

- judge_reasoning: {reasoner_reasoning}
- similar_user_memory: {memory}
- need_filter: {need_filter} ({filter_reason})
- min_keep: {min_keep}

- CANDIDATE ITEMS (Count: {current_count}):
{candidate_items_text}

- PREVIOUS RE-RANKED (if any):
{re_candidate_items_text}
- used_filter_tools (do NOT choose these if possible): {used_filter_tools}

TASK:
1) Check the Candidate Items count ({current_count}).

2) Decide on a filtering strategy (mode) based on the count:

   [CRITICAL RULE: ITEM COUNT CONSTRAINT]
   - IF Count > 10: You MUST select filter_mode="tool".
     The list is too long for manual re-ranking. Do NOT output a re-ranked list.
   - IF Count <= 10: You MAY select filter_mode="llm" or "none".
     You can manually re-rank items by relevance and fit.

3) If mode="tool": Choose the tool and drop_ratio (e.g., 0.1~0.5), choose filter_tool NOT IN used_filter_tools whenever possible.
   If mode="llm": Re-rank the items to put the best ones at the top.

TOOL DIVERSITY POLICY (IMPORTANT):
- Do NOT default to the same tool every time.
- Prefer diverse tool selection across calls/sessions when multiple tools are plausible.
- If similar_user_memory / previous outputs suggest a recently used tool, prefer a different tool unless that tool is clearly best for the current situation.
- When two tools appear equally suitable, break ties by selecting the less recently used (or a different one than you picked last time), and state this explicitly in filter_plan_reason.

TOOL SELECTION HEURISTICS (use these + judge_reasoning):
- SASRec: Strong for sequential/temporal consumption patterns (recent actions matter most).
- GRURec: Also sequence-heavy; good when session order is informative and noisy histories exist.
- LightGCN: Strong general-purpose collaborative filtering on user-item graph; use when you want stable global CF.

DROP RATIO GUIDANCE (keep min_keep in mind):
- If Count is moderately above 10 (e.g., 11~20): drop_ratio around 0.1~0.3.
- If Count is large: drop_ratio around 0.3~0.5.
- Ensure the remaining items can still cover user intent; do not over-drop niche but relevant items.

OUTPUT (STRICT JSON):
{{
    "filter_mode": "tool" or "llm" or "none",
    "filter_tool": "SASRec" or "GRURec" or "LightGCN" or "STAMP" or "GLINTRU" or "KGAT" (only if mode=tool),
    "drop_ratio": 0.0-1.0,
    "filter_plan_reason": "Explain why, mentioning: (1) item count constraint, (2) why this tool fits judge_reasoning/user_history, (3) how tool diversity affected your choice (e.g., tie-break / rotation).",
    "re_ranked_candidate_items": [
        // ONLY output this list if filter_mode is "llm" or "none" (AND Count <= 10).
        // If filter_mode is "tool", leave this array EMPTY [].
        {{"id": "item_X", "movie_title": "Title", "release_year": 1999, "genre": "Genre"}}
    ]
}}
"""

# prompts.py

# def reflector(reasoner_reasoning, candidate_items_text, re_candidate_items_text, user_profile, memory=None, need_filter=False, filter_reason="", min_keep=5):
#     """
#     [Modified] Instructions updated to skip list generation when using tool filtering.
#     """
#     return f"""You are a Recommendation Refiner.

# INPUT:
# - user_history: {user_profile}
# - judge_reasoning: {reasoner_reasoning}
# - similar_user_memory: {memory}
# - need_filter: {need_filter} ({filter_reason})
# - min_keep: {min_keep}

# - CANDIDATE ITEMS (Current List):
# {candidate_items_text}

# - PREVIOUS RE-RANKED (if any):
# {re_candidate_items_text}

# TASK:
# 1) Decide on a filtering strategy (mode):
#    - "tool": Use if the list is LONG or needs massive filtering based on a model score. **DO NOT output the re-ranked list manually in this mode.**
#    - "llm": Use if the list is SHORT (e.g. < 20 items) and you can manually pick the best ones.
#    - "none": Use if the list is already good.

# 2) If mode="llm" or "none": Reorder the items based on content relevance.
# 3) If mode="tool": Choose the tool and drop_ratio. The system will handle the filtering.

# OUTPUT (STRICT JSON):
# {{
#     "filter_mode": "tool" or "llm" or "none",
#     "filter_tool": "SASRec" or "GRURec" or "LightGCN" or "STAMP" or "GLINTRU" or "KGAT" (only if mode=tool),
#     "drop_ratio": 0.0-1.0 (if mode=tool/llm),
#     "filter_plan_reason": "why this mode and ratio",
#     "re_ranked_candidate_items": [
#         // ONLY output this list if filter_mode is "llm" or "none".
#         // If filter_mode is "tool", leave this array EMPTY [].
#         {{"id": "item_X", "movie_title": "Title", "release_year": 1999, "genre": "Genre"}}
#     ]
# }}
# """




# def planner(user_profile, memory=None):

#     planner_prompt = f"""Your task is to analyze the provided user interaction history and an external similar-user hint, then plan the recommendation workflow.

#                     INPUT:
#                     - user_history: {user_profile}
#                     - similar_user_hint: {memory}

#                     You MUST:
#                     1. Examine the user interaction history for viewing patterns and preferences.
#                     2. Use similar_user_hint as auxiliary evidence (NOT ground truth) when tool choice is ambiguous.
#                     3. Choose ONE recommendation tool based on the actual USER BEHAVIOR CHARACTERISTICS.
#                     4. Output must be a valid JSON object containing exactly three keys: "tool_selected", "planner_intention", "planner_reasoning".

#                     IMPORTANT:
#                     - Tool choice MUST be based on FUNCTION, NOT list order. The tools are listed in arbitrary order.
#                     - You MUST analyze user history to infer composite movie genres for next recommendation.
#                     - intention should predict composite genres the user is likely to watch next.
#                     - Do NOT output step-by-step analysis; keep reasoning concise (max 60 words).

#                     Available tools (arbitrary order):
#                     - SASRec: better for users with stable long-term preferences and clear viewing patterns
#                     - GRURec: better for users with recent preference shifts or short-term focused behavior

#                     Decision Guide:
#                     - If user shows consistent genre preferences over long history -> use SASRec
#                     - If user has recent viewing pattern changes -> use GRURec
#                     - If user preferences are complex/mixed -> use SASRec for better pattern recognition

#                     Similar-User Guidance (STRICT):
#                     - Only use similar_user_hint to break ties when behavior evidence from user_history is weak/ambiguous.
#                     - If similar_user_hint[0] is "SASRec" or "GRURec" or LightGCN", you MAY follow it in tie cases.
#                     - target_position_in_candidate_set: smaller is better (closer to 1 indicates that tool worked well for the similar user).
#                     - If similar_user_hint is missing/invalid, ignore it.

#                     ────────────── OUTPUT PROTOCOL (STRICT) ────────────────
#                     You MUST output ONLY a valid JSON object with the following structure:
#                     {{
#                     tool_selected: SASRec or GRURec or LightGCN
#                     planner_intention: 1-2 movie genres for user's next watch
#                     planner_reasoning: Brief explanation of given genres (max 100 words)
#                     }}
#                     """


#     return planner_prompt

# def reasoner(candidate_items, recommendation_intent, reasoning, reasoner_memory, similar_memory=None):
#     if len(reasoner_memory)<1:
#         reasoner_prompt = f"""Judge whether the **top 5 items** in the current candidate_list are reasonably ordered based on the provided intent, reasoning, and similar user's memory.

#                             INPUT:
#                             - candidate_list (full): {candidate_items}
#                             - recommendation_intent: {recommendation_intent}
#                             - planner_reasoning: {reasoning}
#                             - similar_user_memory: {similar_memory}  # Contains: 1) target item hints, 2) preference summary, 3) ranking suggestions/experience.

#                             CRITICAL RULES:
#                             1) **First, Assess Memory Utility**: Before judging the list, evaluate if `similar_user_memory` is **relevant and useful** for the *current user's* `recommendation_intent`.
#                             - **VALID Memory**: It is valid **only if** its content (target hints, preference, suggestions) clearly aligns with or informs the current `recommendation_intent` and `planner_reasoning`. If it's generic, outdated, or aimed at a different goal, it is **INVALID**.
#                             - **Usage**: If memory is **VALID**, use it as an **auxiliary signal** (Rule 2). If **INVALID**, **disregard it completely** and judge ordering based solely on `recommendation_intent` and `planner_reasoning`.

#                             2) **Primary Ordering Driver**: The top 10 items should collectively best satisfy the *recommendation_intent* as interpreted by the *planner_reasoning*. The **very first item** is the most critical.

#                             3) **Memory as Conditional Guidance**: **ONLY IF** memory was judged VALID in Rule 1, use it to resolve ambiguities:
#                             - **Target Alignment**: Check if top items align with the similar user's “target item” hints.
#                             - **Preference Consistency**: Check if top items reflect the similar user's “preference summary”.
#                             - **Experience Heuristic**: Consider the similar user's “ranking suggestions” as experienced advice.

#                             4) **Invalid Judgment**: If the top-10 order is suboptimal relative to the applicable signals (intent+reasoning ± valid memory), it is "invalid". You MUST then provide a concise reorder instruction **using only items from the full candidate_list**.

#                             ─────────────── OUTPUT PROTOCOL (STRICT) ────────────────
#                             You MUST output **ONLY** a valid JSON object with this **exact** structure:
#                             {{
#                             judgment: valid or invalid
#                             confidence: float between 0.0 and 1.0
#                             reasoner_reasoning: Brief summary explanation for reordering if needed to better match user preferences.
#                             }}
#                         """
#     else:
#         reasoner_prompt = f"""You are a Recommendation Judge.

#                         Task: Judge whether the CURRENT ranked candidate list should be kept (valid) or reordered (invalid), using Reasoner Memory and the current candidates, and reflecting on whether the prior reasoning was correct.

#                         INPUT:
#                         - Candidate Items: {candidate_items}
#                         - Reasoner Memory: {reasoner_memory}

#                         CORE RULES (MUST FOLLOW):
#                         1) Learn-from-Memory first (MUST):
#                         - Read Reasoner Memory and extract 1-3 actionable ranking heuristics (experience rules).
#                         - Each heuristic must be directly usable for reordering the CURRENT candidate list.
#                         - If a heuristic is outdated/contradicted by the CURRENT candidates or lacks supporting evidence, discard it.
#                         2) Reorder principle:
#                         - If Reasoner Memory provides clear evidence that some items are already interacted, push those items toward the END.
#                         - Push items implied as more relevant by Reasoner Memory toward the FRONT.
#                         3) Use ONLY provided inputs. Do not add assumptions.
#                         4) If you cannot justify a reorder from the inputs, keep the list as valid.

#                         ────────────── OUTPUT PROTOCOL (STRICT) ────────────────
#                         You MUST output **NOTHING** except the pure, valid JSON object. No introductory text, no explanations, no markdown code block fences (like ```json)
#                         {{
#                         judgment: valid or invalid
#                         confidence: float value between 0.0 and 1.0
#                         reasoner_reasoning: Brief reasoning for valid or invalid
#                         }}
#                         """

#     return reasoner_prompt

# def reflector(reasoner_reasoning, candidate_items, re_candidate_items, memory=None):
#     reflector_prompt = f"""
#             You are a Recommendation Refiner.
#             Task: REORDER items based on judge_reasoning, using similar_user_memory as optional guidance.

#             INPUT:
#             - candidate_items: {candidate_items}
#             - previous_reranked_items: {re_candidate_items}
#             - judge_reasoning: {reasoner_reasoning}
#             - similar_user_memory: {memory}

#             RULES:
#             1) If judge_reasoning says the ranking is reasonable / no changes needed -> keep previous_reranked_items (after cleaning) and stop.
#             2) Otherwise -> start from previous_reranked_items (cleaned), make minimal swaps to better satisfy:
#             (a) judge_reasoning (highest priority),
#             (b) similar_user_memory (tie-breaker: target hints / preference summary /experience) if similar_user_memory is not None
#             3) HARD CONSTRAINTS:
#             - Only reorder; do not add/remove/modify items.
#             - Output must contain exactly the items in candidate_items, each exactly once (no duplicates).
#             - Ignore any item in previous_reranked_items that is not in candidate_items.
#             - If some items in candidate_items are missing from previous_reranked_items, insert them where best matches judge_reasoning/memory; otherwise append to the end.

#             ────────────── OUTPUT PROTOCOL (STRICT) ────────────────
#             You MUST output **NOTHING** except the pure, valid JSON object. No introductory text, no explanations, no markdown code block fences (like ```json). The regenerated re_ranked_candidate_items must include only items that are present in the candidate_items.
#             {{
#             re_ranked_candidate_items: ["same items as candidate_items, reordered. In the form of a string"]
#             }}
#             """


#     return reflector_prompt


# def summary(reasoner_reasoning):

#     summary_prompt = f"""
#             You are a ranking Session Summarizer and Reflector.

#             INPUT:
#             - ranking_session: {reasoner_reasoning}

#             TASK:
#             Produce ONE concise summary (<= 80 words) of the whole re-ranking session that:
#             1) States which rounds were wrong vs. possibly right, grounded in NDCG changes.
#             2) Reflects on what went wrong/right in the ranking logic (e.g., interacted-item handling, genre-based heuristics).
#             3) Gives 1–2 transferable takeaways to improve ranking metrics for similar users (e.g., when to keep interacted items, how to apply genre signals).

#             RULES:
#             - Output ONLY the summary text (no JSON, no bullets, no extra fields).
#             - Must mention the NDCG trend (e.g., 1.0→0.43→0.30→unchanged).
#             - Do not invent facts beyond judge_reasoning.
#             - Keep advice actionable and generalizable to similar users.

#             ────────────── OUTPUT PROTOCOL (STRICT) ────────────────
#             You MUST output ONLY the following one line. Any additional text, explanation, markdown, or formatting will be considered INVALID and discarded. Do not generate the reasoning process.

#             session_summary: <summary (<= 80 words)>
#             """

#     return summary_prompt