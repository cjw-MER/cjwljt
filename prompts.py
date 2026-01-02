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
# 去除sasrec版本
# prompts.py

# prompts.py

def reflector(
    reasoner_reasoning,
    candidate_items_text,
    re_candidate_items_text,  # 兼容旧签名，可不使用
    user_profile,
    memory=None,
    need_filter=False,
    filter_reason="",
    min_keep=5,
    used_filter_tools=None,
):
    """
    - Count > 10: MUST tool, MUST output empty re_ranked_candidate_items []
    - Count <= 10: MAY llm/none
    - Output JSON does NOT include drop_ratio (drop ratio is fixed in code)
    - Encourage tool diversity
    """
    used_filter_tools = used_filter_tools or []
    current_count = len(candidate_items_text.strip().split("\n")) if candidate_items_text else 0

    return f"""You are a Recommendation Refiner.

GOAL:
- Produce a high-quality short list for the next stage.
- When using tools, prefer tool diversity over repeatedly selecting the same tool, unless there is a strong reason.

INPUT:
- judge_reasoning: {reasoner_reasoning}
- similar_user_memory: {memory}
- need_filter: {need_filter} ({filter_reason})
- min_keep: {min_keep}

- CANDIDATE ITEMS (Count: {current_count}):
{candidate_items_text}

- used_filter_tools (avoid these if possible): {used_filter_tools}

TASK:
1) Check the Candidate Items count ({current_count}).

2) Decide a filtering strategy (mode) based on the count:

   [CRITICAL RULE: ITEM COUNT CONSTRAINT]
   - IF Count > 10: You MUST select filter_mode="tool".
     The list is too long for manual re-ranking. Do NOT output a re-ranked list.
   - IF Count <= 10: You MAY select filter_mode="llm" or "none".
     You can manually re-rank items by relevance and fit.

3) If mode="tool":
   - Choose filter_tool NOT IN used_filter_tools whenever possible.
   - You MUST set re_ranked_candidate_items to [].

4) If mode="llm" or "none" (only allowed when Count <= 10):
   - Re-rank the items to put the best ones at the top.

TOOL DIVERSITY POLICY (IMPORTANT):
- Do NOT default to the same tool every time.
- Prefer diverse tool selection across calls/sessions when multiple tools are plausible.
- If two tools appear equally suitable, break ties by selecting a less recently used tool, and state this explicitly.

TOOL SELECTION HEURISTICS:
- SASRec: Strong for sequential/temporal consumption patterns (recent actions matter most).
- glintru: Strong sequential modeling; good general sequential recommender.
- gru4rec: Sequence-heavy; good when order is informative and histories are noisy.
- stamp: Session intent; good when short-burst behavior dominates.

OUTPUT (STRICT JSON):
{{
  "filter_mode": "tool" or "llm" or "none",
  "filter_tool": "gru4rec" or "lightgcn" or "stamp" or "glintru" or "kgat",
  "filter_plan_reason": "Explain why, mentioning: (1) item count constraint, (2) why this mode/tool fits judge_reasoning/user_history, (3) how tool diversity affected your choice.",
  "re_ranked_candidate_items": [
    // ONLY output this list if filter_mode is "llm" or "none" (AND Count <= 10).
    // If filter_mode is "tool", this array MUST be EMPTY [].
    {{"id": "item_X", "movie_title": "Title", "release_year": 1999, "genre": "Genre"}}
  ]
}}
"""


#这是调用三个工具的版本
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
# - SASRec: Strong for sequential/temporal consumption patterns (recent actions matter most).
# - GRURec: Also sequence-heavy; good when session order is informative and noisy histories exist.
# - LightGCN: Strong general-purpose collaborative filtering on user-item graph; use when you want stable global CF.

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

