# prompts.py

def planner(user_profile, dataset, memory=None):

    if dataset == 'ml-1m':
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
                            "tool_selected": "sasrec" | "grurec" | "lightgcn" | "stamp" | "glintru" | "kgat",
                            "planner_intention": "1-2 categories",
                            "user_behavior_pattern": "Concise description of user behavior pattern",
                            "planner_reasoning": "Brief explanation of tool choice and intention. Max 60 words."
                          }}
                          """

    elif dataset == 'yelp':
      planner_prompt = f"""Your task is to analyze a Yelp user's interaction history and a similar-user hint,
        then infer the most likely attributes of the next business the user will visit,
        and select an appropriate recommendation model.

        INPUT:
        - user_history: {user_profile}
        - similar_user_hint: {memory}

        You MUST:
        1. Analyze the user's historical interactions to infer preference patterns.
        2. Predict the MOST LIKELY NEXT business attributes the user will visit, including:
           - city
           - categories
           - stars (rating range or tendency)
        3. Use similar_user_hint ONLY when the model choice is ambiguous.
        4. Select ONE recommendation tool based on observed user behavior characteristics.
        5. Explicitly summarize the user's interaction / decision behavior pattern
           to support downstream tool invocation.
        6. Provide concise reasoning connecting user behavior to both
           (a) business attribute prediction and
           (b) tool selection.

        IMPORTANT:
        - Tool choice must be based on functional suitability, NOT tool order.
        - planner_intention MUST describe business attribute preferences, not abstract interests.
        - user_behavior_pattern should characterize interaction style
          (e.g., exploratory, periodic, multi-interest, session-based).
        - Keep reasoning concise. No step-by-step chain-of-thought.

        Available tools:
        - glintru: strong general sequential filtering.
        - sasrec: temporal / recent-sequence patterns.
        - grurec: noisy but order-informative sequences.
        - stamp: short-session / bursty intent.
        - lightgcn: stable collaborative filtering.
        - kgat: relation/knowledge-aware filtering.
        - fmlprec: long-range periodic patterns via frequency-domain filtering.
        - difsr: disentangled multi-interest modeling under sequential dynamics.

        OUTPUT (JSON only, EXACTLY four keys):
        {{
          "tool_selected": " ",
          "planner_intention": " more than 80 words",
          "user_behavior_pattern": "more than 80 words ",
          "planner_reasoning": "more than 80 words "
        }}
        """


    else:
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
                      "tool_selected": "sasrec" | "grurec" | "lightgcn" | "stamp" | "glintru" | "kgat",
                      "planner_intention": "1-2 categories",
                      "user_behavior_pattern": "Concise description of user behavior pattern",
                      "planner_reasoning": "Brief explanation of tool choice and intention. Max 60 words."
                    }}
                    """

    return planner_prompt

def reasoner(formatted_candidates, planner_intention, planner_reasoning, user_behavior_pattern, reasoner_memory,
             similar_memory=None, min_keep=5):
    n_candidates = len(formatted_candidates.strip().split('\n')) if formatted_candidates else 0

    base_prompt = f"""You are a Recommendation Judge.

              GOAL:
              Keep filtering until the candidate_list is small enough and better aligned with user intent and behavior.

              INPUT:
              - planner_intention: {planner_intention}         
              - user_behavior_pattern: {user_behavior_pattern}  # concise behavior description
              - planner_reasoning: {planner_reasoning}
              - similar_user_memory: {similar_memory}           # optional auxiliary info
              - n_candidates: {n_candidates}
              - min_keep: {min_keep}                            # maximum allowed size to stop filtering                        

              HARD RULE (MUST FOLLOW):
              - If n_candidates > min_keep:
                - judgment = invalid
                - need_filter = true
                - Based on user_behavior_pattern + planner_reasoning, describe the desired filtering tool behavior
                - similar_user_memory is optional auxiliary context for filtering tool.

              - If n_candidates < min_keep:
                - pleanse analyze whether the current candidate list {formatted_candidates} aligns with the user's preferences. 
                - If it aligns: 
                  - judgment = valid
                  - need_filter = false
                  - reasoner_reasoning is empty string
                  - filter_tool_characteristics is empty string
                - If it does not align:
                  - judgment = invalid
                  - need_filter = true
                  - Based on user_behavior_pattern + planner_reasoning, describe the desired filtering tool behavior
                  - similar_user_memory is optional auxiliary context for filtering tool.

              OUTPUT (STRICT JSON ONLY):
              {{
                "judgment": "valid" or "invalid",
                "confidence": 0.0-1.0,
                "need_filter": true or false,
                "reasoner_reasoning": " ",
                "filter_tool_characteristics": " "
              }}
              """


    if len(reasoner_memory) > 0:
        base_prompt += f"\n- Reasoner Memory: {reasoner_memory}"

    return base_prompt

def reflector(filter_tool_characteristics,reasoner_reasoning, memory=None):
    """
    - Count > 10: MUST tool, MUST output empty re_ranked_candidate_items []
    - Count <= 10: MAY llm/none
    - Output JSON does NOT include drop_ratio (drop ratio is fixed in code)
    - Encourage tool diversity
    """

    return f"""You are a Recommendation Reflector.

          GOAL:
          Select the most appropriate recommendation tool based primarily on
          abstract filtering requirements inferred by an upstream judge,
          with similar-user memory used only as auxiliary support.

          INPUT:
          - filter_tool_characteristics: {filter_tool_characteristics}
          - judge_reasoning: {reasoner_reasoning}
          - similar_user_memory: {memory}


          TASK:
          1. Use filter_tool_characteristics as the PRIMARY basis for tool selection.
          2. Use judge_reasoning to refine or disambiguate the choice when needed.
          3. Consult similar_user_memory ONLY as auxiliary context:
             - to check consistency with observed behavior, or
             - as a tie-breaker when multiple tools equally match.
          4. Select ONE recommendation tool whose inductive bias best matches
             the required filtering behavior.

          TOOL HEURISTICS:
          - glintru: strong general sequential filtering.
          - sasrec: temporal / recent-sequence patterns.
          - grurec: noisy but order-informative sequences.
          - stamp: short-session / bursty intent.
          - lightgcn: stable collaborative filtering.
          - kgat: relation/knowledge-aware filtering.
          - fmlprec: long-range periodic patterns via frequency-domain filtering.
          - difsr: disentangled multi-interest modeling under sequential dynamics.

          OUTPUT (STRICT JSON ONLY):
          {{
            "filter_tool": "sasrec" | "grurec" | "lightgcn" | "stamp" | "glintru" | "kgat" | "fmlprec" | "difsr",
            "selected_reasoning": "Concise explanation linking filter_tool_characteristics, judge_reasoning, and (if applicable) similar_user_memory to the selected tool."
          }}
          """



def summary(user_profile):
  
  summary_prompt = f"""
      You are an expert in user behavior analysis for recommender systems.

      Given a single user's interaction history from the Yelp dataset, analyze and summarize the user's behavioral characteristics and preference patterns.

      Input:
      A chronological list of user interactions. Each interaction may include:
      - Business ID: a unique identifier of the business the user interacted with
      - Rating (1–5): the user's explicit feedback score for the business
      - Business categories: one or more category labels describing the business
      - City information: the city where the business is located
      {user_profile}

      Task: Summary
      Based only on the user's historical interactions, provide a concise high-level summary of the user's dominant behavior pattern and preferences in 3–5 bullet points.

      Output Format:
      Return the result strictly in the following JSON format. Do not include any additional text.

      {{
        "summary": "The total length of the summary must be no more than 80 words"
      }}
          """
  return summary_prompt
