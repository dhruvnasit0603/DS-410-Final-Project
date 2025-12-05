import os
import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

INPUT_CSV = Path("tweets_cleaned.csv")
OUTPUT_CSV = Path("tweets_with_topics_v1.csv")

# GPT model name
GPT_MODEL = "gpt-4.1-mini"


# Batch size: number of tweets per API call
BATCH_SIZE = 25


# OpenAI API key
client = OpenAI(api_key="")

# CATEGORY DEFINITIONS


TOPIC_CATEGORIES = [
    {
        "id": "macro",
        "name": "Macroeconomics & Monetary Policies",
        "description": (
            "Tweets referring to broad economic conditions, growth, "
            "unemployment, inflation, GDP, or the Federal Reserve."
        ),
    },
    {
        "id": "trade",
        "name": "Trade Policy & Industrial / Manufacturing",
        "description": (
            "Trade disputes, tariffs, manufacturing policies, "
            "import/export, China trade, supply chains."
        ),
    },
    {
        "id": "energy",
        "name": "Energy, Oil & Gas, Renewables",
        "description": (
            "Energy independence, oil prices, fracking, pipelines, "
            "OPEC references, renewable energy comments."
        ),
    },
    {
        "id": "defense",
        "name": "Defense, Military, Sanctions, Geopolitics",
        "description": (
            "Foreign policy, military deployments, NATO, sanctions, "
            "conflicts, national security."
        ),
    },
    {
        "id": "regulation",
        "name": "Regulation, Antitrust, Legal Actions",
        "description": (
            "Regulatory actions, investigations, bans, tax policy, "
            "antitrust issues, legal threats."
        ),
    },
    {
        "id": "campaign",
        "name": "Campaign / Rally / Election Politics",
        "description": (
            "Campaign events, rallies, slogans, partisan attacks, "
            "polling, endorsements."
        ),
    },
    {
        "id": "social",
        "name": "Personal, Social, or Non-Policy Content",
        "description": (
            "Personal remarks, greetings, daily life, sports, "
            "non-economic commentary."
        ),
    },
    {
        "id": "uncategorized",
        "name": "Uncategorized",
        "description": "Anything that does not clearly fit the above categories.",
    },
]

PROMPT_VERSION = "v1_topic8"  # just for reference 


def _build_category_text() -> str:
    """
    Build a human-readable description of all categories for the prompt.
    """
    lines: list[str] = []
    for cat in TOPIC_CATEGORIES:
        line = f"- ({cat['id']}) {cat['name']}: {cat['description']}"
        lines.append(line)
    return "\n".join(lines)


CATEGORY_TEXT = _build_category_text()


# ============================
# PROMPT BUILDING 
# ============================

def build_system_message() -> str:
    """
    System message: explains the task and required JSON output format.
    """
    return f"""
You are a classification engine. Your job is to assign EXACTLY ONE topic category
to each tweet based on its text.

Use the following categories:

{CATEGORY_TEXT}

For each tweet, you MUST choose exactly one of the category NAMES above.
Do NOT invent new categories.

Output STRICTLY in JSON list format, one object per tweet, like:

[
  {{ "id": "123", "category": "Macroeconomics & Monetary Policies" }},
  {{ "id": "456", "category": "Personal, Social, or Non-Policy Content" }}
]

Rules:
- Use the category NAME exactly as written above (case-sensitive).
- Ensure the JSON is valid.
- The order of objects in the list must match the order of tweets given.

---
System version: {PROMPT_VERSION}
"""


def build_user_message(tweets_batch: list[dict]) -> str:
    """
    User message: contains the actual tweets to classify.
    """
    lines: list[str] = ["Here are the tweets to classify:"]

    for i, tw in enumerate(tweets_batch, start=1):
        tweet_text = (tw.get("text") or "").replace("\n", " ")
        lines.append(f"\nTweet {i}:")
        lines.append(f"ID: {tw['id']}")
        lines.append(f"Text: {tweet_text}")

    return "\n".join(lines)


# ============================
# GPT CALL 
# ============================

def classify_batch_with_gpt(tweets_batch: list[dict]) -> dict:
    """
    Call GPT to classify a batch of tweets.

    Returns:
        A dict mapping tweet_id -> category_name.
    """
    system_message = build_system_message()
    user_message = build_user_message(tweets_batch)

    # Prepare messages for Chat Completions API
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # Simple retry loop in case of transient errors 
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0.0,
            )
            content = response.choices[0].message.content

            # Parse JSON output 
            data = json.loads(content)

            # Expecting a list of objects: [{"id": "...", "category": "..."}, ...]
            id_to_category = {}
            for item in data:
                tid = str(item["id"])
                cat_name = str(item["category"])
                id_to_category[tid] = cat_name

            return id_to_category

        except Exception as e:
            print(f"[WARN] GPT call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2)  # brief backoff 

    # Should not reach here normally 
    return {}


# ============================
# MAIN SCRIPT 
# ============================

def main():
    # 1) Load dataset 
    print(f"Loading tweets from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Ensure 'id' and 'text' exist 
    required_cols = ["id", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert id to string to avoid issues / 
    df["id"] = df["id"].astype(str)

    # 2) Initialize category column 
    if "category" not in df.columns:
        df["category"] = None  # fill later 

    # 3) Build list of all tweet dictionaries 
    tweets = [
        {"id": row["id"], "text": row["text"]}
        for _, row in df.iterrows()
    ]

    total = len(tweets)
    print(f"Total tweets to classify: {total}")

    # 4) Process in batches 
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = tweets[start:end]

        print(f"Classifying batch {start}–{end-1} (size={len(batch)})...")

        # Call GPT for this batch 
        id_to_category = classify_batch_with_gpt(batch)

        # 5) Write results back into DataFrame 
        for tw in batch:
            tid = str(tw["id"])
            if tid in id_to_category:
                df.loc[df["id"] == tid, "category"] = id_to_category[tid]

        # Optional: save intermediate result every few batches
        if (start // BATCH_SIZE) % 20 == 0:  # every 20 batches
            tmp_path = OUTPUT_CSV.with_suffix(".tmp.csv")
            print(f"Saving intermediate result to {tmp_path} ...")
            df.to_csv(tmp_path, index=False)

    # 6) Save final CSV / 保存最终 CSV
    print(f"Saving final labeled tweets to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()