import os
import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ============================
# ============================

INPUT_CSV = Path("tweets_with_topics_v1.csv")
OUTPUT_CSV = Path("tweets_with_topics_v2.csv")

GPT_MODEL = "gpt-4.1-mini"
BATCH_SIZE = 25

# export OPENAI_API_KEY=...
client = OpenAI(api_key="")

PROMPT_VERSION = "v2_blue_yellow"


BLUE_CATEGORIES = [
    {
        "id": 1,
        "name": "Market / Economy / Jobs",
        "description": "Macro economy, stock market, jobs, unemployment, growth, inflation in general.",
    },
    {
        "id": 2,
        "name": "Trade Policy / Tariffs / Manufacturing",
        "description": "Trade policy, tariffs, trade wars, imports/exports, manufacturing policy.",
    },
    {
        "id": 3,
        "name": "Regulation / Antitrust / Legal Actions",
        "description": "Regulation, antitrust cases, lawsuits, investigations, bans, legal threats.",
    },
    {
        "id": 4,
        "name": "Foreign Policy / Geopolitics / Diplomacy",
        "description": "Foreign relations, diplomacy, allies/adversaries, treaties, high-level geopolitics.",
    },
    {
        "id": 5,
        "name": "Defense / Military / National Security / Veterans",
        "description": "Military, defense spending, troops, wars, national security, veterans.",
    },
    {
        "id": 6,
        "name": "Energy / Oil & Gas / Climate / Environment",
        "description": "Oil, gas, pipelines, energy prices, climate change, environmental policy.",
    },
    {
        "id": 7,
        "name": "Healthcare / COVID-19 / Public Health",
        "description": "Healthcare policy, insurance, COVID-19, vaccines, public health issues.",
    },
    {
        "id": 8,
        "name": "Immigration / Border Security",
        "description": "Immigration, border security, walls, migrants, visas.",
    },
    {
        "id": 9,
        "name": "Official Government Announcements / Executive Actions",
        "description": "Official policy announcements, executive orders, proclamations, formal WH actions.",
    },
    {
        "id": 10,
        "name": "Campaign / Elections / Rallies / Political Messaging",
        "description": "Campaign events, election talk, rallies, political branding and messaging.",
    },
    {
        "id": 11,
        "name": "Attacks / Criticism / Conflicts (Media, Opponents, Companies)",
        "description": "Attacks or criticism aimed at media, political opponents, companies or individuals.",
    },
    {
        "id": 12,
        "name": "Personal / Social / Non-Policy Content (Congrats, Holidays, Misc)",
        "description": "Congrats, holidays, personal thanks, sports, general social commentary.",
    },
    {
        "id": 13,
        "name": "Otherwise",
        "description": "Anything that does not clearly fit the above blue categories.",
    },
]


def _build_blue_category_text() -> str:
    lines = []
    for cat in BLUE_CATEGORIES:
        lines.append(f"- {cat['name']}: {cat['description']}")
    return "\n".join(lines)


BLUE_CATEGORY_TEXT = _build_blue_category_text()



YELLOW_CATEGORY_DESCRIPTION = """
For each tweet, you must also infer a set of secondary tags, called the yellow_category.
Represent yellow_category as a JSON object with ALL of the following fields:

- "during_trading_hours": "True" | "False" | "Unknown"
    * True = clearly refers to or occurs during regular US stock market trading hours, or intraday moves.
    * False = clearly about after-hours, weekends, or not time-specific.
    * Unknown = timing cannot be inferred at all from the text.

- "towards_ceo_or_company": "True" | "False"
    * True = the tweet is clearly directed at, or attacking/praising, a specific company, CEO, or corporate figure.
    * False = otherwise.

- "sentiment": "Positive" | "Negative" | "Neutral" | "Mixed" | "Unclear"
    * Overall tone of the tweet.

- "has_market_action_keywords": "True" | "False"
    * True = the tweet explicitly suggests market movement or trading actions, using words like
      'soar', 'crash', 'plunge', 'rally', 'bullish', 'bearish', 'buy', 'sell', 'dump', 'moon', etc.
    * False = otherwise.

- "intensity": "Low" | "Medium" | "High"
    * Low = calm, descriptive language.
    * Medium = somewhat emotional or emphatic.
    * High = very emotional/intense language, often with many capital letters or exclamation marks.
"""


def build_system_message() -> str:

    return f""" You are a classification engine. Input: - For each tweet you are given: id, text, and an existing coarse topic "category" (from a previous model run). Your tasks for EACH tweet: 1) Choose EXACTLY ONE blue_category, using one of these NAMES (case-sensitive):
{BLUE_CATEGORY_TEXT} 2) Produce a yellow_category JSON object with ALL fields as specified: {YELLOW_CATEGORY_DESCRIPTION} Output STRICTLY valid JSON, as a list of objects:
[
  {{
    "id": "123",
    "blue_category": "Market / Economy / Jobs",
    "yellow_category": {{
      "during_trading_hours": "Unknown",
      "towards_ceo_or_company": "False",
      "sentiment": "Neutral",
      "has_market_action_keywords": "False",
      "intensity": "Low"
    }}
  }},
  ...
]

Rules:
- Use blue_category NAMES exactly as written above (case-sensitive).
- Always include ALL fields inside yellow_category.
- Do NOT output anything else besides the JSON list.
- The order of objects in the list must match the order of tweets given.

IMPORTANT:
You are also given a "Timestamp" for each tweet. Use this information when deciding
"during_trading_hours".

Rules for interpreting Timestamp:
- Regular US trading hours: 09:30–16:00 Eastern Time.
- Pre-market: before 09:30.
- After-hours: after 16:00.
- If the time clearly falls in regular trading hours (09:30–16:00 ET): output "True".
- If the time clearly falls outside these hours: output "False".
- If the exact timezone or timing cannot be inferred: output "Unknown".

You MUST use the timestamp to make the best possible judgment.

System version: {PROMPT_VERSION}
"""


def build_user_message(tweets_batch: list[dict]) -> str:
    lines = ["Here are the tweets to classify:"]

    for i, tw in enumerate(tweets_batch, start=1):
        text = (tw.get("text") or "").replace("\n", " ")
        category = tw.get("category") or ""
        date = str(tw.get("date") or "")

        lines.append(f"\nTweet {i}:")
        lines.append(f"ID: {tw['id']}")
        lines.append(f"Timestamp: {date}")
        lines.append(f"ExistingCategory: {category}")
        lines.append(f"Text: {text}")

    return "\n".join(lines)


def classify_batch_with_gpt(tweets_batch: list[dict]) -> dict:
    """
    return:
        { tweet_id: {"blue_category": str, "yellow_category": dict} }
    """
    system_message = build_system_message()
    user_message = build_user_message(tweets_batch)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)

            id_to_labels: dict[str, dict] = {}
            for item in data:
                tid = str(item["id"])
                blue = str(item["blue_category"])
                yellow = item.get("yellow_category", {})

                id_to_labels[tid] = {
                    "blue_category": blue,
                    "yellow_category": yellow,
                }

            return id_to_labels

        except Exception as e:
            print(f"[WARN] GPT call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2)

    return {}



def main():
    # read version 1
    print(f"Loading tweets from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required_cols = ["id", "text", "category"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["id"] = df["id"].astype(str)

    # create column
    if "blue_category" not in df.columns:
        df["blue_category"] = None
    if "yellow_category" not in df.columns:
        df["yellow_category"] = None

    # 3) build GPT prompt
    tweets = [
    {
        "id": row["id"],
        "text": row["text"],
        "date": row["date"],
        "category": row["category"],
    }
    for _, row in df.iterrows()
]

    total = len(tweets)
    print(f"Total tweets to classify (blue/yellow): {total}")

    # 4) loop
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = tweets[start:end]
        print(f"Classifying batch {start}–{end-1} (size={len(batch)})...")

        id_to_labels = classify_batch_with_gpt(batch)

        # back to  DataFrame
        for tw in batch:
            tid = str(tw["id"])
            if tid not in id_to_labels:
                continue

            labels = id_to_labels[tid]
            df.loc[df["id"] == tid, "blue_category"] = labels["blue_category"]

            yellow_val = labels.get("yellow_category", {})
            if isinstance(yellow_val, (dict, list)):
                df.loc[df["id"] == tid, "yellow_category"] = json.dumps(yellow_val)
            else:
                df.loc[df["id"] == tid, "yellow_category"] = str(yellow_val)

        # temp save in case progress lost
        if (start // BATCH_SIZE) % 20 == 0:
            tmp_path = OUTPUT_CSV.with_suffix(".tmp.csv")
            print(f"Saving intermediate snapshot to {tmp_path} ...")
            df.to_csv(tmp_path, index=False)

    # 5) save the final result
    print(f"Saving final labeled tweets to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()