import pandas as pd
import emoji

# Load dataset
df = pd.read_csv("/Users/dhruvnasit/Desktop/DS410 final project datasets/tweets(best one).csv")

# STEP 1: Remove ALL emojis + links

# Remove all emojis using emoji library
def remove_emojis(text):
    return emoji.replace_emoji(text, "")

df["text"] = df["text"].astype(str).apply(remove_emojis)

# Remove URLs (http, https, www)
df["text"] = df["text"].str.replace(r"http\S+|www\S+|https\S+", "", regex=True)

# Clean extra whitespace
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

# STEP 2: Drop rows where text becomes empty
df = df[df["text"].str.len() > 0]

# STEP 3: Remove retweets ("RT") ---
df = df[~df["text"].str.match(r'^\s*"?RT')]

# STEP 4: Remove device and is_flagged columns
df = df.drop(columns=["device", "isFlagged"], errors="ignore")

# Save cleaned dataset
df.to_csv("/Users/dhruvnasit/Desktop/DS410 final project datasets/tweets_cleaned.csv", index=False)