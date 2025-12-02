# ==================================================
# Example Test Runs with Extreme Cases
# ==================================================

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
import pyspark.sql.functions as F
import pyspark.sql.types as T

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vector

# ------------------------------------------
# Start Spark Session
# ------------------------------------------
spark = (
    SparkSession.builder
    .appName("GBTPredictionTest_ExtremeCases")
    .getOrCreate()
)

print("Spark session started!")

# ==================================================
# Example Test Runs with Extreme Cases (String flags)
# ==================================================

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
import pyspark.sql.functions as F
import pyspark.sql.types as T

# ------------------------------------------
# Start Spark Session
# ------------------------------------------
spark = (
    SparkSession.builder
    .appName("GBTPredictionTest_ExtremeCases")
    .getOrCreate()
)

print("Spark session started!")

# ------------------------------------------
# Load trained GBT PipelineModel
# ------------------------------------------
model_path = "/storage/work/yfl5682/Project/models/gbt_model"
gbt_loaded = PipelineModel.load(model_path)

print("Model loaded!")

# ------------------------------------------
# Example Test Data (all string flags)
# ------------------------------------------
data = [
    # Case 1: Baseline (your original)
    (
        "case_01_baseline",                       # id
        "China Trade Deal is going very well!",   # text
        "Trade Policy & Industrial / Manufacturing",  # category
        "Market / Economy / Jobs",                # blue_category
        "Positive",                               # sentiment
        "Medium",                                 # intensity
        "True",                                   # during_trading_hours
        "True",                                   # has_market_action_keywords
        "False",                                  # towards_ceo_or_company
        12000,                                    # favorites
        2300,                                     # retweets
        10,                                       # tweet_hour
        3,                                        # doy
        35,                                       # text_len
        0,                                        # num_exclam
        0.00042,                                  # rv_pre_30m
        3280.55                                   # Close
    ),

    # Case 2: Macro crash warning – viral, very negative, high RV
    (
        "case_02_macro_crash_warning",
        "Markets are CRASHING!!! Emergency rate cut coming, this is going to be ugly!!!",
        "Macroeconomics & Monetary Policies",
        "Market / Economy / Jobs",
        "Negative",
        "High",
        "True",          # during_trading_hours
        "True",          # has_market_action_keywords
        "False",
        350000,          # favorites
        90000,           # retweets
        9,               # tweet_hour
        1,               # doy
        110,             # text_len
        6,               # num_exclam
        0.02,            # rv_pre_30m (very high)
        2700.00
    ),

    # Case 3: Trade war escalation at night – high RV, big engagement
    (
        "case_03_trade_war_midnight",
        "Starting TOMORROW, massive NEW TARIFFS on China. Companies have been warned!",
        "Trade Policy & Industrial / Manufacturing",
        "Market / Economy / Jobs",
        "Negative",
        "High",
        "False",         # during_trading_hours (night)
        "True",
        "False",
        200000,
        80000,
        23,              # tweet_hour
        5,               # doy
        95,
        3,
        0.015,           # rv_pre_30m
        2900.00
    ),

    # Case 4: Energy shock – OPEC+ production cuts
    (
        "case_04_energy_shock",
        "Historic deal! OPEC+ just agreed to MASSIVE production cuts. Oil prices will skyrocket!",
        "Energy, Oil & Gas, Renewables",
        "Market / Economy / Jobs",
        "Positive",
        "High",
        "True",
        "True",
        "False",
        180000,
        60000,
        14,              # tweet_hour
        4,               # doy
        100,
        2,
        0.018,
        3200.00
    ),

    # Case 5: Super boring, low engagement, tiny RV
    (
        "case_05_boring_low_engagement",
        "Had a nice meeting today. Beautiful day in Washington.",
        "Macroeconomics & Monetary Policies",
        "Market / Economy / Jobs",
        "Neutral",
        "Low",
        "True",
        "False",
        "False",
        0,
        0,
        11,
        2,
        60,
        0,
        0.00005,
        3100.00
    ),

    # Case 6: Company-specific attack, high engagement, medium RV
    (
        "case_06_company_specific_attack",
        "This CEO has FAILED. New regulations & investigations coming VERY SOON!",
        "Regulation & Legal / Antitrust / Policy",
        "Company / Earnings / Guidance",
        "Negative",
        "High",
        "True",
        "True",
        "True",          # towards_ceo_or_company
        90000,
        30000,
        15,
        3,
        85,
        3,
        0.008,
        3300.00
    ),
]

# ------------------------------------------
# Schema (string for sentiment/intensity/flags)
# ------------------------------------------
schema = T.StructType([
    T.StructField("id", T.StringType()),
    T.StructField("text", T.StringType()),
    T.StructField("category", T.StringType()),
    T.StructField("blue_category", T.StringType()),
    T.StructField("sentiment", T.StringType()),
    T.StructField("intensity", T.StringType()),
    T.StructField("during_trading_hours", T.StringType()),
    T.StructField("has_market_action_keywords", T.StringType()),
    T.StructField("towards_ceo_or_company", T.StringType()),
    T.StructField("favorites", T.IntegerType()),
    T.StructField("retweets", T.IntegerType()),
    T.StructField("tweet_hour", T.IntegerType()),
    T.StructField("doy", T.IntegerType()),
    T.StructField("text_len", T.IntegerType()),
    T.StructField("num_exclam", T.IntegerType()),
    T.StructField("rv_pre_30m", T.DoubleType()),
    T.StructField("Close", T.DoubleType()),
])

test_df = spark.createDataFrame(data, schema)

print("Test DataFrame created!")
test_df.select(
    "id", "sentiment", "intensity",
    "during_trading_hours", "has_market_action_keywords",
    "towards_ceo_or_company", "favorites", "retweets",
    "rv_pre_30m", "tweet_hour", "doy"
).show(truncate=False)

# ==========================================
# Run Prediction
# ==========================================
# UDF to extract P(label=1) from probability vector
def get_prob_event(v: Vector):
    if v is None:
        return None
    # index 1 = probability for class 1 (event)
    return float(v[1])

get_prob_event_udf = udf(get_prob_event, DoubleType())

pred = gbt_loaded.transform(test_df)

# Add prob_event column
pred = pred.withColumn("prob_event", get_prob_event_udf("probability"))

print("\n=== Prediction Results (sorted by prob_event DESC) ===")
pred.select(
    "id",
    "prediction",
    "prob_event",
    "probability"
).orderBy(F.col("prob_event").desc()) \
 .show(truncate=False)