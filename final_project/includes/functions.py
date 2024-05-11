# Databricks notebook source
# Databricks notebook source
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.streaming import DataStreamWriter
from pyspark.sql.functions import (
    regexp_extract,
    to_timestamp,
    expr,
    regexp_replace,
    pandas_udf,
)
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC Defining **Common** functions

# COMMAND ----------

def create_stream_writer(
    df: DataFrame,
    checkpoint: str,
    queryName: str,
    mode: str = "append",
) -> DataStreamWriter:
    """
    Creates stream writer object with checkpointing at `checkpoint`
    """
    return (
        df.writeStream.format("delta")
        .outputMode(mode)
        .option("checkpointLocation", checkpoint)
        .queryName(queryName)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Defining **Raw** functions

# COMMAND ----------

def read_stream_raw(spark: SparkSession) -> DataFrame:
    """
    Reads stream from `TWEET_SOURCE_PATH` with schema enforcement.
    """
    raw_data_schema = "date STRING, user STRING, text STRING, sentiment STRING"

    return (
        spark.readStream.format("json")
        .schema(raw_data_schema)
        .option("mergeSchema", "true")
        .load(TWEET_SOURCE_PATH)
    )

# COMMAND ----------

def transform_raw(df: DataFrame) -> DataFrame:
    """
    Transforms `df` to include `source_file` and `processing_time` columns.
    """
    return df.select(
        "date",
        "text",
        "user",
        "sentiment",
        input_file_name().alias("source_file"),
        current_timestamp().alias("processing_time"),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Defining **Bronze** functions

# COMMAND ----------

def read_stream_bronze(spark: SparkSession) -> DataFrame:
    """
    Reads stream from `BRONZE_DELTA`.
    """
    return spark.readStream.format("delta").load(BRONZE_DELTA)

# COMMAND ----------

def transform_bronze(df: DataFrame) -> DataFrame:
    """
    Transforms `df` to include `timestamp`, `mention` and `cleaned_text` columns.
    """
    return (
        df.withColumn("timestamp", to_timestamp("processing_time"))
        .withColumn("mention", regexp_extract(col("text"), "@\\w+", 0))
        .withColumn("cleaned_text", regexp_replace(col("text"), "@\\w+", ""))
        .select("timestamp", "mention", "cleaned_text", "sentiment")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Defining **Silver** functions

# COMMAND ----------

def read_stream_silver(spark: SparkSession) -> DataFrame:
    return spark.readStream.format("delta").load(SILVER_DELTA)

# COMMAND ----------

@pandas_udf("score: int, label: string")
def perform_model_inference(s: pd.Series) -> pd.DataFrame:
    predictions = loaded_model.predict(s.tolist())
    return pd.DataFrame({
        "score": predictions["score"].map(lambda x: int(x * 100)).tolist(),
        "label": predictions["label"].tolist(),
    })

# COMMAND ----------

def transform_silver(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn("sentiment_analysis", perform_model_inference(col("cleaned_text")))
        .withColumn("predicted_score", col("sentiment_analysis.score"))
        .withColumn("predicted_sentiment", col("sentiment_analysis.label"))
        .withColumn("sentiment_id", when(col("sentiment") == "positive", 1).otherwise(0))
        .withColumn(
            "predicted_sentiment_id",
            when(col("predicted_sentiment") == "POS", 1).otherwise(0),
        )
        .select(
            "timestamp",
            "mention",
            "cleaned_text",
            "sentiment",
            "predicted_score",
            "predicted_sentiment",
            "sentiment_id",
            "predicted_sentiment_id",
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Defining **Gold** Functions

# COMMAND ----------

def read_stream_gold(spark: SparkSession) -> DataFrame:
    return spark.readStream.format("delta").load(GOLD_DELTA)

# COMMAND ----------

def query_non_empty_mentions(goldDF):
    return goldDF.filter(goldDF.mention.isNotNull()).filter(goldDF.mention != "")

# COMMAND ----------

def query_mention_sentiment_count(goldDF):
    return (
        query_non_empty_mentions(goldDF)
        .groupby("mention")
        .agg(
            count(when(goldDF.sentiment == "neutral", 1)).alias("neutral_count"),
            count(when(goldDF.sentiment == "positive", 1)).alias("positive_count"),
            count(when(goldDF.sentiment == "negative", 1)).alias("negative_count"),
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Defining ML Flow Logging Functions

# COMMAND ----------

def get_tp_tn_fp_fn(df):
    true_positives = df.filter(
        (df.sentiment_id == 1) & (df.predicted_sentiment_id == 1)
    ).count()
    true_negatives = df.filter(
        (df.sentiment_id == 0) & (df.predicted_sentiment_id == 0)
    ).count()
    false_positives = df.filter(
        (df.sentiment_id == 0) & (df.predicted_sentiment_id == 1)
    ).count()
    false_negatives = df.filter(
        (df.sentiment_id == 1) & (df.predicted_sentiment_id == 0)
    ).count()

    return true_positives, true_negatives, false_positives, false_negatives

# COMMAND ----------

def log_confusion_matrix(true_positives, true_negatives, false_positives, false_negatives):
    # Create the confusion matrix
    confusion_matrix = spark.createDataFrame(
        [(true_positives, false_positives), (false_negatives, true_negatives)],
        ["Actual Positive", "Actual Negative"],
    )

    # Plot the confusion matrix
    plt.figure(figsize=(4, 4))
    sns.heatmap(confusion_matrix.toPandas(), annot=True, cmap="YlGnBu")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    # Log confusion matrix image as MLflow artifact
    mlflow.log_artifact("confusion_matrix.png")

# COMMAND ----------

def log_precision_recall_f1(true_positives, true_negatives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    # Store the precision, recall, and F1-score as MLflow metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1_score)

# COMMAND ----------

def log_model_details():
    client = MlflowClient()
    details = client.get_model_version(name=MODEL_NAME, version=1)
    mlflow.log_param("model_name", details.name)
    mlflow.log_param("mlflow_version", details.version)

# COMMAND ----------

def log_silver_version():
    silver_table_version = (
        spark.sql("DESCRIBE HISTORY silver.delta.`{}`".format(SILVER_DELTA))
        .select("version")
        .collect()[0][0]
    )
    mlflow.log_param("silverDF_version", silver_table_version)
