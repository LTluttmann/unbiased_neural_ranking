from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import *

from mt.data.preprocessing.features import get_character_n_gram, get_feature_vocab
from mt.data.preprocessing.big_query import get_big_query_credentials, get_data_from_big_query
from mt.utils import (
    S3Url, ensure_dirs_exists, ensure_proper_s3_folder_path, 
    get_latest_subfolder_from_url, s3_folder_exists_and_not_empty
)
from mt.config import MainConfig, config

import numpy as np
import os
import boto3
from datetime import datetime
from typing import Union


def explode_into_feature_columns(x, num_features):
    return_value = [
        [
            np.atleast_1d(row[feature_idx]).tolist() for row in x
        ] for feature_idx in range(num_features)
    ]
    return return_value


def map_dtype_to_sql_datatype(dtype):
    if "bigint" in dtype:
        return LongType()
    elif "string" in dtype:
        return StringType()
    elif "double" in dtype:
        return FloatType()
    elif "float" in dtype:
        return FloatType()
    elif "date" in dtype:
        return DateType()
    else:
        raise NotImplementedError(f"type {dtype} not known yet")


def prepare_sequence_date(df: DataFrame, id_cols:list):
    """This function brings the clickstream training dataframe in the right format to be saved as 
    tf.SequenceExample. The respective function of the sparktfrecord package requires that features
    (and labels) of the examples (products per query) are stored in a single line per query event,
    i.e. that each query event (sessionid+searchterm) has only one line, where the values of all
    the products belonging to it are represented as arrays:
    
    |   id  |   searchterm  |             feat1                |                feat2             |
    -----------------------------------------------------------------------------------------------
    |   1   |       abc     | [feat1_product1, feat1_product2] | [feat2_product1, feat2_product2] |
    |   1   |       def     | [feat1_product1, feat1_product2] | [feat2_product1, feat2_product2] |
    |   2   |       abc     | [feat1_product1, feat1_product2] | [feat2_product1, feat2_product2] |

    Args:
        df (DataFrame): spark dataframe
        id_cols (list): columns to identify a queryevent
    """
    # first, for every query session, the features of all observations are concatenated in a single list
    feature_and_label_cols = [col for col in df.columns if col not in id_cols]

    df_sequence = (
        df.select(
            *id_cols,
            F.struct([*feature_and_label_cols]).alias("newcol"))
        .groupBy(id_cols)
        .agg(F.collect_set("newcol").alias("collected_col"))
    )
    # define the schema of the resulting dataframe. Each feature (and label) has a column 
    # which is an array of values for all observations that belong to the same query event
    mapping = {
        col: map_dtype_to_sql_datatype(dtype)
        for col, dtype in df.select(feature_and_label_cols).dtypes}

    schema = StructType(
        [StructField(col, ArrayType(ArrayType(dtype))) for col, dtype in mapping.items()])
    # deifne the udf using the schema
    udf_explode = F.udf(lambda x: explode_into_feature_columns(
        x, num_features=len(feature_and_label_cols)), schema)
    # apply the udf
    df_sequence = (
        df_sequence.withColumn("ncol", udf_explode("collected_col"))
        .select(id_cols + [F.col("ncol").getItem(col).alias(col) for col in feature_and_label_cols])
    )  
    return df_sequence


def get_or_create_subfolder(destination: str, prefix:str, timestamp_format: str) -> str:
    if not prefix.endswith("/"):
        prefix += "/"
    client = boto3.client("s3")
    s3url = S3Url(destination)
    current_folder = get_latest_subfolder_from_url(client, s3url, timestamp_format)
    new_destination = os.path.join(destination, current_folder, prefix)
    if s3_folder_exists_and_not_empty(client, new_destination):
        new_timestamp = datetime.now().strftime(timestamp_format)
        new_destination = os.path.join(destination, new_timestamp, prefix)
    return new_destination


def write_df_to_tfrecord(
        df: DataFrame,
        destination: str,
        save_sequence: bool = True,
        max_records_per_file: int = 0,
        partition_cols: list = None,
        id_cols:list = config.ID_COLS):
    """write tfrecords from spark dataframe
    """
    if save_sequence:
        record_type = "SequenceExample"
        df = prepare_sequence_date(df, id_cols=id_cols)
    else:
        record_type = "Example"
    # write
    df_write = df.write \
        .option("recordType", record_type) \
        .option("maxRecordsPerFile", max_records_per_file)
    # partition
    if partition_cols is not None:
        partition_cols = [partition_cols] if not isinstance(partition_cols, list) else partition_cols
        df_write = df_write.partitionBy(*partition_cols)
    # save
    df_write \
        .format("tfrecord") \
        .mode("overwrite") \
        .save(destination) 
    

def train_test_split(spark, df, test_frac_or_size: Union[float, int], judge_path=None):
    frac = test_frac_or_size if isinstance(test_frac_or_size, float) else None
    size = test_frac_or_size if isinstance(test_frac_or_size, int) else None

    total_rows = df.count()

    if judge_path is not None:

        df_judgements = spark.read.parquet(judge_path)
        df_judgements = df_judgements.withColumnRenamed("query", "searchterm_normalized")
        df_w_judge = df.join(
            df_judgements.select("searchterm_normalized", "offer_or_product_id", "hier_click_combined_w_order_judgement"), 
            on=["searchterm_normalized", "offer_or_product_id"], how="left")
        
        # ALTERNATIVE METHOD
        df_w_judge_not_null = df_w_judge.filter(F.col("hierarchical_click_judgement").isNotNull())
        # filter out sessions that dont have a lot of documents with judgement scores
        sess_w_suff_judge = df_w_judge_not_null.groupBy(["sessionid", "searchterm"]).agg(F.count("*").alias("cnt")).filter(F.col("cnt") >= 5)
        df_candidates = df_w_judge_not_null.join(sess_w_suff_judge.select("sessionid", "searchterm"), on=["sessionid", "searchterm"], how="inner")

        # df_w_judge_not_null = df_w_judge \
        #     .groupBy(config.ID_COLS) \
        #     .agg(F.sum(F.col("hierarchical_click_judgement").isNull().cast("int")).alias("num_missing_judgements")) \
        #     .filter(F.col("num_missing_judgements") == 0) \
        #     .drop("num_missing_judgements")
        # df_candidates = df_w_judge.join(df_w_judge_not_null, on=config.ID_COLS, how="inner")

    else:

        df_candidates = df

    # NOTE use the query level as aggregation and remove all queries in the validation set from training set
    tmp_df = df_candidates.select(config.QUERY_COL).distinct().withColumn("id", F.row_number().over(Window.orderBy(F.rand())))

    if frac:
        df_valid = df_candidates.join(tmp_df, on=config.QUERY_COL).filter(F.col("id") < int(frac*total_rows)).drop("id")
    else:
        df_valid = df_candidates.join(tmp_df, on=config.QUERY_COL).filter(F.col("id") < size).drop("id")

    df_train = df.join(df_valid, on=config.QUERY_COL, how="left_anti")

    return df_train, df_valid


def main(
        spark: SparkSession,
        config: MainConfig,
        output_bucket: str,
        prefix: str,
        query: str = None,
        df: DataFrame = None,
        n: int = 0,
        store_vocab: bool = False,
        save_sequence: bool = True,
        return_df: bool = False,
        filter_sequence: bool = True,
        test_frac_or_size: Union[float, int] = None,
        test_days: int = None,
        judgement_path:str = None,
        id_cols:list=config.ID_COLS):
    """Apply a set of preprocessing steps, then bring the data into the right format 
    to be able to transform it into SequenceExample protobufs and write the resulting 
    dataframe as .tfrecord files.
    Args:
        input_path (str, optional): _description_. Defaults to None.
        query (str, optional): _description_. Defaults to None.

    Raises:
        ValueError: either input_path or query must be set
    """
    
    # load normalization lists from s3
    if query is None and df is None:
        raise ValueError("pass either query or df")

    if "s3://" in output_bucket:
        tf_data_destination = get_or_create_subfolder(output_bucket, prefix, config.TIMESTAMP_SIGNATURE_FORMAT)
    else:
        tf_data_destination = os.path.join(output_bucket, prefix)
        ensure_dirs_exists(tf_data_destination)

    if df is None:
        df: DataFrame = get_data_from_big_query(spark, get_big_query_credentials(), query)

    # filter rows with na in any features. Is not supported by sequence examples and 
    # also poses problems in NN training. So drop them here
    # use following query to check null values in bigquery data first:
    print(df.count())
    df = df.na.drop()
    print(df.count())

    if save_sequence and filter_sequence:
        # filter sequences with only clicks or no clicks at all
        df_filter = df.groupBy(config.ID_COLS).agg(F.countDistinct(F.col(config.CLICK_COL)).alias("cnt")).filter(F.col("cnt") > 1).drop("cnt")
        df = df.join(df_filter.select(config.ID_COLS), on=config.ID_COLS, how="inner")

    for text_col in config.NORM_TEXT_COLS:
        if not text_col in df.columns:
            continue
        # normalize text columns
        if config.NORMALIZER:
            df = config.NORMALIZER(df, text_col)
            # filter columns with empty text colum. Should only happen after normalization
            df = df.filter(F.col(text_col) != "")
        # create letter n-grams
        if n > 0:
            df, text_col = get_character_n_gram(df, n, text_col, "#")
        # in case the name of the query col has changed due to preprocessing, update it
        if text_col == config.QUERY_COL:
            config.QUERY_COL = text_col
        # store vocabulary to build embeddings later. We need tokens for vocabulary
        if store_vocab and config.NORMALIZER is not None:
            spark_vocab_path = os.path.join(tf_data_destination, f"{text_col}_{config.SPARK_VOCAB_OUTPUT_SUFFIX}")
            get_feature_vocab(df, text_col, spark_vocab_path)
        

    write_kwargs = {"save_sequence": save_sequence, "id_cols": id_cols}

    if test_frac_or_size:

        df_train, df_valid = train_test_split(spark, df, test_frac_or_size, judgement_path)
        # write train dataframe
        write_df_to_tfrecord(df_train, os.path.join(tf_data_destination, "train"), **write_kwargs)
        # write test dataframe
        write_df_to_tfrecord(df_valid, os.path.join(tf_data_destination, "val"), **write_kwargs)

        if return_df:
            return df_train, df_valid

    elif test_days:

        dates = df.select('query_date').distinct().rdd.map(lambda x: x.query_date).collect()
        dates.sort()

        train_dates = dates[:-test_days]
        valid_dates = dates[-test_days:]

        df_train = df.filter(F.col("query_date").isin(train_dates))
        df_valid = df.filter(F.col("query_date").isin(valid_dates))

        if judgement_path is not None:

            df_judgements = spark.read.parquet(judgement_path)
            df_judgements = df_judgements.withColumnRenamed("query", config.NORMALIZED_QUERY_COL)

            df_valid = df_valid.join(
                df_judgements.select(config.NORMALIZED_QUERY_COL,
                                     config.OFFER_OR_PRODUCT_COL, 
                                     config.JUDGEMENT_COL),
                on=[config.NORMALIZED_QUERY_COL, config.OFFER_OR_PRODUCT_COL], how="left"
            )

            df_valid = df_valid.fillna({config.JUDGEMENT_COL: -1})

        # write train dataframe
        write_df_to_tfrecord(df_train, os.path.join(tf_data_destination, "train"), **write_kwargs)
        # write test dataframe
        write_df_to_tfrecord(df_valid, os.path.join(tf_data_destination, "val"), **write_kwargs)

        if return_df:
            return df_train, df_valid
        

    else:

        if judgement_path is not None:

            df_judgements = spark.read.parquet(judgement_path)
            df_judgements = df_judgements.withColumnRenamed(
                "query", config.NORMALIZED_QUERY_COL)
            df = df.join(
                df_judgements.select(config.NORMALIZED_QUERY_COL,
                                     config.OFFER_OR_PRODUCT_COL, 
                                     config.JUDGEMENT_COL),
                on=[config.NORMALIZED_QUERY_COL, config.OFFER_OR_PRODUCT_COL], how="left")

            df = df.fillna({config.JUDGEMENT_COL: -1})

        write_df_to_tfrecord(df, tf_data_destination, **write_kwargs)

        if return_df:
            return df


if __name__ == "__main__":
    spark = (
        SparkSession
        .builder
        .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.21.1,com.linkedin.sparktfrecord:spark-tfrecord_2.12:0.3.4")
        .appName("mtdata")
        .getOrCreate()
    )

    spark.conf.set("viewsEnabled", "true")
    spark.conf.set("temporaryGcsBucket", "temp-spark-data")

    query = "SELECT * FROM XXXXXXXX"
    main(spark, config, output_bucket="tmp", query=query, store_vocab=True)
