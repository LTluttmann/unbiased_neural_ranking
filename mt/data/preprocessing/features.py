from pyspark.ml.feature import NGram, HashingTF, IDF, Tokenizer
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.linalg import SparseVector, VectorUDT
# from pyspark.mllib.linalg import VectorUDT

from mt.data.preprocessing.big_query import get_big_query_credentials, get_data_from_big_query
from mt.data.preprocessing.normalization_lists import NormalizationLists
from mt.data.preprocessing.normalize import clean_text_full, full_normalization
from mt.utils import delete_dir
from mt.config import config 
from mt.data.queries.get_lpf_id_and_titles import query as lpf_query

import numpy as np
from typing import Tuple, List


def pad_tokens(df:DataFrame, column, pad_character):
    pad_df = (
        df.select(column).distinct()
        .rdd.map(lambda row: (
            row[column], 
            [f"{pad_character}{token}{pad_character}" for token in row[column]])
        )
        .toDF([column, f"{column}_padded"])
    )     
    return pad_df


def get_character_n_gram(df:DataFrame, n:int, column:str, pad_character:str="") -> Tuple[DataFrame, str]:
    # rdd operation to split words in characters and add a hash at the 
    # start and end of each token
    char_df = (
        df.select(column).distinct()
        .rdd.map(lambda row: (
            row[column], 
            [[char for char in f"{pad_character}{token}{pad_character}"] for token in row[column]])
        )
        .toDF([column, "chars"])
    )   
    # explode chars of multiple tokens of column value into multiple rows
    # This is necessary, since the NGram function cannot work with nested arrays
    exp_char_df = char_df.withColumn("chars", F.explode("chars"))
    # get ngrams
    n_gram_col_name = f"{column}_{n}gram"
    n_gram = NGram(n=n, inputCol="chars", outputCol=n_gram_col_name)
    n_gram_df = n_gram.transform(exp_char_df)
    # remove white spaces between characters of n-gram
    n_gram_df = n_gram_df.withColumn(
        n_gram_col_name, F.expr(f"""transform({n_gram_col_name},x-> regexp_replace(x,"\ ",""))""")
    )
    # collect ngrams of all tokens of column value
    n_gram_df = (
        n_gram_df
        .groupBy(column)
        .agg(F.collect_list(n_gram_col_name)
        .alias(n_gram_col_name))
    )
    # and finally flatten the nested ngram arrays per column value
    n_gram_df = n_gram_df.withColumn(n_gram_col_name, F.flatten(n_gram_col_name))
    # join n gram column on original data frame
    df = df.join(n_gram_df.select(column, n_gram_col_name), on=[column], how="inner")
    return df, n_gram_col_name



def get_token_vocab(df: DataFrame, column:str, pad_character:str=""):
    vocab = (
        df
        .select(column).distinct()
        .rdd.flatMap(
            lambda row: [f"{pad_character}{w}{pad_character}" for w in row[column]]
            )
        .collect()
    )
    return np.unique(vocab).tolist()


def get_normalized_text_cols(spark: SparkSession, nls: NormalizationLists, colnames:List[str], limit: int = None):
    # cc = {"a": "v", "b": "w"}
    # ",".join([f"{key} AS {value}" for key, value in cc.items()])
    
    get_vocab_query = """
        SELECT DISTINCT {cols}
        FROM `XXX`
        WHERE lpf_date=CAST(FORMAT_DATE("%Y%m%d", CURRENT_DATE-7) AS integer)
        AND lpf_time = (SELECT MAX(lpf_time)
            FROM `XXX`
            WHERE lpf_date=CAST(FORMAT_DATE("%Y%m%d", CURRENT_DATE-7) AS integer)
        )
    """.format(cols=",".join(colnames))
    if limit:
        get_vocab_query = get_vocab_query + f"LIMIT {limit}"
    df = get_data_from_big_query(spark, get_big_query_credentials(), get_vocab_query)
    norm_udf = F.udf(
        lambda c: clean_text_full(c, nls),
        ArrayType(StringType())
    )
    # normalize text columns
    for col in colnames:
        df = df.withColumn(col, norm_udf(F.col(col)))
    return df


def get_feature_vocab(df_token: DataFrame, column: str, output_path: str):
    """Function to generate a text file vocabulary from a column of tokens

    Args:
        df_token (DataFrame): Spark dataframe with normalized and tokenized text / categorical columns
        column (str): column to create vocabulary of
        output_path (str): path the vocab is save to. filename format is as follows: <column>_vocab.txt
    """
    df_token = df_token.select(F.explode(column).alias("token")).distinct()
    # spark rdds safe function requires that the folder is not there, so delete if it exists
    if not "s3://" in output_path:
        delete_dir(output_path)
    # fetch tokens in rdd and write as text file
    df_token.rdd.map(lambda x: x.token).saveAsTextFile(output_path)


def calc_tfidf_and_bm25(spark, dates: List[int], df_tfidf_path=None, hashing_tf_path=None):
    """dates must be provisioned in format %Y%m%d in integer format"""

    # retrieve all product titles and the corresponding offer or product id from lhotse (BQ)
    query = lpf_query.format(dates=",".join([str(x) for x in dates]))
    df_text = get_data_from_big_query(spark, get_big_query_credentials(), query)
    # repartition for parallel execution
    df_text = df_text.repartition(32)
    df_text = df_text.select("offer_or_product_id", "content__product__name") \
        .groupby("offer_or_product_id") \
        .agg(F.first(F.col("content__product__name")).alias("content__product__name"))
    # normalize and tokenize text
    df_tokens = full_normalization(df_text, "content__product__name")
    # cache result
    #df_tokens.cache()
    #df_tokens.count()

    # hash tokens
    hashingTF = HashingTF(inputCol="content__product__name", outputCol="tf", numFeatures=1048576)
    df_tf = hashingTF.transform(df_tokens)
    idf = IDF(inputCol="tf", outputCol="idf")
    idf_model = idf.fit(df_tf)
    df_idf = idf_model.transform(df_tf)

    # cache result
    #df_idf.cache()
    #df_idf.count()

    # compute the average document length
    avgdl = df_tokens.select(F.mean(F.size(F.col("content__product__name")))).rdd.flatMap(lambda x: x).collect()[0]

    def log_scaled_tfidf(tf:SparseVector,idf:SparseVector):
        """function to perform elementwise multiplication of two sparse vectors.
        By doing so, this funciton can be used to calculate tfxidf from tf and idf vectors
        """
        return SparseVector(tf.size, tf.indices, np.log1p(tf.values) * idf.values)

    def bm25(tf:SparseVector, idf:SparseVector, k=1.2, b=0.75):
        """calculate the bm25 score as per this definiton: https://en.wikipedia.org/wiki/Okapi_BM25"""
        return SparseVector(
            tf.size, tf.indices, 
            idf.values * ((tf.values * (k+1)) / tf.values + k * (1-b + b * (tf.numNonzeros()/avgdl)))
        )

    elementwise_multiplication_udf = F.udf(log_scaled_tfidf, VectorUDT())
    bm25_udf = F.udf(bm25, VectorUDT())

    df_tfidf = df_idf.withColumn("tfidf", elementwise_multiplication_udf(F.col("tf"), F.col("idf")))
    df_tfidf = df_tfidf.withColumn("bm25", bm25_udf(F.col("tf"), F.col("idf")))
    df_tfidf = df_tfidf.drop("tf").drop("idf").drop("content__product__name")


    if df_tfidf_path and hashing_tf_path:
        df_tfidf.write.parquet(df_tfidf_path)
        hashingTF.save(hashing_tf_path)

    return df_tfidf, hashingTF


def get_tfidf_and_bm25_features(spark:SparkSession, df, dates: List[int] = None, df_tfidf_path=None, hashing_tf_path=None, inner_join:bool = True):
    join_how = "inner" if inner_join else "left"

    if df_tfidf_path == None or hashing_tf_path == None:
        assert dates is not None, "provide lpf dates to determine tfidf"
        df_tfidf, hashingTF = calc_tfidf_and_bm25(spark, dates)  
    else:
        df_tfidf = spark.read.parquet(df_tfidf_path)
        hashingTF = HashingTF.load(hashing_tf_path)

    df = Tokenizer(inputCol="searchterm_normalized", outputCol="searchterm_tokens").transform(df)
    df = hashingTF.setInputCol("searchterm_tokens").setOutputCol("searchterm_hash").transform(df)
    df = df.drop("searchterm_tokens")

    df_w_tfidf = df.join(df_tfidf, on="offer_or_product_id", how=join_how)

    def dot_fn(array):
        if array[1] is None:
            """caution."""
            result = 0.0
        else:
            result = array[0].dot(array[1])
        return float(result)

    dot_udf = F.udf(dot_fn, FloatType())

    df_w_tfidf = df_w_tfidf.withColumn("tfidf_score", dot_udf(F.array(F.col("searchterm_hash"), F.col("tfidf"))))
    df_w_tfidf = df_w_tfidf.withColumn("bm25_score", dot_udf(F.array(F.col("searchterm_hash"), F.col("bm25"))))

    df_w_tfidf = df_w_tfidf.drop("searchterm_hash").drop("tfidf").drop("bm25")

    return df_w_tfidf

