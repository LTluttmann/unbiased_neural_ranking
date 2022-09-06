import base64
import json

import boto3

from mt.config import config


def get_big_query_credentials():
    return "XXXXXXXX"


def get_data_from_big_query(spark_session, big_query_credentials, query):
    df = (
        spark_session
            .read
            .format("bigquery")
            .option("parentProject", "xxx")
            .option("credentials", big_query_credentials)
            .option("materializationProject", "xxx")
            .option("materializationDataset", "xxx")
            .option("query", query)
            .load()
    )

    return df
