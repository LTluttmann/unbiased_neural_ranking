from mt.data.preprocessing.normalize import NormalizationFunction, simple_normalization_and_lowercasing

from dataclasses import dataclass



@dataclass
class PathConfig:
    SSM_BIG_QUERY_CREDS_PATH: str = "xxx"
    # standard s3 paths
    BLOCKLIST_PATH:str = "xxx"
    NORMALIZATION_LISTS_PATH: str = "xxx"
    MT_DR_BANNER_URL: str = "xxx"
    MT_OUTPUT_URL: str = "xxx"
    # spark
    SPARK_VOCAB_OUTPUT_SUFFIX: str = "vocab"
    # bert
    BERT_VOCAB_SUFFIX: str = "vocab"
    # pbk
    PBK_DATA_SUFFIX: str = "pbk_data"
    # encoder
    ENCODER_TRAINING_DATA_PREFIX: str= "encoder_data"
    ENCODER_FOLDER_SUFFIX: str = "model"
    ENCODER_FILENAME: str = "encoder"


@dataclass
class FeatureConfig:

    NORMALIZED_QUERY_COL: str = "searchterm_normalized"
    QUERY_COL: str = "searchterm"
    PRODUCT_TITLE_COL: str = "product_pd_Name"
    SESSION_ID_COL: str = "sessionid"

    CLICK_COL: str = "click"
    ORDER_COL:str = "ordered"
    LABELS = ["click", "ordered"]

    BERT_MERGE_VOCABS: bool = True

    CLICKSTREAM_CLASSIFICATION_GROUP_COL:str = "product_pd_ClassificationGroup"

    POSITION_COL:str = "actual_position"
    DEVICE_COL: str = "device"
    LAYOUT_COL: str = "layout_type"
    PAGE_COL: str = "page_number"
    RETRIEVED_PRODUCTS_COL: str = "num_retrieved_products"

    PRICE_COL: str = "price"
    PBK_COL:str = "product_pd_AttributePbk"
    RATING_COL: str = "product_pd_AverageRating"
    NUM_RATINGS_COL:str = "product_pd_RatingCount"
    DISCOUNT_COL: str = "product_pd_DiscountPercentage"
    NUM_CLICKS_COL:str = "num_clicks"
    DEMAND_COL: str = "num_orders"
    FRESHNESS_COL: str = "avg_freshness"
    COEC_COL:str = "coec"
    CATEGORY_COL:str = "product_pd_Category"


    SAMPLE_WEIGHT_ON_LOSS_COL:str = "example_weight"
    SAMPLING_WEIGHT_COL:str = "sample_weight"

    TARGET_MATCH_COL:str = "target_match"
    BRAND_MATCH_COL:str = "brand_match"
    COLOR_MATCH_COL:str = "color_match"
    TFIDF_COL:str = "tfidf_score"
    BM25_COL:str = "bm25_score"
    AVAILABILITY_COL:str = "delivery_time"
    QUERY_LENGTH_COL:str = "query_length"
    QUERY_FAME_COL:str = "query_occurances"
    SHIPPING_COST_COL:str = "shipping_cost"


    POS_BIAS_FEATURE_COL:str = "pb_features"
    NUMERIC_FEATURES_COL:str = "num_features"
    MATCH_HIST_COL:str = "histogram"

    OFFER_OR_PRODUCT_COL:str = "offer_or_product_id"
    JUDGEMENT_COL:str = "hier_click_combined_w_order_judgement"

    MAX_SEQ_LENGTH:int = 30
    MAX_TOKENS:int = 32

    DEVICE_VOCAB = ["mobile", "desktop", "tablet", "unknown"]
    LAYOUT_VOCAB = ["galleryPortrait", "list", "galleryLandscape", "galleryQuad"]
    CATEGORY_VOCAB = ["FASHION_SPORT", "ELECTRONIC_DIGITAL", "WAESCHE_BADEMODEN", "HOME_LIVING", "SONSTIGES", "GARDEN_DIY"]

    @property
    def ID_COLS(self):
        return [self.SESSION_ID_COL, self.QUERY_COL]

    def __post_init__(self):
        self.RAW_TEXT_COLUMNS = [self.QUERY_COL, self.PRODUCT_TITLE_COL, self.NORMALIZED_QUERY_COL]
        self.NORM_TEXT_COLS = [self.QUERY_COL, self.PRODUCT_TITLE_COL]

        self.POSITION_BIAS_FEATURES = [
            self.POSITION_COL, 
            self.DEVICE_COL, 
            self.LAYOUT_COL, 
            #self.PAGE_COL, 
            #self.RETRIEVED_PRODUCTS_COL
        ]

        self.CATEGORICAL_FEATURES = {self.CATEGORY_COL: self.CATEGORY_VOCAB}
        
        self.NUMERICAL_COLUMNS = [
            self.TFIDF_COL, 
            self.BRAND_MATCH_COL, 
            self.PRICE_COL, 
            self.DISCOUNT_COL,
            self.DEMAND_COL, 
            self.NUM_CLICKS_COL,
            self.RATING_COL,
            self.NUM_RATINGS_COL,
            self.AVAILABILITY_COL,
            self.FRESHNESS_COL,
            self.QUERY_FAME_COL,
            self.QUERY_LENGTH_COL,
            self.SHIPPING_COST_COL,
            self.COEC_COL
            # self.TARGET_MATCH_COL,
            # self.COLOR_MATCH_COL,
            # self.RETRIEVED_PRODUCTS_COL
        ]

        self.QUERY_ITEM_FEATURES = [self.TFIDF_COL, self.BM25_COL, self.BRAND_MATCH_COL, self.TARGET_MATCH_COL, self.COLOR_MATCH_COL]
        
        self.LOG1P_TRANSFORM_COLS = [
            self.PRICE_COL,
            self.TFIDF_COL,
            self.DEMAND_COL,
            self.NUM_CLICKS_COL,
            self.NUM_RATINGS_COL,
            self.AVAILABILITY_COL,
            self.FRESHNESS_COL,
            self.QUERY_FAME_COL,
            self.QUERY_LENGTH_COL,
            # self.SHIPPING_COST_COL,
            # self.RETRIEVED_PRODUCTS_COL
        ]

        self.MERGE_COLS = list(self.CATEGORICAL_FEATURES.keys()) + self.NUMERICAL_COLUMNS

        self.BERT_VOCABS = [self.PRODUCT_TITLE_COL, self.QUERY_COL]

        self.RANKER_FEATURES = list(self.CATEGORICAL_FEATURES.keys()) + self.BERT_VOCABS + self.NUMERICAL_COLUMNS + [self.PBK_COL]

        self.ALL_COLS = self.LABELS + self.RANKER_FEATURES + self.POSITION_BIAS_FEATURES + [self.SAMPLING_WEIGHT_COL, self.OFFER_OR_PRODUCT_COL]

        self.NORMALIZER: NormalizationFunction = simple_normalization_and_lowercasing

        if self.BERT_MERGE_VOCABS:
            self.BERT_VOCAB_FILENAME = f"{'_'.join(self.BERT_VOCABS)}_vocab.txt"
        else:
            self.BERT_VOCAB_FILENAMES = [f"{col}_vocab.txt" for col in self.BERT_VOCABS]

@dataclass
class ModelConfig:
    VOCAB_SIZE: int = 35_000

@dataclass
class DatasetConfig:
    start_date:str
    end_date:str


@dataclass
class MainConfig(
    PathConfig,
    FeatureConfig,
    ModelConfig
):
    TIMESTAMP_SIGNATURE_FORMAT: str = "%Y%m%d-%H:%M:%S"


config = MainConfig()

if __name__ == "__main__":
    print(config.NORMALIZER)