import re
from typing import Callable, List, Dict
from lxml import html, etree
from nltk.stem.snowball import SnowballStemmer

import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import DataFrame

from mt.data.preprocessing.normalization_lists import NormalizationLists


def remove_empty_and_long_tokens(str_in: List[str]):
    return [token for token in str_in if len(token) <= 1000 and token != '']


def stem_tokens(str_in: List[str]) -> List[str]:
    stemmer = SnowballStemmer("german")
    return [stemmer.stem(token) if len(token) > 6 else token for token in str_in]


def remove_stopwords(str_in: List[str], stopwords: List[str]) -> List[str]:
    return [token for token in str_in if token not in stopwords]


def lemmatize(str_in: List[str], lemmatizing_dict: Dict[str, str]) -> List[str]:
    return [lemmatizing_dict.get(token, token) for token in str_in]


def split_on_slashes_except_fraction(str_in: List[str]) -> List[str]:
    # also removes empty tokens
    token_list = [[token] if re.match(r"\d+/\d+", token) else token.split("/") for token in str_in]
    return [token for sublist in token_list for token in sublist if token != ""]


def tokenize_hyphenated_strings(str_in: List[str]) -> List[str]:
    token_list = []
    for token in str_in:
        split_token = token.split("-")
        token_list.append(token.replace("-", ""))
        if len(split_token) > 1:
            [token_list.append(subtoken) for subtoken in split_token if len(subtoken) > 1]
    return token_list


def split_at_prefix(str_in: str, prefixes: List[str]) -> str:
    for prefix in prefixes:
        str_in = re.sub(fr"\b{prefix}(?P<suffix>\w+)", fr"{prefix} \g<suffix>", str_in)
    return str_in


def handle_special_cases(str_in: str, specialcases: Dict[str, str]) -> str:
    for original_term, modified_term in specialcases.items():
        str_in = f" {str_in} ".replace(f" {original_term} ", f" {modified_term} ")
    return str_in.strip()


def replace_accents(str_in: str) -> str:
    str_in = re.sub(u"[àáâãå]", 'a', str_in)
    str_in = re.sub(u"[èéêë]", 'e', str_in)
    str_in = re.sub(u"[ìíîï]", 'i', str_in)
    str_in = re.sub(u"[òóôõ]", 'o', str_in)
    str_in = re.sub(u"[ùúû]", 'u', str_in)
    str_in = re.sub(u"[ýÿ]", 'y', str_in)
    str_in = re.sub(u"[ñ]", 'n', str_in)
    return str_in


def insert_umlaut(str_in: str, protected_words: List[str]) -> str:
    return_list = []
    for token in str_in.split(" "):
        # TODO why do we do it in this direction and not the other way around?
        if token not in protected_words:
            token = re.sub(u"ae", 'ä', token)
            token = re.sub(u"ue", 'ü', token)
            token = re.sub(u"oe", 'ö', token)
            token = re.sub(u"ß", 'ss', token)
        return_list.append(token)
    return " ".join(return_list)


# TODO exceptions list for o'reilly etc?
def replace_apostrophes_and_quotes(str_in: str):
    pattern = "[\u0027\u0060\u00B4\u2018\u2019\"]"
    return re.sub(pattern, '', str_in)


def lowercase(str_in: str) -> str:
    return str_in.lower()


# https://stackoverflow.com/questions/3411771/best-way-to-replace-multiple-characters-in-a-string
def replace_special_char_with_space(text):
    result = text
    split_tokens = ' \t\n\x0B\f\r+;:©®℗℠™()!#$%&*<=>?@[]^_`{|}~<>»«'
    for char in split_tokens:
        if char in result:
            result = result.replace(char, " ")
    return result


def uniform_spaces_to_whitespace(text):
    word_list = text.split()
    cleaned_text = " ".join([word.strip() for word in word_list])
    return cleaned_text


def replace_all_commas_with_dots(text):
    cleaned_text = re.sub(',', '.', text)
    return cleaned_text


def remove_useless_dots_commas(text):
    cleaned_text = re.sub(r'(?<!\d)[.,;:]|[.,;:](?!\d)', ' ', text)
    return cleaned_text


def remove_html_tags(text):
    try:
        doc = html.fromstring(text)
        return " ".join(etree.XPath("//text()")(doc))
    except etree.ParserError:
        return text

def clean_text_full(text: str, normalization_lists: NormalizationLists) -> List[str]:
    
    if not text:
        return []
    cleaned_text = remove_html_tags(text)  # not in query
    cleaned_text = remove_useless_dots_commas(cleaned_text)
    cleaned_text = replace_all_commas_with_dots(cleaned_text)
    cleaned_text = replace_special_char_with_space(cleaned_text)
    cleaned_text = uniform_spaces_to_whitespace(cleaned_text)
    cleaned_text = lowercase(cleaned_text)
    cleaned_text = replace_apostrophes_and_quotes(cleaned_text)
    cleaned_text = insert_umlaut(cleaned_text, normalization_lists.list_protectedumlauts)
    cleaned_text = replace_accents(cleaned_text)
    cleaned_text = handle_special_cases(cleaned_text, normalization_lists.specialcases)
    cleaned_text = split_at_prefix(cleaned_text, normalization_lists.list_prefixes)
    cleaned_tokens = cleaned_text.split(" ")
    cleaned_tokens = tokenize_hyphenated_strings(cleaned_tokens)
    cleaned_tokens = split_on_slashes_except_fraction(cleaned_tokens)  # not in query
    cleaned_tokens = remove_stopwords(cleaned_tokens, normalization_lists.list_stopwords)
    cleaned_tokens = lemmatize(cleaned_tokens, normalization_lists.dict_lemmatizing)
    cleaned_tokens = stem_tokens(cleaned_tokens)
    cleaned_tokens = remove_empty_and_long_tokens(cleaned_tokens)  # in query before stemming
    # remove duplicates missing here
    return cleaned_tokens


NormalizationFunction = Callable[[DataFrame, str], DataFrame] 

def full_normalization(df, text_col):
    nls = NormalizationLists()
    udf_normalization = F.udf(lambda c: clean_text_full(c, nls), ArrayType(StringType()))
    df = df.withColumn(text_col, udf_normalization(F.col(text_col)))
    return df


def simple_normalization_and_lowercasing(df, text_col):
    REGEX_STRIP = "[^A-Za-z0-9_\s-\u00c4\u00e4\u00d6\u00f6\u00dc\u00fc\u00df]"
    df = df.withColumn(text_col, F.regexp_replace(F.col(text_col), REGEX_STRIP, ""))
    df = df.withColumn(text_col, F.regexp_replace(F.col(text_col), "-", " "))
    df = df.withColumn(text_col, F.trim(F.lower(F.col(text_col))))
    return df
