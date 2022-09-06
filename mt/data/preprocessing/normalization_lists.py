import csv
from typing import List, Dict
import os 
from mt.utils import S3Url
import tempfile
import boto3
import io 

class NormalizationLists():

    filename_to_function_mapper = {
        "list_protectedumlauts.txt": "read_protectedumlauts",
        "specialcases.csv": "read_specialcases",
        "list_stopwords.csv": "load_stopwords",
        "list_prefixes.csv": "load_prefixes",
        "dict_lemmatizing.csv": "load_lemmatizing_dict"
    }

    def __init__(self, local_path:str=None) -> None:
        if not local_path:
            from mt.config import config
            self.s3_url = S3Url(config.NORMALIZATION_LISTS_PATH)
            self.load_normalization_lists_from_s3()
        else:
            self.local_path = local_path
            self.load_normalization_lists_from_local_files()

    def load_normalization_lists_from_s3(self):
        
        for filename, fn_name in self.filename_to_function_mapper.items():

            with tempfile.TemporaryFile(mode='w+b') as temp_file:
                client = boto3.client('s3')
                client.download_fileobj(
                    self.s3_url.bucket, 
                    "/".join([self.s3_url.key.rstrip("/"), filename]), 
                    temp_file
                )
                # go back to start of file
                temp_file.seek(0)
                # transform to text mode (required by csv.reader)
                temp_file = io.TextIOWrapper(temp_file, encoding='utf-8')
                fn = getattr(self, fn_name)
                ret_val = fn(temp_file)
                setattr(self, filename.split(".")[0], ret_val)

    def load_normalization_lists_from_local_files(self):
        
        for filename, fn_name in self.filename_to_function_mapper.items():

            with open(os.path.join(self.local_path, filename), "r") as f:
                fn = getattr(self, fn_name)
                ret_val = fn(f)
                setattr(self, filename.split(".")[0], ret_val)

    def read_protectedumlauts(self, f: tempfile.TemporaryFile) -> List[str]:
        reader = csv.reader(f)
        protected_words = list(reader)
        return [word for sublist in protected_words for word in sublist]


    def read_specialcases(self, f: tempfile.TemporaryFile) -> Dict[str, str]:
        reader = csv.reader(f, delimiter=';')
        specialcases = {row[0]: row[1] for row in reader}
        return specialcases


    def load_stopwords(self, f: tempfile.TemporaryFile):
        reader = csv.reader(f)
        stopwords = list(reader)
        stopwords = [word for sublist in stopwords for word in sublist]
        return stopwords


    def load_prefixes(self, f: tempfile.TemporaryFile):
        reader = csv.reader(f)
        prefixes = list(reader)
        prefixes = [word for sublist in prefixes for word in sublist]
        return prefixes


    def load_lemmatizing_dict(self, f: tempfile.TemporaryFile):
        reader = csv.reader(f, delimiter=';')
        lemmatizers = list(reader)
        lemmatizing_dict = {word[0]: word[1] for word in lemmatizers}
        return lemmatizing_dict