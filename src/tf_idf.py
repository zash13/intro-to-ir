import os
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from pandas.io.formats.format import math

# ------------- global variables  -------------
#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "04.testset")
TF_IDF_FILE_PATH = os.path.join(BASE_DIR, "..", "dataset", "tf_idf.csv")
DATASET_PATH = os.path.normpath(DATASET_PATH)
TF_IDF_FILE_PATH = os.path.normpath(TF_IDF_FILE_PATH)


# ------------- cleaning the dataset -------------
#
def create_block(data):
    clean_xml = []
    for line in data.splitlines():
        strriped = line.strip()
        if strriped:
            clean_xml.append(strriped)
    data = "".join(clean_xml)
    xml_block = (
        "<top>\n"
        + data.replace("<num>", "  <num>")
        .replace("<title>", "\n  </num>\n  <title>")
        .replace("<desc>", "\n  </title>\n  <desc>")
        .replace("<narr>", "\n  </desc>\n  <narr>")
        .replace("</top>", "\n  </narr>\n</top>")
        + "\n"
    )
    return xml_block


def preprocess_dataset(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        dataset = file.read()
    dataset_blocks = dataset.split("<top>")
    xml_dataset = []
    for block in dataset_blocks:
        if "<num>" not in block:
            continue
        xml_block = create_block(block)
        xml_dataset.append(xml_block)
    xml_dataset = "<root>\n" + "".join(xml_dataset) + "\n </root>"
    dir_name, base_name = os.path.split(file_path)
    new_file_name = "xml_" + base_name + ".xml"
    xml_dataset_file_path = os.path.join(dir_name, new_file_name)
    with open(xml_dataset_file_path, mode="w", encoding="utf-8") as file:
        file.write(xml_dataset)
    return xml_dataset_file_path


DATASET_PATH = preprocess_dataset(DATASET_PATH)


# ------------- read  dataset  -------------
#


def read_dataset(file_path):
    dataset = []
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    soup = BeautifulSoup(data, "lxml-xml")

    for top in soup.find_all("top"):
        num = top.find("num").get_text(strip=True).replace("Number:", "").strip()
        title = top.find("title").get_text(strip=True)
        desc = top.find("desc").get_text(strip=True).replace("Description:", "").strip()
        narr = top.find("narr").get_text(strip=True).replace("Narrative:", "").strip()
        data = {"num": num, "title": title, "desc": desc, "narr": narr}
        dataset.append(data)
    return dataset


data = read_dataset(DATASET_PATH)
data = pd.DataFrame(data)
""" 
print(data.info()) : 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 250 entries, 0 to 249
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   num     250 non-null    object
 1   title   250 non-null    object
 2   desc    250 non-null    object
 3   narr    250 non-null    object
dtypes: object(4)
memory usage: 7.9+ KB
None
"""


# ------------- helpers -------------
#


class Token:
    def __init__(self, vocab) -> None:
        self.special_tokens = {
            "<#START>": 0,
            "<#PAD>": 1,
            "<#UNKNOWN>": 2,
            "<#END>": 3,
        }
        self.token_map = self._generate_token_map(vocab)

    def _generate_token_map(self, vocab) -> dict[str, int]:
        token_map = {
            word: (idx + len(self.special_tokens)) for idx, word in enumerate(vocab)
        }
        return {**self.special_tokens, **token_map}

    def get_token_map(self):
        return self.token_map

    def tokenize(self, input):
        result = []
        input = input.lower()
        input = self.clean_input(input)

        for word in input.split():
            result.append(self.token_map.get(word, self.token_map["<#UNKNOWN>"]))
        return [self.token_map["<#START>"]] + result + [self.token_map["<#END>"]]

    @staticmethod
    def clean_input(input):
        input = (
            input.replace(",", "")
            .replace("!", "")
            .replace("?", "")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "")
        )
        return input


class CSVStorage:
    @staticmethod
    def save(df, filename, index=False):
        df.to_csv(filename, index=index)

    @staticmethod
    def load(filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist")
        return pd.read_csv(filename)


# ------------- general setup  -------------
# generaly i am combinning both title and Description
#
combind_title_desc = data["title"] + " " + data["desc"]
words_in_cols = combind_title_desc.str.lower().str.split()
all_words = [word for sublist in words_in_cols for word in sublist]
unique_words = set(all_words)
unique_words = sorted(unique_words)
vocab = [Token.clean_input(word) for word in unique_words]
vocab = set(vocab)
vocab_size = len(vocab)
tokenhelper = Token(vocab)
"""
print(vocab_size) :
1687
"""
# ------------- tf -------------
#
tf = []
for idx, doc in combind_title_desc.str.lower().items():
    tf_doc = {word: doc.count(word) for word in vocab}
    tf_doc["doc_id"] = idx
    tf.append(tf_doc)
tf = pd.DataFrame(tf)
cols = ["doc_id"] + [col for col in tf.columns if col != "doc_id"]
tf = tf[cols]
""" 
print(tf.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 250 entries, 0 to 249
Columns: 1688 entries, doc_id to investigationsof
dtypes: int64(1688)
memory usage: 3.2 MB
None
"""
# ------------- df  -------------
#
df = {word: 0 for word in vocab}
lower_docs = combind_title_desc.str.lower()
for word in vocab:
    for doc in lower_docs:
        if word in doc:
            df[word] += 1
df = pd.DataFrame(list(df.items()), columns=["word", "df"])
"""
print(df.info())
print(df.head(10))
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1687 entries, 0 to 1686
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   word    1687 non-null   object
 1   df      1687 non-null   int64 
dtypes: int64(1), object(1)
memory usage: 26.5+ KB
None
                 word  df
0             testing   1
1           personnel   2
2           poisoning   2
3          repeatedly   1
4  members'activities   1
5           republics   1
6                  as  97
.
.
"""
# ------------- idf  -------------
#
N = len(combind_title_desc)
idf = {}
for word in vocab:
    word_df = df.loc[df["word"] == word, "df"]
    if not word_df.empty:
        idf[word] = math.log(N / (1 + word_df.values[0]))
    else:
        idf[word] = math.log(N / 1)
idf = pd.DataFrame(list(idf.items()), columns=["word", "idf"])
""" 
print(idf.head(4))
print(idf.info())
          word       idf
0  planningand  5.521461
1     exposure  4.828314
2      specify  4.828314
3    elections  4.828314
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1687 entries, 0 to 1686
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   word    1687 non-null   object 
 1   idf     1687 non-null   float64
dtypes: float64(1), object(1)
memory usage: 26.5+ KB
None
"""

# ------------- tf_idf  -------------
#
try:
    tf_idf = CSVStorage.load(TF_IDF_FILE_PATH)
except FileNotFoundError as _:
    print("tf_idf file not exist , try to calculate it ")
    tf_idf_rows = []
    for idx, doc_id in enumerate(tf["doc_id"]):
        tf_idf_doc = {"doc_id": doc_id}
        for word in vocab:
            word_tf = tf.loc[tf["doc_id"] == doc_id, word].values[0]
            word_idf = idf.loc[idf["word"] == word, "idf"].values[0]
            tf_idf_doc[word] = word_tf * word_idf

        tf_idf_rows.append(tf_idf_doc)

    tf_idf = pd.DataFrame(tf_idf_rows)
    cols = ["doc_id"] + [col for col in tf_idf.columns if col != "doc_id"]
    tf_idf = tf_idf[cols]
    CSVStorage.save(tf_idf, TF_IDF_FILE_PATH, False)

"""
its taking a year to calculate everything ,  but i am not in optimizing mode , so 
if you want to run it faster , keep them dict and work there , or even easyear , use libs !! , 
anyway , i try to store it , its taking so long 
print(tf_idf.head(4))
print(tf_idf.info())
   doc_id  determine  out  ...  for  connection  against
0       0        0.0  0.0  ...  0.0         0.0      0.0
1       1        0.0  0.0  ...  0.0         0.0      0.0
2       2        0.0  0.0  ...  0.0         0.0      0.0
3       3        0.0  0.0  ...  0.0         0.0      0.0

[4 rows x 1688 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 250 entries, 0 to 249
Columns: 1688 entries, doc_id to against
dtypes: float64(1687), int64(1)
memory usage: 3.2 MB
None
"""
