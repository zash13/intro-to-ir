import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from embeding_model import CBOW, SkipGram


BASE_DIR = os.getcwd()

DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "Text8", "text8.txt")
DATASET_PATH = os.path.normpath(DATASET_PATH)


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
        for word in input.split():
            result.append(self.token_map.get(word, self.token_map["<#UNKNOWN>"]))
        return [self.token_map["<#START>"]] + result + [self.token_map["<#END>"]]

    def binary_vector(self, token_list: list[int]):
        result_list = np.zeros(len(self.token_map), dtype=int)
        for token in token_list:
            if 0 <= token < len(self.token_map):
                result_list[token] = 1

        return result_list

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


def plot_loss(loss_values, filename="loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label="Training Loss", color="blue")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def generate_cbow_skipgram_data(
    tokenizer: "Token", sentences, window_size, vocab_size, model_type="cbow"
):
    # if window size is even
    if window_size % 2 == 0:
        return False, False
    half_window = (window_size - 1) / 2
    half_window = int(half_window)
    cbow_inputs, cbow_targets = [], []
    cbow_inputs2, cbow_targets2 = [], []
    skipgram_inputs, skipgram_targets = [], []

    for sentence in sentences:
        tokenized = tokenizer.tokenize(sentence)
        length = len(tokenized)

        # for example if i have 5 word and window_size is 3 , then i will get
        # pad word0 word1
        # word0 word1 word2
        # word1 word2 word3
        # ....
        # word4 word5 pad
        # while always middle word is target
        # so i will have 5 target (each word have chance to be a target )
        # len - 2 , case i have 2 pad , in beggening and end
        for idx in range(half_window, length - 2):
            context = (
                tokenized[idx - half_window : idx]
                + tokenized[idx + 1 : idx + half_window + 1]
            )
            target = tokenized[idx]

            binary_vector_input = tokenizer.binary_vector(context)
            binary_vector_target = tokenizer.binary_vector([target])
            if model_type == "cbow":
                cbow_inputs.append(binary_vector_input)
                cbow_targets.append(binary_vector_target)
                cbow_inputs2.append(context)
                cbow_targets2.append(target)
            elif model_type == "skipgram":
                for context_word in context:
                    skipgram_inputs.append(binary_vector_target)
                    skipgram_targets.append(binary_vector_input)

    if model_type == "cbow":
        return (cbow_inputs, cbow_targets)
    else:
        return skipgram_inputs, skipgram_targets


# preprocess data
data = []
with open(DATASET_PATH) as f:
    data = f.read()
print(type(data))
# output : <class 'str'>
data_clean = Token.clean_input(data)
words_list = data.lower().split()
unique_words_list = set(words_list)
unique_words_list = sorted(unique_words_list)
vocab = [Token.clean_input(word) for word in unique_words_list]
vocab = list(set(vocab))
vocab_size = len(vocab)
print(vocab_size, vocab[:10])
# output : 253854 ['excellite', 'spins', 'supertoys', 'xdarwin', 'neuharth', 'strettodimessina', 'cowpuncher', 'bloomington', 'rounding', 'operaci']
tokenhelper = Token(vocab)
toekn_map = tokenhelper.get_token_map()
print(type(toekn_map))
print(toekn_map.get("excellite", " "))
# output : <class 'dict'> 187747

# data set hase no abbility to become list of sentences , its just words , without anything that help me to seprate them into sentences
max_word = 20  # words per sentence
chunk_size = 100  # jumber of sentences to process at once


def generate_sentences(words, words_per_sentence):
    for i in range(0, len(words), words_per_sentence):
        yield " ".join(words[i : i + words_per_sentence])


cbow_model = CBOW(
    vocab_size=len(tokenhelper.token_map), window_size=5, embedding_size=500, epoch=20
)

all_loss = []
sentence_generator = generate_sentences(words_list, max_word)

while True:
    chunk = []
    for _ in range(chunk_size):
        try:
            chunk.append(next(sentence_generator))
        except StopIteration:
            break

    if not chunk:
        break

    print(
        f"Processing chunk of {len(chunk)} sentences -> total words : {len(chunk) * len(chunk[0])}"
    )

    cbow_inputs, cbow_targets = generate_cbow_skipgram_data(
        tokenhelper,
        chunk,
        window_size=5,
        vocab_size=len(tokenhelper.token_map),
        model_type="cbow",
    )

    chunk_loss = cbow_model.fit(cbow_inputs, cbow_targets)
    all_loss.extend(chunk_loss)

plot_loss(all_loss, "cbow_loss.png")
y_pred = cbow_model.predict(["hello", "are"])
