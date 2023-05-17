import sys
import os
import re
import string
import json
import urllib.request
import numpy as np
from tqdm import tqdm
import torch
import pickle

torch.manual_seed(42)
np.random.seed(42)
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchcrf import CRF
from sklearn.metrics import f1_score

torch.manual_seed(1)
device = torch.device("cpu")

BATCH_SIZE = 32
NUMERIC_KEY = "<numeric>"
UNK_KEY = "<unk>"
PAD_KEY = "<pad>"
CONC_KEY = "<conc>"
TILDA_KEY = "<tilda>"
TILDA_NUM_KEY = "<til_num>"
SPEED_KEY = "<speed>"


def read_file(filename):
    with open(filename, "r") as file:
        text = file.readlines()
    return text


def check_if_conc(word):
    # check if word is a concentration
    if re.match(r"[a-zA-Z]*\/[a-zA-Z]*", word):
        return True
    elif word == "%":
        return True
    return False


def check_numeric(word):
    word = word.replace(",", "")
    word = word.replace("-", "", 1)
    word = word.replace(".", "", 1)
    if word.isdigit():
        return True
    return False


def check_if_speed(word):
    if "xg" in word:
        return True
    elif "rpm" in word:
        return True
    return False


def get_idx_inference(word, vocab):
    if "~" in word:
        temp = word.replace("~", "")
        if check_numeric(temp):
            return vocab[TILDA_NUM_KEY]
        else:
            return vocab[TILDA_KEY]
    elif check_if_speed(word):
        return vocab[SPEED_KEY]
    elif check_numeric(word):
        return vocab[NUMERIC_KEY]
    elif check_if_conc(word):
        return vocab[CONC_KEY]
    elif word in vocab:
        return vocab[word]
    else:
        return vocab[UNK_KEY]


def get_data(data):
    sent_idx = []
    all_idx = []
    for line in data:
        if line != "\n":
            word = line.split("\t")[0]
            word = word.strip()
            word = word.lower()
            sent_idx.append(get_idx_inference(word, vocab))
        else:
            sent_idx = np.array(sent_idx)
            all_idx.append(sent_idx)
            sent_idx = []
    return np.asarray(all_idx, dtype=object)


class BiLSTMCRF(nn.Module):
    def __init__(self, weights_matrix, hidden_dim, tagset_size):
        super(BiLSTMCRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze=False)
        embedding_dim = weights_matrix.shape[1]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.dropout_layer = nn.Dropout(p=0.5)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentence, labels, mask):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout_layer(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return -self.crf(emissions, labels, mask=mask)

    def predict(self, sentence, mask):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout_layer(lstm_out)
        scores = self.hidden2tag(lstm_out)
        return self.crf.decode(scores, mask=mask)


def custom_collate(data):

    batch_size = len(data)

    max_len = -1
    for i in range(batch_size):
        if len(data[i]) > max_len:
            max_len = len(data[i])

    seq_lengths = []
    for i in range(batch_size):
        seq_lengths.append(len(data[i]))

    padded_data = []
    mask = []
    for i in range(batch_size):
        padded_data.append(
            np.pad(
                data[i],
                (0, max_len - len(data[i])),
                "constant",
                constant_values=(vocab["<pad>"]),
            )
        )
        mask.append(
            np.pad(
                np.ones(len(data[i])),
                (0, max_len - len(data[i])),
                "constant",
                constant_values=0,
            ).astype(bool)
        )

    padded_data = torch.from_numpy(np.array(padded_data))
    mask = torch.from_numpy(np.array(mask))

    return [padded_data, seq_lengths, mask]


if __name__ == "__main__":
    TEST_DATA_FILE = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]

    with open("labels.json") as f:
        labels = json.load(f)

    labels_inv = {v: k for k, v in labels.items()}

    test_data = read_file(TEST_DATA_FILE)

    with open("vocab.json") as f:
        vocab = json.load(f)

    testX = get_data(test_data)

    matrix_len = len(vocab)
    with open("weights_matrix.pt", "rb") as f:
        weights_matrix = pickle.load(f)

    model = BiLSTMCRF(weights_matrix, 256, 38)

    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    testDataLoader = DataLoader(
        testX, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate
    )

    test_preds = []
    for batch in testDataLoader:
        X, seq_lens, mask = batch
        predictions = model.predict(X, mask)
        test_preds.extend(predictions)
    test_preds = np.array(test_preds, dtype=object)

    with open(OUTPUT_FILE, "w") as f:
        for i in range(len(test_preds)):
            for j in range(len(test_preds[i])):
                f.write(labels_inv[test_preds[i][j]] + "\n")
            f.write("\n")
