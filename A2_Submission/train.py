import sys
import os
import re
import string
import json
import urllib.request
import numpy as np
import pickle

from tqdm import tqdm

import torch

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


def read_file(filename):
    with open(filename, "r") as file:
        text = file.readlines()
    return text


def build_train_vocab(data):
    vocab = {}
    num_words = 0
    for line in data:
        split_line = line.split("\t")
        if len(split_line) == 2:
            word = split_line[0]
            word = word.lower()
            if word not in vocab:
                vocab[word] = 1
    return vocab


def get_vector(word):
    word = word.replace("~", "")
    temp = word.replace(",", "")
    temp = temp.replace("-", "")
    if temp.replace(".", "", 1).isdigit():
        return embeddings[NUMERIC_KEY]
    elif word in embeddings:
        return embeddings[word]
    else:
        return np.random.normal(scale=0.6, size=(emb_dim,))


def get_data(data):
    sent_labels = []
    all_labels = []
    sent_idx = []
    all_idx = []
    for line in data:
        split_line = line.split("\t")
        if len(split_line) == 2:
            word = split_line[0]
            tag = split_line[1]
            tag = tag.replace("\n", "")
            word = word.lower()
            if word in vocab:
                sent_idx.append(vocab[word])
            else:
                sent_idx.append(vocab["<unk>"])
            tag_idx = labels[tag]
            sent_labels.append(tag_idx)
        elif line == "\n":
            sent_idx = np.array(sent_idx)
            sent_labels = np.array(sent_labels)
            all_idx.append(sent_idx)
            all_labels.append(sent_labels)
            sent_idx = []
            sent_labels = []
        else:
            print(line)
    return np.asarray(all_idx, dtype=object), np.asarray(all_labels, dtype=object)


def custom_collate(data):

    batch_size = len(data)

    max_len = -1
    for i in range(batch_size):
        if len(data[i][0]) > max_len:
            max_len = len(data[i][0])

    seq_lengths = []
    for i in range(batch_size):
        seq_lengths.append(len(data[i][0]))

    padded_data = []
    padded_labels = []
    mask = []
    for i in range(batch_size):
        padded_data.append(np.pad(data[i][0], (0, max_len - len(data[i][0])), "constant", constant_values=(vocab["<pad>"])))
        padded_labels.append(np.pad(data[i][1], (0, max_len - len(data[i][1])), "constant", constant_values=["37"]))
        mask.append(np.pad(np.ones(len(data[i][0])), (0, max_len - len(data[i][0])), "constant", constant_values=0).astype(bool))

    padded_data = torch.from_numpy(np.array(padded_data))
    padded_labels = torch.from_numpy(np.array(padded_labels))
    mask = torch.from_numpy(np.array(mask))

    return [padded_data, padded_labels, seq_lengths, mask]


class BiLSTMCRF(nn.Module):
    def __init__(self, weights_matrix, hidden_dim, tagset_size):
        super(BiLSTMCRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze=False)
        embedding_dim = weights_matrix.shape[1]
        # self.embedding = nn.Embedding(len(vocab), embedding_dim)
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


def train_one_epoch(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in tqdm(iterator):
        optimizer.zero_grad()
        X, y, seq_lens, mask = batch
        loss = model(X, y, mask)
        predictions = model.predict(X, mask)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def get_scores(preds, gold):
    flatten_preds = []
    flatten_gold = []
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            flatten_preds.append(preds[i][j])
            flatten_gold.append(gold[i][j])
    idx = np.where(np.array(flatten_gold) != 0)[0]
    micro_f1 = f1_score(np.array(flatten_preds)[idx], np.array(flatten_gold)[idx], average="micro")
    macro_f1 = f1_score(np.array(flatten_preds)[idx], np.array(flatten_gold)[idx], average="macro")
    return micro_f1, macro_f1


def train_model(model, epochs):
    loss_function = nn.CrossEntropyLoss(ignore_index=37)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_f1s = []
    val_f1s = []

    best_epoch = 0
    best_val_f1 = 0

    for epoch in range(epochs):
        print("Training Epoch {}".format(epoch))
        training_loss = train_one_epoch(model, trainDataLoader, optimizer, loss_function)
        print("Training Loss: {}".format(training_loss))

        model.eval()

        train_preds = []
        for batch in trainDataLoader:
            X, y, seq_lens, mask = batch
            predictions = model.predict(X, mask)
            train_preds.extend(predictions)
        train_preds = np.array(train_preds, dtype=object)

        train_micro_f1, train_macro_f1 = get_scores(train_preds, trainY)

        val_preds = []
        for batch in valDataLoader:
            X, y, seq_lens, mask = batch
            predictions = model.predict(X, mask)
            val_preds.extend(predictions)
        val_preds = np.array(val_preds, dtype=object)

        val_micro_f1, val_macro_f1 = get_scores(val_preds, valY)

        print("Training Micro F1: {}".format(train_micro_f1))
        print("Training Macro F1: {}".format(train_macro_f1))
        print("Validation Micro F1: {}".format(val_micro_f1))
        print("Validation Macro F1: {}".format(val_macro_f1))

        train_f1 = (train_micro_f1 + train_macro_f1) / 2
        val_f1 = (val_micro_f1 + val_macro_f1) / 2

        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        if val_f1 > best_val_f1:
            print("New Best Model at Epoch {}".format(epoch))
            print("Validation Micro F1: {}".format(val_micro_f1))
            print("Validation Macro F1: {}".format(val_macro_f1))
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model.pt")

        if epoch >= best_epoch + 3:
            break

        model.train()

    return model, train_f1s, val_f1s


if __name__ == "__main__":
    TRAIN_FILE_PATH = sys.argv[1]
    VAL_FILE_PATH = sys.argv[2]

    with open("labels.json") as f:
        labels = json.load(f)

    labels_inv = {v: k for k, v in labels.items()}

    train_data = read_file(TRAIN_FILE_PATH)
    val_data = read_file(VAL_FILE_PATH)

    embeddings = {}
    emb_dim = 50
    with open("glove.6B/glove.6B.50d.txt", "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector

    NUMERIC_KEY = "<numeric>"
    UNK_KEY = "<unk>"

    ADDITIONAL_KEYS = [NUMERIC_KEY, UNK_KEY]

    for k in ADDITIONAL_KEYS:
        embeddings[k] = np.random.normal(scale=0.6, size=(emb_dim,))

    vocab_keys = []
    vocab_keys.append("<unk>")
    vocab_keys.append("<pad>")
    vocab_keys.append("<numeric>")
    vocab = {k: v for v, k in enumerate(vocab_keys)}

    train_vocab = build_train_vocab(train_data)
    idx = len(vocab)
    for word in train_vocab:
        if word not in vocab:
            vocab[word] = idx
            idx += 1

    with open("vocab.json", "w") as fp:
        json.dump(vocab, fp)

    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))

    for i, word in enumerate(vocab):
        # weights_matrix[i] = get_vector(word)
        if word in embeddings:
            weights_matrix[i] = embeddings[word]
        else:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))

    weights_matrix = torch.from_numpy(weights_matrix).float()

    with open("weights_matrix.pt", "wb") as fp:
        pickle.dump(weights_matrix, fp)

    trainX, trainY = get_data(train_data)
    valX, valY = get_data(val_data)

    trainData = []
    valData = []
    for i in range(len(trainX)):
        trainData.append((trainX[i], trainY[i]))
    for i in range(len(valX)):
        valData.append((valX[i], valY[i]))
    trainData = np.array(trainData, dtype=object)
    valData = np.array(valData, dtype=object)

    trainDataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

    ner = BiLSTMCRF(weights_matrix, 256, 38)

    ner, train_f1s, val_f1s = train_model(ner, 30)

    ner.load_state_dict(torch.load("best_model.pt"))
    torch.save(ner, "ee1190577_model.pt")
