import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import tokenize
from tqdm import tqdm
from sklearn.svm import LinearSVC
import sys
import pickle


def check_deps():
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("punkt")
    return True


def clean_data(df):
    profiles = df["profile"]
    cleaned_profiles = []
    for profile in tqdm(profiles):
        profile = re.sub(r"[^a-zA-Z .]", " ", profile)
        profile = profile.lower()
        profile = profile.split()
        profile = [lemmatizer.lemmatize(word) for word in profile if not word in set(stopwords)]
        profile = " ".join(profile)
        cleaned_profiles.append(profile)
    df["profile"] = cleaned_profiles
    return df


def remove_punctuation(df, column):
    df[column] = df[column].str.replace("[^\w\s]", "")
    return df


def get_X(data_df):
    X_vec = vectorizer.transform(data_df["profile"])
    return X_vec


if __name__ == "__main__":
    check_deps()

    stopwords = nltk.corpus.stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    TEST_DATA_PATH = sys.argv[1]
    OUTPUT_FILE_NAME = sys.argv[2]

    df = pd.read_csv(TEST_DATA_PATH)
    df = clean_data(df)
    df = remove_punctuation(df, "profile")

    with open("2019EE10577.model", "rb") as f:
        vectorizer, modelSVC = pickle.load(f)

    X_test = get_X(df)
    y_pred = modelSVC.predict(X_test)

    with open(OUTPUT_FILE_NAME, "w") as f:
        f.write('"' + "profession" + '"' + "\n")
        for pred in y_pred:
            new_pred = '"' + pred + '"'
            f.write(new_pred + "\n")
