import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import tokenize
from sklearn.svm import LinearSVC
import sys
import pickle

np.random.seed(42)


def check_deps():
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("punkt")
    return True


def augment_data(df, profession, target_num):
    profiles = df[df["profession"] == profession]["profile"].values
    if len(profiles) > target_num:
        return df
    all_sentences = []
    for profile in profiles:
        sentences = tokenize.sent_tokenize(profile)
        sentences = [sentence for sentence in sentences if len(sentence) > 10]
        all_sentences.extend(sentences)
    new_profiles = []
    np.random.shuffle(all_sentences)
    NUM_SENTENCES = 4
    for i in range(target_num - len(profiles)):
        try:
            new_profile = ".".join(np.random.choice(all_sentences, NUM_SENTENCES))
            new_profiles.append(new_profile)
        except:
            continue
    new_df = pd.DataFrame({"profile": new_profiles, "profession": profession})
    combined_df = pd.concat([df, new_df])
    return combined_df


def clean_data(df):
    profiles = df["profile"]
    cleaned_profiles = []
    for profile in profiles:
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


def get_X_and_y(data_df):
    y = data_df["profession"]
    X_vec = vectorizer.transform(data_df["profile"])
    return X_vec, y


if __name__ == "__main__":
    check_deps()

    TRAIN_DATA_PATH = sys.argv[1]

    stopwords = nltk.corpus.stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    df = pd.read_csv(TRAIN_DATA_PATH)
    df = df.dropna()

    professions = df["profession"].unique()

    df = clean_data(df)

    AUGMENT_DATA = True
    if AUGMENT_DATA:
        max_num = int(max(df["profession"].value_counts()) / 12)
        for profession in professions:
            train_data = augment_data(df, profession, max_num)

    train_data = df
    train_data = remove_punctuation(train_data, "profile")

    MAX_FEATURES = 100000
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES, sublinear_tf=True)
    vectorizer.fit_transform(train_data["profile"])

    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train_vec, y_train = get_X_and_y(train_data)

    model = LinearSVC(C=0.45, penalty="l2", dual=False, max_iter=10000, random_state=42).fit(X_train_vec, y_train)

    with open("2019EE10577.model", "wb") as f:
        pickle.dump((vectorizer, model), f)
