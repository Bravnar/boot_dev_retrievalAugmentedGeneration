from nltk.stem import PorterStemmer
import string


def preprocess_string(token_string, stopwords=None):
    tokens = (
        token_string.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .split()
    )
    if stopwords is not None:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t not in stopwords]

    return tokens
