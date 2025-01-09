import nltk
import  numpy as np

# Download the 'punkt_tab' tokenizer model if you haven't already
nltk.download('punkt_tab')


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentences, words=None):
    sentence_word = [stem(word) for word in tokenized_sentences]
    # initialize bag with all zeros
    bag = np.zeros(len(words), dtype=np.float32)
    for index, w in enumerate(words):
        if w in sentence_word:
            bag[index] = 1

    return bag


