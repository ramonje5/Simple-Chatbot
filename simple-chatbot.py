import nltk
import numpy as np
import random
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined text
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "hola")
GREETING_ENDS = ["bye", "adios"]
GREETING_RESPONSES = [
    "hi",
    "hey",
    "hi there",
    "hello",
    "buenas",
    "I am glad! You are talking to me",
]

# Reading the data from a .txt file (EN only)
f = open("library.txt", "r", errors="ignore")
raw = f.read()
raw = raw.lower()  # converts all text to lowercase

sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


# Removes punctuation characters: !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    ramonBOT_response = ""
    sent_tokens.append(user_response)

    #TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    TfidfVec = TfidfVectorizer(stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        ramonBOT_response = ramonBOT_response + "I am sorry! I don't understand you"
        return ramonBOT_response
    else:
        ramonBOT_response = ramonBOT_response + sent_tokens[idx]
        return ramonBOT_response


print(
    "ramonBOT: My name is ramonBOT and I will answer your queries. If you want to exit, type Bye"
)

flag = True
while flag == True:
    user_response = input("USER: ")
    user_response = user_response.lower()
    if user_response not in GREETING_ENDS:
        if user_response == "thanks" or user_response == "thank you":
            flag = False
            print("ramonBOT: You are welcome")
        else:
            if greeting(user_response) != None:
                print("ramonBOT: " + greeting(user_response))
            else:
                print("ramonBOT: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ramonBOT: Bye! See you soon!")
