# %load Summarize.py


# In[7]:


from gensim.summarization import summarize
import pandas as pd
from nltk.corpus import brown, stopwords
from nltk.cluster.util import cosine_distance
import math
import numpy as np
from operator import itemgetter 
import re



def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((new_P - P).sum())
        if delta <= eps:
            return new_P
        P = new_P
 


# In[3]:


def vectorizer(sent,vect_len,stopwords,all_words):
    vector = [0] * vect_len
    for w in sent:
        if w in stopwords:
            continue
        vector[all_words.index(w)] += 1
 
    return vector


# In[4]:


def cosine_dist(sent1, sent2, stopwords=None):
    #sentence similary
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
    vect_len = len(all_words)
    
    vector1 = vectorizer(sent1,vect_len,stopwords,all_words)
    vector2 = vectorizer(sent2,vect_len,stopwords,all_words)

    dist = cosine_distance(vector1, vector2)
    
    if math.isnan(dist):
        return 0
    
    return dist


# In[8]:


# get the english list of stopwords
stop_words = stopwords.words('english')
 
def build_similarity_matrix(sentences, stopwords=None):
    # Create an empty similarity matrix
    sent_len = len(sentences)
    S = np.zeros((sent_len, sent_len))
 
    for i in range(sent_len):
        for j in range(sent_len):
            if i == j:
                continue
 
            S[i][j] = cosine_dist(sentences[i], sentences[j], stop_words)
 
    # normalize the matrix row-wise
    for i in range(len(S)):
        S[i] /= S[i].sum()
 
    return S


# In[24]:


def summarizer(sentences, num=10, stopwords=None):

    S = build_similarity_matrix(sentences, stop_words) 
    sentence_ranks = pagerank(S)
 
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:num])
    summary = itemgetter(*selected_sentences)(sentences)
    return summary


# In[25]:


def split_into_sentences(text):
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"


    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


# In[26]:


def text_format(sentences):
    splited = split_into_sentences(sentences)
    word_split = [i.split(" ") for i in splited]
    return word_split


