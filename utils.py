import re
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
stopword_set = stopwords.words('english')
lemma = WordNetLemmatizer()

def remove_keywords(x):
    pattern = r'@\w*\b'
    no_pattern = r'&#\w*\b'
    x = re.sub(no_pattern,'',x)
    x = re.sub(pattern,'',x)
    x = x.replace('RT ','')
    x = x.replace('NFN ','')
    x = re.sub('|!|:|"|;|\'|\.|','',x)
    return x

def remove_stopword_lemmatize(sent):
    sent = word_tokenize(sent)
    new_sent = ''
    for word in sent:
        if word not in stopword_set:
            new_sent+=word+ ' '
    return new_sent

def response(model_val):
    res = tf.squeeze(tf.round(model_val)).numpy()
    if 1 not in res:
        return "Model can't able to decide"
    res_index = np.where(res==1)[0][0]
    response_text = {0:'Hate Speech',
                    1:'Offensive Language',
                    2:'Neither Hate nor Offensive'}
    return response_text[res_index]

def clean_tweet(x):
    x = remove_keywords(x)
    x = remove_stopword_lemmatize(x.lower())
    return x