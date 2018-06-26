from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import pickle
import spacy
import numpy as np
import math
from nltk.stem import PorterStemmer
from nltk import ngrams
from functools import reduce
nlp = spacy.load('en_core_web_lg')
with open('data/stem_count.pkl', 'rb') as f:
    stem_rating = pickle.load(f)
stemmer = PorterStemmer()

def match_words_vec(row):
    choosen_POS = ['PROPN', 'VERB', 'NOUN', 'SYM', 'NUM']
    text1 = [t for t in row['question1'] if t.pos_ in choosen_POS]
    text2 = [t for t in row['question2'] if t.pos_ in choosen_POS]
    count = 0
    for text_x in text1:
        for text_y in text2:
            try:
                if text_x.similarity(text_y) > 0.9:
                    count += 1
                    break
            except:
                continue
    try:
        return count / len(text1)
    except:
        return 0

def match_sub_root(q1, q2):
    result = 0
    for t1 in q1.noun_chunks:
        for t2 in q2.noun_chunks:
            if t1.root.lemma_ == t2.root.lemma_ and t1.text.lower() == t2.text.lower():
                result = 1
                break

    return result

def match_chunk(q1, q2):
    result = 0
    for t1 in q1.noun_chunks:
        for t2 in q2.noun_chunks:
            if t1.similarity(t2) > 0.8:
                result += 1
                break
    try:
        result = result / (len(list(q1.noun_chunks)) + len(list(q2.noun_chunks)) - result)
    except:
        return 0
    return result

def num_verbs(q1, q2):
    v1 = len([x for x in q1 if x.pos_ == 'VERB'])
    v2 = len([x for x in q2 if x.pos_ == 'VERB'])
    try:
        result = np.clip(abs(v1 - v2) * 2 / (v1 + v2), 0, 1)
    except:
        result = 0
    return result

def word_match(q1, q2):
    choosen_POS = ['PROPN', 'VERB', 'NOUN', 'SYM', 'NUM']
    result = 0
    for t1 in q1:
        for t2 in q2:
            try:
                if t1.similarity(t2) > 0.9 and t1.pos_ in choosen_POS:
                    result += 1
                    break
            except:
                pass
    try:
        result = result / (len(q1) + len(q2) - result)
    except:
        result = 0
    return result

def word_impotance(q1, q2):

    q1 = list(set([stemmer.stem(x.text) for x in q1 if stemmer.stem(x.text) in stem_rating]))
    q2 = list(set([stemmer.stem(x.text) for x in q2 if stemmer.stem(x.text) in stem_rating]))
    match_rating = 0
    for t1 in q1:
        for t2 in q2:
            if t1 == t2:
                match_rating += stem_rating[t1]
                break
    total_rating = 0
    list_words = [stem_rating[t] for t in q1] + [stem_rating[t] for t in q2]
    for word in list_words:
        total_rating += word
    result = match_rating / total_rating
    if math.isnan(result):
        return 0
    else:
        return np.clip(result * 2, 0, 1)

def match_ngram(q1, q2):
    q1s = list([stemmer.stem(x.text) for x in q1])
    q2s = list([stemmer.stem(x.text) for x in q2])
    result = 0
    rating = {3: 0.33, 4: 0.66, 5: 1}
    for n in [3, 4, 5]:
        match = False
        for ng1 in ngrams(q1s, n):
            if match:
                break
            for ng2 in ngrams(q2s, n):
                if ng1 == ng2:
                    result = rating[n]
                    match = True
                    break
    return result

def match_set(row):
    q1 = row['q1_token']
    q2 = row['q2_token']
    result = {'word_match': 0, 'match_sub_root': 0, 'match_set': 0, 'n_words': 0,
              'chunk_sim': 0, 'n_verb': 0, 'word_importance': 0, 'match_ngram': 0}
    # calc numbers of words that have vector similarity
    result['word_match'] = word_match(q1, q2)
    # verify the similarity of the vectors of two sentences
    result['match_set'] = q1.similarity(q2)
    # calculate if the number of words are close
    result['n_words'] = np.clip(abs(len(q1) - len(q2)) * 2 / (len(q1) + len(q2)), 0, 1)
    # verify if the subject and the root verb lemma are equals
    result['match_sub_root'] = match_sub_root(q1, q2)
    # calculate the similarity of each chunk
    result['chunk_sim'] = match_chunk(q1, q2)
    # verify if is the same number of verbs
    result['n_verb'] = num_verbs(q1, q2)
    # give a greater rating for major rarity, and see weighty the match of the words
    result['word_importance'] = word_impotance(q1, q2)
    result['match_ngram'] = match_ngram(q1, q2)
    return result['word_match'], result['match_sub_root'], result['match_set'], result['n_words'], \
           result['chunk_sim'], result['n_verb'], result['word_importance'], result['match_ngram']

def create_features(file, destination, n_files, sample=False):

    df = pd.read_csv(file)
    df = df.dropna(how='any')
    if sample:
        df = df.head(sample)
    heads = list(df) + ['q1_token', 'q2_token']
    break_num = int(len(df) / n_files)
    new_file = []
    for i, row in enumerate(df.values):
        token_q1 = nlp(row[3])
        token_q2 = nlp(row[4])
        new_file.append([c for c in row] + [token_q1, token_q2])
        if (i % break_num == 0 and i > 0) or i == len(df) - 1:
            file_n = int(i / break_num)
            if i == len(df) - 1:
                file_n += 1
            print('Saving file {}, {:.2%} completed!'.format(file_n, i / len(df)))
            export = pd.DataFrame(new_file, columns=heads)
            new_file = []
            feats = pd.DataFrame([[r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]] for r in export.apply(match_set, axis=1, raw=True)],
                                 columns=['match_words', 'match_sub_root', 'match_set', 'n_words', 'chunk_sim', 'n_verb', 'word_importance', 'match_ngram'])
            export = pd.concat([export, feats], axis=1)
            export.to_csv(destination.replace('.csv', '_{:02d}.csv'.format(file_n)))
            del export
            del feats

def word_importance_file():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    stem_count = {}
    for col in [df_train['question1'], df_train['question2'], df_test['question1'], df_test['question2']]:
        for row in col:
            try:
                temp_row = ''.join([l for l in row if l not in punctuation])
            except:
                continue
            for word in temp_row.split():
                stem_word = stemmer.stem(word)
                try:
                    stem_count[stem_word] += 1
                except:
                    stem_count[stem_word] = 1

    max_count = max(stem_count.values())
    max_repeat = int(max_count / 75)

    for key in stem_count:
        stem_count[key] = np.clip(1 - (stem_count[key] / max_repeat), 0, 1)

    with open('data/stem_count.pkl', 'wb') as f:
        pickle.dump(stem_count, f)


create_features('data\\train.csv', 'data\\train\\train.csv', 20)
#word_importance_file()


