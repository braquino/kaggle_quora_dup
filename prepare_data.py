#import numpy as np
import pandas as pd
#from nltk import wordokenize
from util import *
from sklearn.metrics import log_loss
#from nltk.corpus import wordnet as wn
#from nltk.stem.wordnet import WordNetLemmatizer
import spacy

nlp = spacy.load('en_core_web_lg')

def prepare(raw_file, return_file):
    df_master = pd.read_csv(raw_file)
    df_master = df_master.dropna(how='any')
    last = 0
    step = int(len(df_master) / 23)
    for i in range(step, len(df_master), step):

        token_list = df_master.loc[last:i].copy(deep=True)
        for col in ['question1', 'question2']:
            token_list[col] = [nlp(x) for x in token_list[col]]

        token_list['match_sub_root_vec'] = [match_sub_root_vec(row[0], row[1]) for row in token_list[['question1', 'question2']].values]
        token_list['match_words'] = token_list.apply(match_words, axis=1, raw=True)
        token_list['match_sub_root'] = [match_sub_root(row[0], row[1]) for row in token_list[['question1', 'question2']].values]
        token_list['match_ent'] = [match_ent(row[0], row[1]) for row in token_list[['question1', 'question2']].values]
        token_list['match_words_vec'] = token_list.apply(match_words_vec, axis=1, raw=True)
        token_list['match_sent'] = token_list.apply(match_sent, axis=1, raw=True)

        #print(log_loss(token_list['is_duplicate'], token_list['match_words']))
        #print(log_loss(token_list['is_duplicate'], token_list['match_sub_root_vec']))
        #print(log_loss(token_list['is_duplicate'], token_list['match_sub_root']))
        #print(log_loss(token_list['is_duplicate'], token_list['match_ent']))
        #print(log_loss(token_list['is_duplicate'], token_list['match_words_vec']))
        #print(log_loss(token_list['is_duplicate'], token_list['match_sent']))

        token_list.to_csv(return_file + '{}.csv'.format(int(i/step)))
        last += step
def prepare_unique(raw_file, return_file):
    df_master = pd.read_csv(raw_file)
    df_master = df_master.dropna(how='any')
    token_list = df_master.copy(deep=True)
    for col in ['question1', 'question2']:
        token_list[col] = [nlp(x) for x in token_list[col]]

    feats = pd.DataFrame([[r[0], r[1], r[2]] for r in token_list.apply(match_set, axis=1, raw=True)], columns=['match_words', 'match_sub_root', 'match_set'])
    token_list = pd.concat([token_list, feats], axis=1)
    token_list.to_csv(return_file)

#prepare('data/train.csv', 'data/features')
#prepare('data/test.csv', 'data/test_feats')
#prepare_unique('data/train.csv', 'data/prep_train.csv')

