import pandas as pd
import numpy as np
import ipdb
import pickle
import re
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def create_prompt_dataset(few_shot_prompt, test_df, seed, few_shot_num, number_of_test_samples, selection_strategy,sep=', '):
    
    random = np.random.RandomState(seed)
        
    test_df['test_ready_prompt'] = [few_shot_prompt+'\n\n'+empty_prompt for empty_prompt in test_df['empty_prompts']]
    
    if number_of_test_samples != 'all':
        #Making sure the samples are the same as the first batch
        random = np.random.RandomState(42)
        chosen_test_ids = random.permutation(test_df.index)[:number_of_test_samples]
        
        chosen_test_df = test_df.loc[chosen_test_ids]
    else:
        chosen_test_df = test_df
    
    return {'seed':seed,'few_shot_prompt': few_shot_prompt, 'chosen_prompt_ids':chosen_prompt_ids,'sep': sep,'test_df': chosen_test_df}

def create_prompt_dataset(train_df, test_df, seed, few_shot_num, number_of_test_samples, selection_strategy,sep=', '):
    
    random = np.random.RandomState(seed)
    
    few_shot_prompt, chosen_prompt_ids = create_few_shot_prompt(train_df, random, few_shot_num, selection_strategy)
    
    test_df['test_ready_prompt'] = [few_shot_prompt+'\n\n'+empty_prompt for empty_prompt in test_df['empty_prompts']]
    
    if number_of_test_samples != 'all':
        #Making sure the samples are the same as the first batch
        random = np.random.RandomState(42)
        chosen_test_ids = random.permutation(test_df.index)[:number_of_test_samples]
        
        chosen_test_df = test_df.loc[chosen_test_ids]
    else:
        chosen_test_df = test_df
    
    return {'seed':seed,'few_shot_prompt': few_shot_prompt, 'chosen_prompt_ids':chosen_prompt_ids,'sep': sep,'test_df': chosen_test_df}

def create_few_shot_prompt(train_df, random, few_shot_num, selection_strategy):
    
    if selection_strategy == 'random':
        chosen_inds = random.permutation(train_df.index)[:few_shot_num]
    
    few_shot_prompt_list = train_df.loc[chosen_inds, 'prompts'].values
    few_shot_prompt = '\n\n'.join(few_shot_prompt_list)
    
    return few_shot_prompt, chosen_inds

def reverse_token_preprocessing(sent):
    sent = re.sub(" (?=[\'\)\]\,\-\.\%\:\;\/])","", sent)
    sent = re.sub("(?<=[\'\-\(\[\.\/]) ","", sent)
    
    return sent

def token_preprocessing(sent):
    sent = re.sub("(?=[\'\)\]\,\-\.\%\:\;\/])"," ", sent)
    sent = re.sub("(?<=[\'\-\(\[\.\/])"," ", sent).strip()
    
    return sent

def get_embeddings(sent, model, tokenizer, mode='cls'):
    if mode == 'sbert':
        embedding = model.encode(sent)
    else:
        input_dict = tokenizer(sent, return_tensors='pt').to('cuda')
        embedding = model(**input_dict)['last_hidden_state'].cpu().numpy()[0]

        if mode == 'cls':
            embedding = embedding[0]
        elif mode == 'avg':
            embedding = np.mean(embedding,axis=0)

    return embedding


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1