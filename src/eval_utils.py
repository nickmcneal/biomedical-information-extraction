import pickle
import pandas as pd
import os
from transformers import GPT2Tokenizer
import openai
import numpy as np
import ipdb
import re
import tqdm
import conlleval

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import spacy
from data_utils import *
from scipy import special

nlp = spacy.load("en_core_web_sm")

def create_bio_preds(df, pred_name):
    """Function to create a BIO Tag from GPT-3 Predicted entities"""
    
    bio_preds = []
    post_processed_ents_col = []

    for i, row in df.iterrows():
        try:
            sent = row['orig_tok_sent'].lower()
        except:
            sent = row['sents'].lower()
            
        bio_tags = row['ner_seq']
        predicted_entities = [p.strip() for p in row[pred_name]]
        
        post_predicted_ents = post_processing(sent, predicted_entities)
        post_processed_ents_col.append(post_predicted_ents[:])

        #Sort by Length, Longest to Shortest
        pred_ent_inds_by_length = np.argsort([len(e) for e in post_predicted_ents], kind='mergesort')[::-1]
        post_predicted_ents = np.array(post_predicted_ents)[pred_ent_inds_by_length]
        
        bio_pred_seq = ' '+sent+' '
        
        for pred_ent in post_predicted_ents:
            pred_ent = token_preprocessing(' '.join([s.text for s in nlp.tokenizer(str(pred_ent))]))

            pred_bios = ['I|||' for _ in pred_ent.split()]
            pred_bios[0] = 'B|||'
            
            pred_bios = ' '.join(pred_bios)
            
            bio_pred_seq = bio_pred_seq.replace(' '+pred_ent+' ', ' '+pred_bios+' ')
        
        bio_pred_seq = ' '.join(['O' if (w != 'B|||' and w != 'I|||') else w for w in bio_pred_seq.split()])
        bio_pred_seq = bio_pred_seq.replace('|','')
        bio_pred_seq = bio_pred_seq.strip()
        
        assert len(bio_tags.split()) == len(bio_pred_seq.split()), ipdb.set_trace()
        bio_preds.append(bio_pred_seq)
        
    df['bio_preds'] = bio_preds
    df[pred_name+'.post'] = post_processed_ents_col

    return df

def evaluate_preds(df, pred_name):
    
    tp = 0
    fp = 0
    fn = 0

    post_processed_ents_col = []
    
    for i, row in df.iterrows():
        sent = row['sents']
        true_ents = row['entities']
        predicted_entities = row[pred_name]
    
        post_predicted_ents = post_processing(sent, predicted_entities)
        post_processed_ents_col.append(post_predicted_ents[:])
        
        #Metric is not equivalent to CONLL since it is possible
        #for GPT-3 to generate a phrase "A B C" and also another phrase "B C"
        for ent_ind in range(len(true_ents)):
            ent = true_ents[ent_ind]
            
            if ent in post_predicted_ents:
                tp += 1
                post_predicted_ents.pop(post_predicted_ents.index(ent))
            else:
                fn += 1
                
        fp += len(post_predicted_ents)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)
    
    print(f1)
    print("precision: {}  recall: {}  tp: {}  fp: {} fn: {}".format(precision, recall, tp, fp, fn))
    
    df[pred_name+'.post'] = post_processed_ents_col

    return df, f1, precision, recall

def post_processing(sentence, predicted_ents):
    post_predicted_ents = []
    
    for ent in predicted_ents:
        ent = ent.replace(',','')

        #Tokenizing generated text in same way as original dataset
        ent = ' '.join([s.text for s in nlp.tokenizer(ent)])
            
        #Removing phrases which are not standalone in sentence
        if ' '+ent+' ' in ' '+sentence.lower()+' ':
            post_predicted_ents.append(ent)
            
    return list(set(post_predicted_ents))

def conlleval_eval(true, preds):
    true = [[t+'-X' for t in s.split()] for s in true]
    preds = [[t+'-X' for t in s.split()] for s in preds]
    true = np.concatenate(true)
    preds = np.concatenate(preds)

    prec, recall, f1 = conlleval.evaluate(true,preds)
    
    display(pd.DataFrame([(f1,prec,recall)],columns=['F1','Precision','Recall']))
    
    return f1, prec, recall

def recalibrate_positive_entity_threshold(df, positive_entity_threshold = 0.4):
    
    gold_pred_col = []
    unfiltered_entities_probs = []

    for i,row in df.iterrows():
        entities = row['unfiltered_' + "predictions"]
        entity_probs = row['gpt3_output_' + "predictions"][1]
        true_ents = row['entities']

        filtered_entities = []
        for entity in entities:
            if entity_probs[entity][1] > positive_entity_threshold:
                filtered_entities.append(entity)
            if entity in true_ents:
                unfiltered_entities_probs.append((entity,entity_probs[entity][0],entity_probs[entity][1],1))
            else:
                unfiltered_entities_probs.append((entity,entity_probs[entity][0],entity_probs[entity][1],0))

        gold_pred_col.append(filtered_entities)

    df["predictions"] = gold_pred_col
    
    return df

def extract_yes_no_probs(entity_list, predicted_sample):
    predicted_text = predicted_sample['choices'][0]['text']
    print('GPT3 Output:' + predicted_text)
    entity_probs = []
    predicted_lines = predicted_text.split('\n')
    
    curr_token = 0
    for i, phrase_to_pred in enumerate(entity_list):
    
        predicted_line = predicted_lines[i]
        tokens = tokenizer.encode(predicted_line)
        curr_token += len(tokens)

        if i == 0:
            phrase, prediction = phrase_to_pred, predicted_line
        else:
            phrase, prediction = predicted_line.split(':')
        
        try:
            yes_no_dict = dict(predicted_sample['choices'][0]['logprobs']['top_logprobs'][curr_token-1])
            prob = calc_prob(yes_no_dict)
        except:
            ipdb.set_trace()
        entity_probs.append((phrase_to_pred.replace('"',''), prob[0], prob[1]))
        
        curr_token += 1
        
    return entity_probs
    
def calc_prob(yes_no_dict):
    entity_logits = []
    for opt in ['No','Yes']:
        entity_logits.append(yes_no_dict[opt])

    return special.softmax(entity_logits)

def recalibrate_positive_entity_threshold(df, filename_root, positive_entity_threshold = 0.4):
    
    gold_pred_col = []
    unfiltered_entities_probs = []

    for i,row in df.iterrows():
        entities = row['unfiltered_' + filename_root]
        entity_probs = row['gpt3_output_' + filename_root][1]
        true_ents = row['entities']

        filtered_entities = []
        for entity in entities:
            if entity_probs[entity][1] > positive_entity_threshold:
                filtered_entities.append(entity)
            if entity in true_ents:
                unfiltered_entities_probs.append((entity,entity_probs[entity][0],entity_probs[entity][1],1))
            else:
                unfiltered_entities_probs.append((entity,entity_probs[entity][0],entity_probs[entity][1],0))

        gold_pred_col.append(filtered_entities)

    df[filename_root] = gold_pred_col
    
    return df, unfiltered_entities_probs