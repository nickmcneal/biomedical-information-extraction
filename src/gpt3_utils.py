import pickle
import pandas as pd
import os
import openai
import numpy as np
import ipdb
import re
from tqdm import tqdm

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import spacy
import scipy

openai.api_key= os.environ['OPENAI_KEY']

def run_gpt3(engine, prompt, logit_bias_text, max_tokens=10, sep=None, logit_bias=0.1):
    #constrain potential output to tokens in test sentence using logit bias
    logit_biases = {}
    tokens = tokenizer.encode(logit_bias_text)
    for token in tokens:
        logit_biases[token] = logit_bias
    
    if sep is not None:
        sep_token = tokenizer.encode(sep)
        logit_biases[sep_token[0]] = logit_bias         
    
    #use API to generate completion
    sample = openai.Completion.create(engine=engine,
                                      prompt=prompt,
                                      max_tokens=max_tokens,
                                      temperature=0,
                                      logit_bias = logit_biases,
                                      presence_penalty=-0.001,
                                      stop=["\n", "<|endoftext|>"])
    return sample

def run_gpt3_ner(engine, 
                 prompt, 
                 logit_bias_text, 
                 max_tokens=10, 
                 sep=None, 
                 logit_bias=0.1, 
                 sep_logit_bias=0.1, 
                 new_line_logit_bias=0.1):
    
    #constrain potential output to tokens in test sentence using logit bias
    logit_biases = {}
    tokens = tokenizer.encode(logit_bias_text)
    for token in tokens:
        logit_biases[token] = logit_bias
    
    #Adding bias for separator
    if sep is not None:
        sep_token = tokenizer.encode(sep)
        logit_biases[sep_token[0]] = sep_logit_bias         

    #Adding bias for newline (token id 198)
    logit_biases[198] = new_line_logit_bias 
    
    #use API to generate completion
    sample = openai.Completion.create(engine=engine,
                                      prompt=prompt,
                                      max_tokens=max_tokens,
                                      temperature=0.0,
                                      logit_bias = logit_biases,
                                      logprobs=1,
                                      presence_penalty=-0.001,
                                      stop=["\n", "<|endoftext|>"])
    return sample

def run_gpt3_ner_post_filtering(engine, 
                                     filtering_prompt,
                                     filter_template,
                                    entities):
    
    logit_biases = {}
    tokens = ['Yes','No']
    
    for token in tokens:
        token_id = tokenizer.encode(token)[0]
        logit_biases[token_id] = 10
    
    #Adding bias for newline (token id 198)
    logit_biases[198] = -10 

    entity_probs = {}
    
    for entity in entities:
        filter_prompt = filtering_prompt + filter_template.format(entity)
        filter_sample = openai.Completion.create(engine=engine,
                                          prompt=filter_prompt,
                                          max_tokens=1,
                                          temperature=0.0,
                                          logprobs=2,
                                          logit_bias = logit_biases,
                                          stop=["\n", "<|endoftext|>"])
        filter_dict = dict(filter_sample['choices'][0]['logprobs']['top_logprobs'][0])
        
        entity_logits = []
        for opt in ['No','Yes']:
            entity_logits.append(filter_dict[opt])
        
        entity_probs[entity] = scipy.special.softmax(entity_logits)

    return entity_probs

def run_gpt3_post_filtering_only(engine, 
                                 filter_template,
                                 prompts,
                                 test_entities):
    """
    engine: ada, davinci, etc.
    filter_template: string template with two '{}' slots, one for the entity and the other for the label 
    prompts: List where each element should be a list of tuples of the form (entity, label) which will be used to make the prompts for each entity to classify. 
    """
    
    logit_biases = {}
    tokens = ['Yes','No']
    
    for token in tokens:
        token_id = tokenizer.encode(token)[0]
        logit_biases[token_id] = 10
    
    #Adding bias for newline (token id 198)
    logit_biases[198] = -10 

    entity_probs = {}
    
    for prompt_entities_labels, entity in zip(prompts,test_entities):
        
        filtering_prompt = []
        
        prompt_entity,label,score = prompt_entities_labels
        filtering_prompt.append(filter_template.format(entity,label))
        
        filtering_prompt = '\n\n'.join(filtering_prompt)    
        
        filter_prompt = filtering_prompt + filter_template.format(entity,'')
        ipdb.set_trace()
        filter_sample = openai.Completion.create(engine=engine,
                                          prompt=filter_prompt,
                                          max_tokens=1,
                                          temperature=0.0,
                                          logprobs=2,
                                          logit_bias = logit_biases,
                                          stop=["\n", "<|endoftext|>"])
        filter_dict = dict(filter_sample['choices'][0]['logprobs']['top_logprobs'][0])
        
        entity_logits = []
        for opt in ['No','Yes']:
            entity_logits.append(filter_dict[opt])
        
        entity_probs[entity] = scipy.special.softmax(entity_logits)

    return entity_probs, filter_prompt, filter_sample

def run_gpt3_on_df_post_filtering(engine, 
                          test_ready_prompt_dataframe, 
                          prompts,
                          filter_prompt,
                          filter_template,
                          max_tokens=10, 
                          sep=None, 
                          logit_bias=0.1, 
                          sep_logit_bias=0.1, 
                          new_line_logit_bias=0.1,
                                  positive_entity_threshold = 0.4
                         ):
    
    filtering = True
    gpt3_output = []
    predictions = []
    unfiltered_predictions = []

    for i, prompt in tqdm(enumerate(prompts)):    
        test_sent = test_ready_prompt_dataframe.sents.values[i]
        sample = run_gpt3_ner(engine,
                              prompt, 
                              test_sent, 
                              30, 
                              sep, 
                              logit_bias=logit_bias, 
                              sep_logit_bias=sep_logit_bias, 
                              new_line_logit_bias=new_line_logit_bias)

        prediction = sample['choices'][0]['text']
        #Lowercasing all predictions
        prediction = prediction.lower().strip()
        entities = prediction.split(sep.strip())

        entity_probs = run_gpt3_ner_post_filtering(engine, filter_prompt, filter_template, entities)

        filtered_entities = []
        for entity in entities:
            if entity_probs[entity][1] > positive_entity_threshold:
                filtered_entities.append(entity)

        unfiltered_predictions.append(entities)
        predictions.append(filtered_entities)
        gpt3_output.append((sample, entity_probs))

    df = test_ready_prompt_dataframe
    df["predictions"] = predictions
    df['unfiltered_{}'.format("predictions")] = unfiltered_predictions
    df['gpt3_output_{}'.format("predictions")] = gpt3_output
    
    return df

def run_gpt3_on_df(engine, 
                   test_ready_prompt_dataframe, 
                   prompts,
                   max_tokens=10, 
                   sep=None, 
                   logit_bias=0.1, 
                   sep_logit_bias=0.1, 
                   new_line_logit_bias=0.1
                  ):
    
    gpt3_output = []
    predictions = []
    
    for i, prompt in tqdm(enumerate(prompts)):


        test_sent = test_ready_prompt_dataframe.sents.values[i]
        sample = run_gpt3_ner(engine, 
                              prompt, 
                              test_sent, 
                              30, 
                              sep, 
                              logit_bias=logit_bias, 
                              sep_logit_bias=sep_logit_bias, 
                              new_line_logit_bias=new_line_logit_bias)

        gpt3_output.append(sample)
        prediction = sample['choices'][0]['text']
        #Lowercasing all predictions
        prediction = prediction.lower().strip()
        predictions.append(prediction)

    df = test_ready_prompt_dataframe
    df["predictions"] = [p.split(sep.strip()) for p in predictions]
    df['gpt3_output_{}'.format("predictions")] = gpt3_output
    
    return df