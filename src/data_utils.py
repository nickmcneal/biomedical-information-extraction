import pandas as pd
import numpy as np
import ipdb
import pickle
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def create_prompt_dataset(train_df, test_df, seed, few_shot_num, number_of_test_samples, selection_strategy,sep=', '):
    
    random = np.random.RandomState(seed)
    
    few_shot_prompt, chosen_prompt_ids = create_few_shot_prompt(train_df, random, few_shot_num, selection_strategy)
    
    test_df['test_ready_prompt'] = [few_shot_prompt+'\n\n'+empty_prompt for empty_prompt in test_df['empty_prompts']]
    
    if number_of_test_samples != 'all':
        #Making sure the samples are the same as the first batch
        random = np.random.RandomState(42)
        random.permutation(train_df.index)
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