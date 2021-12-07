!pip install openai
import openai

import random
import os
import pandas as pd
import numpy as np
import pickle as pkl

#connect to API with your info
openai.api_key=" "

#in create:
  #presence penalty
  #logit bias
  #https://beta.openai.com/docs/api-reference/completions/create
input_filename = "development_examples.txt" # loads this file
output_filename = "output.txt" # creates this file
output_map_filename = "output_maps.pkl"

# read text file, keep all lines (including those that start with commas)
input = pd.read_csv(input_filename, header=None, sep='\n')
input = input[0].str.split('\s\|\s', expand=True)
sentences = []
tags = []
sentence_maps = []
sentence_maps_space = []

# convert to np array for processing
input = np.asarray(input)

# convert to two separate arrays
for idx, line in enumerate(input):
  if idx % 2 == 0:
    sentence = input[idx][0].lower()
    sentence_space = input[idx][0].lower()
    sentences.append(sentence)

    sentence_map = dict()
    sentence_map_space = dict()
    sentence = sentence.split(' ')
    sentence_space = sentence_space.split(' ')
    for word in sentence_space:
      word = " " + word
    for word in sentence_space:
      if word in sentence_map_space:
        sentence_map_space[word] += 1
      else:
        sentence_map_space[word] = 1
    
    for word in sentence:
      if word in sentence_map:
        sentence_map[word] += 1
      else:
        sentence_map[word] = 1
    sentence_maps.append(sentence_map)
    sentence_maps_space.append(sentence_map_space)
  else:
    tags.append(input[idx][0])

# check if number of lines is equal in both arrays
if np.size(sentences) != np.size(tags):
  raise ValueError('Number of sentence lines not equal to number of tag lines.')

# separate by space, so that we can check that the number of terms is equal in both arrays
sentences_tmp = sentences[:] # copy instead of reference
tags_tmp = tags
for idx, line in enumerate(sentences_tmp):
  sentences_tmp[idx] = sentences_tmp[idx].split()
for idx, tag in enumerate(tags_tmp):
  tags_tmp[idx] = tags_tmp[idx].split()

# check if number of terms is equal in both arrays
if len(sentences_tmp) != len(tags_tmp):
  raise ValueError('Number of terms in "sentences" not equal to number of terms in "tags"')
for i in range(len(sentences_tmp)):
  if len(sentences_tmp[i]) != len(tags_tmp[i]):
    raise ValueError('Unequal number of terms at line (index) ' + str(i))



# create output array and file and save dictionary of correct quantified drug labels for each example
output = []
drugs_maps = []
for line in range(len(sentences_tmp)):
  drugs_current_line = []
  sentence = "Sentence: " + str(sentences[line])
  output.append(sentence)
  for drug_idx, label_list in enumerate(tags[line]):
    if tags[line][drug_idx] == "B":
      temporary_drug_term = sentences_tmp[line][drug_idx]
      drugs_current_line.append(sentences_tmp[line][drug_idx])
    if tags[line][drug_idx] == "I":
      while drugs_current_line:
        popped = drugs_current_line.pop()
      temporary_drug_term = sentences_tmp[line][drug_idx]
      new_word = popped + temporary_drug_term
      drugs_current_line.append(new_word)
      # print(drugs_current_line)
      drugs_current_line.append(sentences_tmp[line][drug_idx])
  drugs = "Drugs: " + ", ".join(str(x).lower() for x in drugs_current_line)
  output.append(drugs)
  print(drugs_current_line)
  #create and save dictionary of drug names
  drugs_map = dict()
  for drug in drugs_current_line:
    if drug in drugs_map:
      drugs_map[drug] += 1
    else:
      drugs_map[drug] = 1
  drugs_maps.append(drugs_map)

with open(output_filename, mode='wt') as output_file:
    output_file.write('\n'.join(output)) # give it a minute to save
