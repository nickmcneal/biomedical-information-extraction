#COMPUTE CoNLL F1 SCORE

tp = 0
fp = 0
tn = 0
fn = 0

sentence_total=0

print(prompts[0])
#get indices of sentences to get entries from dict
# sentence_indices = sentence_indices[num_prompt_samples:] #ONLY RUN ONCE EACH TIME!
counter = 0
for sentence_idx in test_sentence_indices:
  print(output[prompt_indices[num_prompt_samples + counter]])
  print("Words in sentence:",sentence_maps[sentence_idx])
  print("True labels:",drugs_maps[sentence_idx])
  print("Predicted labels:",predictions_maps[counter],'\n')


  #increment tp and fn
  for key in drugs_maps[sentence_idx]:
    if key in predictions_maps[counter]:
      tp += min(drugs_maps[sentence_idx][key], predictions_maps[counter][key])
    else: #key not in prediction
      fn += drugs_maps[sentence_idx][key]

  #increment fp and fn
  for key in predictions_maps[counter]:
    if key not in drugs_maps[sentence_idx]:
      fp += predictions_maps[counter][key]
    elif key in drugs_maps[sentence_idx] and predictions_maps[counter][key] > drugs_maps[sentence_idx][key]:
      fp += predictions_maps[counter][key] - drugs_maps[sentence_idx][key]
    elif key in drugs_maps[sentence_idx] and predictions_maps[counter][key] < drugs_maps[sentence_idx][key]:
      fn += drugs_maps[sentence_idx][key] - predictions_maps[counter][key]


  #increment tn
  for key in sentence_maps[sentence_idx]:
    sentence_total += sentence_maps[sentence_idx][key]
    if key not in predictions_maps[counter] and key not in drugs_maps[counter]:
      tn += sentence_maps[sentence_idx][key]
    else: #key in prediction
      if sentence_maps[sentence_idx][key] > predictions_maps[counter][key] and key not in drugs_maps[counter]:
        tn += sentence_maps[sentence_idx][key] - predictions_maps[counter][key]

  counter += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)

f1 = 2 * (precision * recall) / (precision + recall)
print(f1)
print("precision: {}  recall: {}  tp: {}  fp: {}  tn: {}  fn: {}".format(precision, recall, tp, fp, tn, fn))
print("sentence total:",sentence_total)
print("pred total:",tp+fp+fn+tn)
