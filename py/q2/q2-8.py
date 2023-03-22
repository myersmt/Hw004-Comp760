"""
Created by: Matt Myers
Question 002-6
"""
import os
import numpy as np
import random

def get_num(string):
    return(int(string[1:].split('.')[0]))

# Path
train_path = 'Assignment/languageID/'

vocab = 'abcdefghijklmnopqrstuvwxyz '
langs = ['e', 's', 'j']

# define smoothing parameter
alpha = 1/2

count_vector = np.zeros((27,))
theta = {}

num_train = 0
num_dict = {}

for lang in langs:
    num_dict[lang] = 0

# files
for filename in os.listdir(train_path):
    num_train += 1
    for lang in langs:
        if filename[0] == lang:
            num_dict[lang] += 1

for lang in langs:
    count_matrix = np.zeros((27,))

    for file in os.listdir(train_path):
        if file.startswith(lang) and get_num(file) < 10:
            with open(os.path.join(train_path, file), 'r') as f:
                doc = f.read()
                for char in doc:
                    if char in vocab:
                        count_matrix[vocab.index(char)] += 1

    theta[lang] = (count_matrix + alpha) / (np.sum(count_matrix) + len(vocab) * alpha)

# test
doc_num = random.randint(10,19)  # choose a document number between 10 and 19
doc_lang = langs[random.randint(0,2)] # choose the language of the document
doc_path = os.path.join(train_path, doc_lang + str(doc_num) + '.txt')

with open(doc_path, 'r') as f:
    doc = f.read()
    doc_list = list(doc)
    random.shuffle(doc_list)
    shuffled_doc = ''.join(doc_list)
    print(shuffled_doc)
    for char in shuffled_doc:
        if char in vocab:
            count_vector[vocab.index(char)] += 1

# bag-of-words
x = count_vector

log_prob = {}
log_posterior = {}
prob_dict = {}
for lang in langs:
    prob_dict[lang] = (num_dict[lang] + 0.5) / (num_train + 1.5)
    log_prior = np.log(prob_dict[lang])
    log_likelihood = np.sum(np.log(theta[lang]) * x)
    log_posterior[lang] = log_prior + log_likelihood

# Predict
predicted_lang = max(log_posterior, key=log_posterior.get)

# Print
for ind, lang in enumerate(langs):
    print(r'\hat{p}(y ='+f' {lang} | x) = ','e^{'+f'{log_posterior[lang]}'+'}')
print(f"Predicted language: {predicted_lang}, Actual language: {doc_lang}")
