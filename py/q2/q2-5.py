"""
Created by: Matt Myers
Question 002-5
"""
import os
import numpy as np

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
with open(os.path.join(train_path, 'e10.txt'), 'r') as f:
    doc = f.read()
    for char in doc:
        if char in vocab:
            count_vector[vocab.index(char)] += 1

# bag-of-words for e10.txt
x = count_vector

log_prob = {}
for lang in langs:
    log_prob[lang] = np.sum(np.log(theta[lang]) * x)

# Print
for lang in langs:
    print(r'\log(\hat{p}(x | y ='+f' {lang})) = ', log_prob[lang],'\\\\')


