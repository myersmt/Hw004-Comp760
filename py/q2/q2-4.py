"""
Created by: Matt Myers
Question 002-4
"""
import os
import numpy as np

# Path
train_path = 'Assignment/languageID/'

vocab = 'abcdefghijklmnopqrstuvwxyz '

# define smoothing parameter
alpha = 1/2

# initialize count vector
count_vector = np.zeros((27,))

# read test document
with open(os.path.join(train_path, 'e10.txt'), 'r') as f:
    doc = f.read()
    for char in doc:
        if char in vocab:
            count_vector[vocab.index(char)] += 1

for ind, let in enumerate(vocab):
    if ind == 0:
        print(r'\begin{center}'+'\n'+r'\begin{tabular}{|c|c|}'+r'\hline'+f'\nLetter & Percentage \\\\')
    print(r'\hline')
    print(let,f' & {count_vector[ind]:.4}', r'\\')
print(r'\hline'+'\n'+r'\end{tabular}'+'\n'+r'\end{center}')

# print bag-of-words
print(count_vector, len(count_vector))

