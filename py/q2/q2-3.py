"""
Created by: Matt Myers
Question 002-3
"""
import os
import numpy as np

def get_num(string):
    return(int(string[1:].split('.')[0]))

# Path
train_path = 'Assignment/languageID/'

vocab = 'abcdefghijklmnopqrstuvwxyz '

# define smoothing parameter
alpha = 1/2

theta = {}

for lang in ['s', 'j']:
    count_matrix = np.zeros((27,))

    for file in os.listdir(train_path):
        if file.startswith(lang) and get_num(file) < 10:
            with open(os.path.join(train_path, file), 'r') as f:
                doc = f.read()
                for char in doc:
                    if char in vocab:
                        count_matrix[vocab.index(char)] += 1

    theta[lang] = (count_matrix + alpha) / (np.sum(count_matrix) + len(vocab) * alpha)

for lang in ['s', 'j']:
    for ind, let in enumerate(vocab):
        if ind == 0:
            print(r'\begin{tabular}{|c|c|}'+r'\hline'+f'\nLetter (\\theta_{lang}) & Percentage \\\\')
        print(r'\hline')
        print(let,f' & {theta[lang][ind]:.4}', r'\\')
    print(r'\hline'+'\n'+r'\end{tabular}')

[float(i) for i in theta[lang]]

for lang in ['s', 'j']:
    print(f'\\theta_{lang} = {theta[lang]}')
