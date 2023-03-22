"""
Created by: Matt Myers
Question 002-2

Using the same training data, estimate the class conditional probability (multinomial parameter) for English
θi,e := p(ci | y = e)
where ci is the i-th character. That is, c1 = a, . . . , c26 = z, c27 = space. Again, use additive smoothing
with parameter 1
2. Give the formula for additive smoothing with parameter 1
2 in this case. Print θe which is
a vector with 27 elements.
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

count_matrix = np.zeros((27,))

for file in os.listdir(train_path):
    if file.startswith('e') and get_num(file) < 10:
        with open(os.path.join(train_path, file), 'r') as f:
            doc = f.read()
            for char in doc:
                if char in vocab:
                    count_matrix[vocab.index(char)] += 1

theta_e = (count_matrix + alpha) / (np.sum(count_matrix) + len(vocab) * alpha)

for ind, let in enumerate(vocab):
    if ind == 0:
        print(r'\begin{tabular}{|c|c|}'+r'\hline'+f'\nLetter & Percentage \\\\')
    print(r'\hline')
    print(let,f' & {theta_e[ind]:.4}', r'\\')
print(r'\hline')

print(theta_e)
