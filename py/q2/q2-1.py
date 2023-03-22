"""
Created by: Matt Myers
Question 002-1

Use files 0.txt to 9.txt in each language as the training data.
Estimate the prior probabilities p(y = e),
p(y = j), p(y = s) using additive smoothing with parameter 1
2. Give the formula for additive smoothing
with parameter 1
2 in this case. Print the prior probabilities.
"""
import os

# Path
train_path = 'Assignment/languageID/'

langs = ['e','s','j']
num_train = 0
num_dict = {}

for lang in langs:
    num_dict[lang] = 0

# loop through files
for filename in os.listdir(train_path):
    num_train += 1
    for lang in langs:
        if filename[0] == lang:
            num_dict[lang] += 1

# prior probabilities w/ 1/2
prob_dict = {}
for lang in langs:
    prob_dict[lang] = (num_dict[lang] + 0.5) / (num_train + 1.5)

# =Print
for lang in langs:
    print(f'p(y={lang}) = {prob_dict[lang]}')
