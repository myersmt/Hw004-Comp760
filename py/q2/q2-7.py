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
test_path = 'Assignment/languageID/'
train_path = test_path

langs = ['e','s','j']
num_test = 0
num_dict = {}

for lang in langs:
    num_dict[lang] = 0

# loop through files
for filename in os.listdir(test_path):
    num_test += 1
    for lang in langs:
        if filename[0] == lang:
            num_dict[lang] += 1

# prior probabilities w/ 1/2
prob_dict = {}
for lang in langs:
    prob_dict[lang] = (num_dict[lang] + 0.5) / (num_test + 1.5)

# define vocabulary
vocab = 'abcdefghijklmnopqrstuvwxyz '

# define smoothing parameter
alpha = 1/2

# initialize count matrix for each language
count_matrix_dict = {}
for lang in langs:
    count_matrix_dict[lang] = np.zeros((27,))

# loop over training files and count character occurrences
for lang in langs:
    for file in os.listdir(train_path):
        if file.startswith(lang) and get_num(file) < 10:
            with open(os.path.join(train_path, file), 'r') as f:
                doc = f.read()
                for char in doc:
                    if char in vocab:
                        count_matrix_dict[lang][vocab.index(char)] += 1

# compute class conditional probability for each language
theta_dict = {}
for lang in langs:
    theta_dict[lang] = (count_matrix_dict[lang] + alpha) / (np.sum(count_matrix_dict[lang]) + len(vocab) * alpha)

# confusion matrix
confusion_matrix = np.zeros((3, 3))

# loop over test files and compute predicted language
for file in os.listdir(test_path):
    if file.endswith('.txt') and get_num(file) >= 10:
        with open(os.path.join(test_path, file), 'r') as f:
            doc = f.read()
            count_vector = np.zeros((27,))
            for char in doc:
                if char in vocab:
                    count_vector[vocab.index(char)] += 1
            # compute log likelihoods for each language
            log_likelihoods = np.zeros((3,))
            for i, lang in enumerate(langs):
                log_likelihoods[i] = np.sum(count_vector * np.log(theta_dict[lang]))

            # compute log posterior probabilities for each language
            log_posteriors = np.zeros((3,))
            for i, lang in enumerate(langs):
                log_posteriors[i] = np.log(prob_dict[lang]) + log_likelihoods[i]

            # find predicted language
            predicted_lang = langs[np.argmax(log_posteriors)]

            # update confusion matrix
            true_lang = file[0]
            confusion_matrix[langs.index(true_lang), langs.index(predicted_lang)] += 1


# print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix)
