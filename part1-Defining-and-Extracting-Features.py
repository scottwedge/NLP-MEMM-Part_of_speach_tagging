# -*- coding: utf-8 -*-
"""NLP_1-POS-Tagger.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/eyalbd2/097215_Natural-Language-Processing_Workshop-Notebooks/blob/master/NLP_1_POS_Tagger.ipynb

# <img src="https://img.icons8.com/dusk/64/000000/mind-map.png" style="height:50px;display:inline"> IE 097215 - Technion - Natural Language Processing

## Part 0 - Project Structure
Part Of Speech (POS) tagger is a well known NLP task. As a result, many solutions were proposed to this setup. We present a general solution guidelines to this task (while this is definately not obligatory to use these guidelines to solve HW1). \
A POS tagger can be divided to stages:


*   Pre-training:
    1.   Preprocessing data
    2.   Features engineering
    3.   Define the objective of the model


*   During training:
    1.   Represent data as feature vectors (Token2Vector)
    2.   Optimization - We need to tune the weights of the model inorder to solve the objective

*   Inference:
    1.   Use dynamic programing (Viterbi) to tag new data based on MEMM
"""

# Anaconda Environment Setup for HW1 (on Azure machine or on your laptop)



"""## Part 1 - Defining and Extracting Features
In class we saw the importance of extracting good features for NLP task. A good feature is such that (1) appear many times in the data and (2) gives information that is relevant for the label.

### Counting feature appearances in data
We would like to include features that appear many times in the data. Hence we first count the number of appearances for each feature. \
This is done at pre-training step.
"""

from collections import OrderedDict


class feature_statistics_class():

    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        # ---Add more count dictionaries here---

    def get_word_tag_pair_count(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = split(line, (' ', '\n'))
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = split(splited_words[word_idx], '_')
                    if (cur_word, cur_tag) not in self.words_tags_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1

    # --- ADD YOURE CODE BELOW --- #


"""### Indexing features 
After getting feature statistics, each feature is given an index to represent it. We include only features that appear more times in text than the lower bound - 'threshold'
"""


class feature2id_class():

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each featue gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word\Tag pairs features

        # Init all features dictionaries
        self.words_tags_dict = collections.OrderedDict()

    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = split(line, (' ', '\n'))
                del splited_words[-1]

                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = split(splited_words[word_idx], '_')
                    if ((cur_word, cur_tag) not in self.words_tags_dict) \
                            and (self.statistics.words_tags_dict[(cur_word, cur_tag)] >= self.threshold):
                        self.words_tags_dict[(cur_word, cur_tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs

    # --- ADD YOURE CODE BELOW --- #


"""### Representing input data with features 
After deciding which features to use, we can represent input tokens as sparse feature vectors. This way, a token is represented with a vec with a dimension D, where D is the total amount of features. \
This is done at training step.

### History tuple
We define a tuple which hold all relevant knowledge about the current word, i.e. all that is relevant to extract features for this token.
"""


def represent_input_with_features(history, word_tags_dict):
    """
        Extract feature vector in per a given history
        :param history: touple{word, pptag, ptag, ctag, nword, pword}
        :param word_tags_dict: word\tag dict
            Return a list with all features that are relevant to the given history
    """
    word = history[0]
    pptag = history[1]
    ptag = history[2]
    ctag = history[3]
    nword = history[4]
    pword = history[5]
    features = []

    if (word, ctag) in word_tags_dict:
        features.append(word_tags_dict[(word, ctag)])

    # --- CHECK APEARANCE OF MORE FEATURES BELOW --- #

    return features

