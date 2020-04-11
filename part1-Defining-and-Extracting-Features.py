import os
from collections import OrderedDict

"""
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


"""## Part 1 - Defining and Extracting Features
In class we saw the importance of extracting good features for NLP task. A good feature is such that (1) appear many times in the data and (2) gives information that is relevant for the label.

### Counting feature appearances in data
We would like to include features that appear many times in the data. Hence we first count the number of appearances for each feature. \
This is done at pre-training step.
"""


class FeatureStatisticsClass:

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
                # splited_words = split(line, (' ', '\n'))
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    # cur_word, cur_tag = split(splited_words[word_idx], '_')
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    # if (cur_word, cur_tag) not in self.words_tags_dict:
                    if (cur_word, cur_tag) not in self.words_tags_count_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1

    # --- ADD YOURE CODE BELOW --- #


"""### Indexing features 
After getting feature statistics, each feature is given an index to represent it. We include only features that appear more times in text than the lower bound - 'threshold'
"""


class Feature2idClass:

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word\Tag pairs features

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()

    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                # splited_words = split(line, (' ', '\n'))
                splited_words = line.split(' ')
                del splited_words[-1]

                for word_idx in range(len(splited_words)):
                    # cur_word, cur_tag = split(splited_words[word_idx], '_')
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if ((cur_word, cur_tag) not in self.words_tags_dict) \
                            and (self.feature_statistics.words_tags_count_dict[(cur_word, cur_tag)] >= self.threshold):
                        # and (self.statistics.words_tags_dict[(cur_word, cur_tag)] >= self.threshold):
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


def represent_input_with_features(history, word_tags_dict, tag = None):
    """
        Extract feature vector in per a given history
        :param history: touple{ppword, pptag, pword, ptag, cword, ctag}
        :param word_tags_dict: word\tag dict
            Return a list with all features that are relevant to the given history
    """
    ppword = history[0]
    pptag = history[1]
    pword = history[2]
    ptag = history[3]
    cword = history[4]
    ctag = history[5]
    features = []

    if (cword, ctag) in word_tags_dict:
        features.append(word_tags_dict[(cword, ctag)])

    # --- CHECK APEARANCE OF MORE FEATURES BELOW --- #

    return features


def main():
    file_path = os.path.join("data", "train2.wtag")
    my_feature_statistics_class = FeatureStatisticsClass()
    my_feature_statistics_class.get_word_tag_pair_count(file_path)

    num_occurrences_threshold = 3
    my_feature2id_class = Feature2idClass(my_feature_statistics_class, num_occurrences_threshold)
    my_feature2id_class.get_word_tag_pairs(file_path)

    with open(file_path) as f:
        lines = f.readlines()
        line = lines[8]
        line_split = line.split(' ')
        curr_history = []
        for idx in range(0, 3):
            cur_word, cur_tag = line_split[idx].split('_')
            curr_history.append(cur_word)
            curr_history.append(cur_tag)
        #param history: touple{ppword, pptag, pword, ptag, cword, ctag}
        features = represent_input_with_features(curr_history, my_feature2id_class.words_tags_dict)
    pass


if __name__ == '__main__':
    main()
