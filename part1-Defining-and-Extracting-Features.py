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
        self.array_count_dicts = []
        for i in range(0, 8):
            self.array_count_dicts.append(OrderedDict())


        # self.words_tags_count_dict_100 = OrderedDict()
        # # ---Add more count dictionaries here---
        # self.words_tags_count_dict_101 = OrderedDict()
        # self.words_tags_count_dict_102 = OrderedDict()

    def get_word_tag_pair_count_100(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if (cur_word, cur_tag) not in self.array_count_dicts[0]:
                        self.array_count_dicts[0][(cur_word, cur_tag)] = 1
                    else:
                        self.array_count_dicts[0][(cur_word, cur_tag)] += 1

    # --- ADD YOURE CODE BELOW --- #
    def get_word_tag_pair_count_101(self, file_path): ## currently checks only if word ends with "ing"
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if len(cur_word) > 3:
                        if cur_word[-3:] == "ing":
                            if (cur_word, cur_tag) not in self.array_count_dicts[1]:
                                self.array_count_dicts[1][(cur_word, cur_tag)] = 1
                            else:
                                self.array_count_dicts[1][(cur_word, cur_tag)] += 1

    def get_word_tag_pair_count_102(self, file_path): ## currently checks only if word begins with "pre"
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if len(cur_word) > 3:
                        if cur_word[0:3] == "pre":
                            if (cur_word, cur_tag) not in self.array_count_dicts[2]:
                                self.array_count_dicts[2][(cur_word, cur_tag)] = 1
                            else:
                                self.array_count_dicts[2][(cur_word, cur_tag)] += 1

"""### Indexing features 
After getting feature statistics, each feature is given an index to represent it. We include only features that appear more times in text than the lower bound - 'threshold'
"""


class Feature2idClass:

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word\Tag pairs features
        self.featureIDX = 0   # index for each feature
        # Init all features dictionaries
        self.array_of_words_tags_dicts = []
        for i in range(0, 8):
            self.array_of_words_tags_dicts.append(OrderedDict())


    def get_word_tag_pairs_100_to_102(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for i in range(0, 3):
                for line in f:
                    splited_words = line.split(' ')
                    del splited_words[-1]

                    for word_idx in range(len(splited_words)):
                        cur_word, cur_tag = splited_words[word_idx].split('_')

                        if ((cur_word, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                            and (self.feature_statistics.array_count_dicts[i][(cur_word, cur_tag)] >= self.threshold):
                                    #and (self.feature_statistics.words_tags_count_dict_100[(cur_word, cur_tag)] >= self.threshold):
                            self.array_of_words_tags_dicts[i][(cur_word, cur_tag)] = self.featureIDX
                            self.featureIDX += 1
                            self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs

    # --- ADD YOURE CODE BELOW --- #

    # def get_word_tag_pairs_101(self, file_path):
    #     """
    #         Extract out of text all word/tag pairs
    #         :param file_path: full path of the file to read
    #             return all word/tag pairs with index of appearance
    #     """
    #     with open(file_path) as f:
    #         for line in f:
    #             splited_words = line.split(' ')
    #             del splited_words[-1]
    #
    #             for word_idx in range(len(splited_words)):
    #                 cur_word, cur_tag = splited_words[word_idx].split('_')
    #                 if ((cur_word, cur_tag) not in self.words_tags_dict) \
    #                         and (self.feature_statistics.words_tags_count_dict_101[(cur_word, cur_tag)] >= self.threshold):
    #                     self.words_tags_dict[(cur_word, cur_tag)] = self.featureIDX
    #                     self.featureIDX += 1
    #                     self.n_tag_pairs += 1
    #     self.n_total_features += self.n_tag_pairs
    #
    #
    #
    # def get_word_tag_pairs_102(self, file_path):
    #     """
    #         Extract out of text all word/tag pairs
    #         :param file_path: full path of the file to read
    #             return all word/tag pairs with index of appearance
    #     """
    #     with open(file_path) as f:
    #         for line in f:
    #             splited_words = line.split(' ')
    #             del splited_words[-1]
    #
    #             for word_idx in range(len(splited_words)):
    #                 cur_word, cur_tag = splited_words[word_idx].split('_')
    #                 if ((cur_word, cur_tag) not in self.words_tags_dict) \
    #                         and (self.feature_statistics.words_tags_count_dict_101[(cur_word, cur_tag)] >= self.threshold):
    #                     self.words_tags_dict[(cur_word, cur_tag)] = self.featureIDX
    #                     self.featureIDX += 1
    #                     self.n_tag_pairs += 1
    #     self.n_total_features += self.n_tag_pairs


"""### Representing input data with features 
After deciding which features to use, we can represent input tokens as sparse feature vectors. This way, a token is represented with a vec with a dimension D, where D is the total amount of features. \
This is done at training step.

### History tuple
We define a tuple which hold all relevant knowledge about the current word, i.e. all that is relevant to extract features for this token.
"""


def represent_input_with_features(history, Feature2idClass, tag = None):
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
    if tag:
        ctag = tag
    features = []
    words_tags_dict_100 = Feature2idClass.words_tags_dict_100 = Feature2idClass.array_of_words_tags_dicts[0]
    words_tags_dict_101 = Feature2idClass.words_tags_dict_101 = Feature2idClass.array_of_words_tags_dicts[1]
    words_tags_dict_102 = Feature2idClass.words_tags_dict_101 = Feature2idClass.array_of_words_tags_dicts[2]


    if (cword, ctag) in words_tags_dict_100: #words_tags_dict_100
        features.append(words_tags_dict_100[(cword, ctag)])

    # --- CHECK APEARANCE OF MORE FEATURES BELOW --- #
    if (cword, ctag) in words_tags_dict_101:
        features.append(words_tags_dict_101[(cword, ctag)])

    if (cword, ctag) in words_tags_dict_102:
        features.append(words_tags_dict_102[(cword, ctag)])
    return features


def main():
    file_path = os.path.join("data", "train2.wtag")
    my_feature_statistics_class = FeatureStatisticsClass()
    my_feature_statistics_class.get_word_tag_pair_count_100(file_path)
    my_feature_statistics_class.get_word_tag_pair_count_101(file_path)
    my_feature_statistics_class.get_word_tag_pair_count_102(file_path)

    num_occurrences_threshold = 1
    my_feature2id_class = Feature2idClass(my_feature_statistics_class, num_occurrences_threshold)
    my_feature2id_class.get_word_tag_pairs_100_to_102(file_path)

    tag = "VBD"

    with open(file_path) as f:
        lines = f.readlines()
        line = lines[54]
        line_split = line.split(' ')
        curr_history = []
        for idx in range(0, 3):
            cur_word, cur_tag = line_split[idx].split('_')
            curr_history.append(cur_word)
            curr_history.append(cur_tag)
        #param history: touple{ppword, pptag, pword, ptag, cword, ctag}
        features = represent_input_with_features(curr_history, my_feature2id_class)
    pass


if __name__ == '__main__':
    main()
