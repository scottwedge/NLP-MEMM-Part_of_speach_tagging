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
    def get_word_tag_pair_count_101(self, file_path):
        """
            Extract out of text all suffixes with length 3/tag pairs
            :param file_path: full path of the file to read
                return all suffixes with length 3/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if len(cur_word) > 3:
                        three_letter_suffix = cur_word[-3:]
                        if (three_letter_suffix, cur_tag) not in self.array_count_dicts[1]:
                            self.array_count_dicts[1][(three_letter_suffix, cur_tag)] = 1
                        else:
                            self.array_count_dicts[1][(three_letter_suffix, cur_tag)] += 1

    def get_word_tag_pair_count_102(self, file_path): ## currently checks only if word begins with "pre"
        """
            Extract out of text all prefixes with length 3/tag pairs
            :param file_path: full path of the file to read
                return all prefixes with length 3/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if len(cur_word) > 3:
                        three_letter_prefix = cur_word[0:3]
                        if (three_letter_prefix, cur_tag) not in self.array_count_dicts[2]:
                            self.array_count_dicts[2][(three_letter_prefix, cur_tag)] = 1
                        else:
                            self.array_count_dicts[2][(three_letter_prefix, cur_tag)] += 1

    def get_tag_threesome_count_103(self, file_path):
        """
            Extract out of threesomes of consecutive tags
            :param file_path: full path of the file to read
                return all threesomes of consecutive tags with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(2, len(splited_words)): ## pay attention: starting from idx 2 due to the need of having two previous tags
                    ppword, pptag = splited_words[word_idx - 2].split('_')
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    cword, ctag = splited_words[word_idx].split('_')
                    three_consecutive_tags = (pptag, ptag, ctag)
                    if three_consecutive_tags not in self.array_count_dicts[3]:
                        self.array_count_dicts[3][three_consecutive_tags] = 1
                    else:
                        self.array_count_dicts[3][three_consecutive_tags] += 1


    def get_tag_couples_count_104(self, file_path):
        """
            Extract out of couples of consecutive tags
            :param file_path: full path of the file to read
                return all couples of consecutive tags with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(1, len(splited_words)): ## pay attention: starting from idx 1 due to the need of having one previous tag
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    cword, ctag = splited_words[word_idx].split('_')
                    couple_consecutive_tags = (ptag, ctag)
                    if couple_consecutive_tags not in self.array_count_dicts[4]:
                        self.array_count_dicts[4][couple_consecutive_tags] = 1
                    else:
                        self.array_count_dicts[4][couple_consecutive_tags] += 1

    def get_tag_count_105(self, file_path):
        """
            Extract out of text all tags
            :param file_path: full path of the file to read
                return all tags with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(0, len(splited_words)):
                    cword, ctag = splited_words[word_idx].split('_')
                    if ctag not in self.array_count_dicts[5]:
                        self.array_count_dicts[5][ctag] = 1
                    else:
                        self.array_count_dicts[5][ctag] += 1


    def get_prev_word_curr_tag_pair_count_106(self, file_path):
        """
            Extract out of text all prev word/tag pairs
            :param file_path: full path of the file to read
                return all prev word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(1, len(splited_words)):
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    cword, ctag = splited_words[word_idx].split('_')
                    if (pword, ctag) not in self.array_count_dicts[6]:
                        self.array_count_dicts[6][(pword, ctag)] = 1
                    else:
                        self.array_count_dicts[6][(pword, ctag)] += 1

    def get_next_word_curr_tag_pair_count_107(self, file_path):
        """
            Extract out of text all next word/tag pairs
            :param file_path: full path of the file to read
                return all next word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(0, len(splited_words)-1):
                    cword, ctag = splited_words[word_idx].split('_')
                    nword, ntag = splited_words[word_idx + 1].split('_')
                    if (nword, ctag) not in self.array_count_dicts[7]:
                        self.array_count_dicts[7][(nword, ctag)] = 1
                    else:
                        self.array_count_dicts[7][(nword, ctag)] += 1

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


    def get_id_for_features_over_threshold(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        num_features = 8
        for i in range(0, num_features):
            with open(file_path) as f:

                for line in f:
                    splited_words = line.split(' ')
                    del splited_words[-1]

                    for word_idx in range(len(splited_words)):
                        cur_word, cur_tag = splited_words[word_idx].split('_')

                        if i == 0:
                            if ((cur_word, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                             and (self.feature_statistics.array_count_dicts[i][(cur_word, cur_tag)] >= self.threshold):
                                self.array_of_words_tags_dicts[i][(cur_word, cur_tag)] = self.featureIDX
                                self.featureIDX += 1
                                self.n_tag_pairs += 1

                        elif i == 1:
                            if len(cur_word) > 3:
                                three_letter_suffix = cur_word[-3:]

                                if ((three_letter_suffix, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                                 and (self.feature_statistics.array_count_dicts[i][(three_letter_suffix, cur_tag)] >= self.threshold):
                                    self.array_of_words_tags_dicts[i][(three_letter_suffix, cur_tag)] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 2:
                            if len(cur_word) > 3:
                                three_letter_prefix = cur_word[0:3]
                                if ((three_letter_prefix, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][
                                                 (three_letter_prefix, cur_tag)] >= self.threshold):
                                    self.array_of_words_tags_dicts[i][(three_letter_prefix, cur_tag)] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 3:
                            if word_idx >= 2:
                                ppword, pptag = splited_words[word_idx - 2].split('_')
                                pword, ptag = splited_words[word_idx - 1].split('_')
                                cword, ctag = splited_words[word_idx].split('_')
                                three_consecutive_tags = (pptag, ptag, ctag)

                                if (three_consecutive_tags not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][
                                                 three_consecutive_tags] >= self.threshold):
                                    self.array_of_words_tags_dicts[i][three_consecutive_tags] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 4:
                            if word_idx >= 1:
                                pword, ptag = splited_words[word_idx - 1].split('_')
                                cword, ctag = splited_words[word_idx].split('_')
                                couple_consecutive_tags = (ptag, ctag)

                                if (couple_consecutive_tags not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][
                                                 couple_consecutive_tags] >= self.threshold):
                                    self.array_of_words_tags_dicts[i][couple_consecutive_tags] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 5:
                                cword, ctag = splited_words[word_idx].split('_')

                                if (ctag not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][ctag] >= self.threshold):
                                    self.array_of_words_tags_dicts[i][ctag] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 6:
                            if word_idx > 0:
                                prev_word, prev_tag = splited_words[word_idx - 1].split('_')
                                if ((prev_word, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                                 and (self.feature_statistics.array_count_dicts[i][(prev_word, cur_tag)] >= self.threshold):
                                    self.array_of_words_tags_dicts[i][(prev_word, cur_tag)] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 7:
                            if word_idx < len(splited_words) - 1:
                                next_word, next_tag = splited_words[word_idx + 1].split('_')
                                if ((next_word, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                                 and (self.feature_statistics.array_count_dicts[i][(next_word, cur_tag)] >= self.threshold):
                                    self.array_of_words_tags_dicts[i][(next_word, cur_tag)] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1
                        else:
                            pass

        self.n_total_features += self.n_tag_pairs

    # --- ADD YOURE CODE BELOW --- #




"""### Representing input data with features 
After deciding which features to use, we can represent input tokens as sparse feature vectors. This way, a token is represented with a vec with a dimension D, where D is the total amount of features. \
This is done at training step.

### History tuple
We define a tuple which hold all relevant knowledge about the current word, i.e. all that is relevant to extract features for this token.
"""


def represent_input_with_features(history, Feature2idClass, tag = None):
    """
        Extract feature vector in per a given history
        :param history: touple{ppword, pptag, pword, ptag, cword, ctag, nword, ntag}
        :param word_tags_dict: word\tag dict
            Return a list with all features that are relevant to the given history
    """
    ppword = history[0]
    pptag = history[1]
    pword = history[2]
    ptag = history[3]
    cword = history[4]
    ctag = history[5]
    nword = history[6]
    ntag = history[7]
    if tag:
        ctag = tag
    features = []
    words_tags_dict_100 = Feature2idClass.array_of_words_tags_dicts[0]
    words_tags_dict_101 = Feature2idClass.array_of_words_tags_dicts[1]
    words_tags_dict_102 = Feature2idClass.array_of_words_tags_dicts[2]
    threesome_tags_dict_103 = Feature2idClass.array_of_words_tags_dicts[3]
    couple_tags_dict_104 = Feature2idClass.array_of_words_tags_dicts[4]
    tags_dict_105 = Feature2idClass.array_of_words_tags_dicts[5]
    words_tags_dict_106 = Feature2idClass.array_of_words_tags_dicts[6]
    words_tags_dict_107 = Feature2idClass.array_of_words_tags_dicts[7]

    ## 100 ##
    if (cword, ctag) in words_tags_dict_100:
        features.append(words_tags_dict_100[(cword, ctag)])

    ## 101 ##
    three_letter_suffix = cword[-3:]
    if (three_letter_suffix, ctag) in words_tags_dict_101:
        features.append(words_tags_dict_101[(three_letter_suffix, ctag)])

    ## 102 ##
    three_letter_prefix = cword[0:3]
    if (three_letter_prefix, ctag) in words_tags_dict_102:
        features.append(words_tags_dict_102[(three_letter_prefix, ctag)])

    ## 103 ##
    three_consecutive_tags = (pptag, ptag, ctag)
    if three_consecutive_tags in threesome_tags_dict_103:
        features.append(threesome_tags_dict_103[three_consecutive_tags])

    ## 104 ##
    couple_consecutive_tags = (ptag, ctag)
    if couple_consecutive_tags in couple_tags_dict_104:
        features.append(couple_tags_dict_104[couple_consecutive_tags])

    ## 105 ##
    if ctag in tags_dict_105:
        features.append(tags_dict_105[ctag])

    ## 106 ##
        if (pword, ctag) in words_tags_dict_106:
            features.append(words_tags_dict_106[(pword, ctag)])

    ## 107 ##
        if (nword, ctag) in words_tags_dict_107:
            features.append(words_tags_dict_107[(nword, ctag)])

    return features


def main():
    file_path = os.path.join("data", "train2.wtag")
    my_feature_statistics_class = FeatureStatisticsClass()
    my_feature_statistics_class.get_word_tag_pair_count_100(file_path)
    my_feature_statistics_class.get_word_tag_pair_count_101(file_path)
    my_feature_statistics_class.get_word_tag_pair_count_102(file_path)
    my_feature_statistics_class.get_tag_threesome_count_103(file_path)
    my_feature_statistics_class.get_tag_couples_count_104(file_path)
    my_feature_statistics_class.get_tag_count_105(file_path)
    my_feature_statistics_class.get_prev_word_curr_tag_pair_count_106(file_path)
    my_feature_statistics_class.get_next_word_curr_tag_pair_count_107(file_path)

    num_occurrences_threshold = 1
    my_feature2id_class = Feature2idClass(my_feature_statistics_class, num_occurrences_threshold)
    my_feature2id_class.get_id_for_features_over_threshold(file_path)

    tag = "VBD"

    with open(file_path) as f:
        lines = f.readlines()
        line = lines[54]
        line_split = line.split(' ')
        curr_history = []
        for idx in range(0, 4):
            cur_word, cur_tag = line_split[idx].split('_')
            curr_history.append(cur_word)
            curr_history.append(cur_tag)
        #param history: touple{ppword, pptag, pword, ptag, cword, ctag, nword, ntag}
        features = represent_input_with_features(curr_history, my_feature2id_class)
    pass


if __name__ == '__main__':
    main()
