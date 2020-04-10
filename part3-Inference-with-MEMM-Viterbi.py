"""## Part 3 - Inference with MEMM-Viterbi
Recall - the MEMM-Viterbi takes the form of:

"""


def memm_viterbi():
    """
    Write your MEMM Vitebi imlementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """

    return tags_infer


from IPython.display import HTML
from base64 import b64encode
! git
clone
https: // github.com / eyalbd2 / 0
97215
_Natural - Language - Processing_Workshop - Notebooks.git



"""Notation:
  - *w* refers to the trained weights vector
  - *u* refers to the previous tag
  - *v* refers to the current tag
  - *t* refers to the tag previous to *u*

The video above presents a vanilla memm viterbi. \
There are several methods to improve the performence of the algorithm, we will specify two of them: 


1.   Dividing the algorithm to multiple processes 
2.   Implementing beam search viterbi, and reducing Beam size 


Notice that the latter might affect the results, hence beam size is required to be chosen wisely.

## Accuracy and Confusion Matrix
![Accuracy and Confusion Matrix](https://raw.githubusercontent.com/eyalbd2/097215_Natural-Language-Processing_Workshop-Notebooks/master/conf_mat_slide.PNG)

## Interface for creating competition files
In your submission, you must implement a python file named `generate_comp_tagged.py` which generates your tagged competition files for both datasets in a single call.
It should do the following for each dataset:

1. Load trained weights to your model
2. Load competition data file
3. Run inference on competition data file
4. Write results to file according to .wtag format (described in HW1)
5. Validate your results file comply with .wtag format (according to instructions described in HW1)

"""