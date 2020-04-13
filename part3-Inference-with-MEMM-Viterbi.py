"""## Part 3 - Inference with MEMM-Viterbi"""
import numpy as np


def feature_list_to_probs(sentences_features_list, v):
    sentences_q_list = []
    for feature_list in sentences_features_list:
        q = []
        for tri_mat in feature_list:
            num_ppt = len(tri_mat)
            num_pt = len(tri_mat[0])
            num_ct = len(tri_mat[0][0])
            mat = np.empty((num_ppt, num_pt, num_ct))
            for ppt in range(num_ppt):
                for pt in range(num_pt):
                    for ct in range(num_ct):
                        mat[ppt][pt][ct] = v[tri_mat[ppt][pt][ct]].sum()
            mat = np.exp(mat)
            sum_exp = np.sum(mat, axis=2).reshape((num_ppt, num_pt, 1))
            mat /= sum_exp
            q.append(mat)
        sentences_q_list.append(q)
    return sentences_q_list


def memm_viterbi(sentences_q_list):
    tags_infer = []
    for q in sentences_q_list:
        num_h = len(q)
        pi = [q[0].reshape(q.shape[1:])]
        bp = np.empty(num_h+1)
        for k in range(1, num_h):
            pi_prev = pi[k-1].reshape(pi[k-1].shape[0], -1, 1)
            curr_bp = np.argmax(pi_prev * q[k])
            bp[k-1] = curr_bp
            i, j = np.ogrid[: q.shape[1], : q.shape[2]]
            pi.append(q[k][curr_bp, i, j])
        tags_infer.append(bp[1:])
    return tags_infer


