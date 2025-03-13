import numpy as np

def build_context_dictionary(sequences, max_depth):
    context_to_index = {}
    idx = 0

    for seq in sequences:
        L = len(seq)
        for pos in range(L):
            for depth in range(1, max_depth + 1):
                if pos + depth > L:
                    break
                sub = seq[pos: pos + depth]
                if sub not in context_to_index:
                    context_to_index[sub] = idx
                    idx += 1
    return context_to_index

def apply_idf_weighting(F):
    n, d = F.shape
    idf_vec = np.empty(d, dtype=np.float64)

    for j in range(d):
        df = np.count_nonzero(F[:, j] > 0)
        idf = np.log(n / (1.0 + df))
        idf_vec[j] = idf

    F *= idf_vec
    return F

def compute_context_tree_feature_matrix_weighted(sequences, max_depth, gamma=0.8, context_to_index=None):
    n = len(sequences)
    if context_to_index is None:
        context_to_index = build_context_dictionary(sequences, max_depth)

    num_contexts = len(context_to_index)
    F = np.zeros((n, num_contexts), dtype=np.float64)

    for i, seq in enumerate(sequences):
        L = len(seq)
        for pos in range(L):
            for depth in range(1, max_depth + 1):
                if pos + depth > L:
                    break
                sub = seq[pos: pos + depth]
                idx = context_to_index[sub]
                F[i, idx] += gamma ** depth

    F = apply_idf_weighting(F)
    return F, context_to_index

def context_tree_kernel_matrix(features):
    return features.dot(features.T)

def compute_context_tree_feature_matrix_with_dict_weighted(sequences, max_depth, context_to_index, gamma=0.8):
    n = len(sequences)
    num_contexts = len(context_to_index)
    F = np.zeros((n, num_contexts), dtype=np.float64)

    for i, seq in enumerate(sequences):
        L = len(seq)
        for pos in range(L):
            for depth in range(1, max_depth + 1):
                if pos + depth > L:
                    break
                sub = seq[pos: pos + depth]
                if sub in context_to_index:
                    idx = context_to_index[sub]
                    F[i, idx] += gamma ** depth

    F = apply_idf_weighting(F)
    return F

def predict_context(alpha, K):
    f_vals = K.dot(alpha)
    preds = np.where(f_vals >= 0.0, 1, -1)
    return f_vals, preds
