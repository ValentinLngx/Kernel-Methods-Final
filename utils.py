import numpy as np
import pandas as pd
import math
from collections import Counter

def compute_context_embedding(sequences):
    embedding_matrix = []
    for seq in sequences:
        counts = [seq.count(nuc) for nuc in "ACGT"]
        embedding_matrix.append(counts)
    return np.array(embedding_matrix)

def positional_encoding(seq_length, d_model):
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    pos_encoding = np.zeros((seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pos_encoding

def upgraded_context_embedding(sequences, embedding_dim=10, conv_kernel_size=3, num_filters=16):
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    embedding_matrix = np.random.randn(4, embedding_dim)

    conv_kernel = np.random.randn(conv_kernel_size, embedding_dim, num_filters)
    
    embeddings = []
    
    for seq in sequences:
        seq_length = len(seq)
        indices = [nucleotide_to_index[nuc] for nuc in seq]
        seq_embed = embedding_matrix[indices, :]

        pos_enc = positional_encoding(seq_length, embedding_dim)
        seq_embed = seq_embed + pos_enc

        conv_outputs = []
        for i in range(seq_length - conv_kernel_size + 1):
            window = seq_embed[i:i+conv_kernel_size, :]
            conv_window = np.zeros(num_filters)
            for j in range(num_filters):
                kernel = conv_kernel[:, :, j]
                conv_window[j] = np.sum(window * kernel)
            conv_outputs.append(conv_window)
        conv_outputs = np.array(conv_outputs)

        conv_outputs = np.maximum(0, conv_outputs)

        pooled = conv_outputs.mean(axis=0)
        embeddings.append(pooled)
    embeddings = np.array(embeddings)
    return embeddings

##############################################
# Spectrum Kernel Functions
##############################################

def extract_kmers(sequence, k):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return kmers

def build_kmer_dictionary(sequences, k):
    kmer_set = set()
    for sequence in sequences:
        kmers = extract_kmers(sequence, k)
        for kmer in kmers:
            kmer_set.add(kmer)
    kmer_to_index = {kmer: i for i, kmer in enumerate(kmer_set)}
    return kmer_to_index

def seq_to_kmer_sequence(seq, k=3, kmer_to_index=None):
    kmers = extract_kmers(seq, k)
    if kmer_to_index is None:
        raise ValueError("kmer_to_index dictionary must be provided")
    return [kmer_to_index[kmer] for kmer in kmers if kmer in kmer_to_index]

def compute_document_frequency(sequences, k, kmer_to_index):
    df = np.zeros(len(kmer_to_index))
    for sequence in sequences:
        unique_kmers = set(extract_kmers(sequence, k))
        for kmer in unique_kmers:
            if kmer in kmer_to_index:
                df[kmer_to_index[kmer]] += 1
    return df

def compute_idf(df, N):
    return np.log((N + 1) / (df + 1)) + 1

def compute_feature_vector(sequence,k,kmer_to_index, idf=None):
    feature_vector = np.zeros(len(kmer_to_index))
    kmers=extract_kmers(sequence,k)
    for kmer in kmers:
        if kmer in kmer_to_index:
            feature_vector[kmer_to_index[kmer]]+=1
    if idf is not None:
        feature_vector = feature_vector * idf
    return feature_vector

def compute_feature_matrix(sequences,k):
    kmer_to_index = build_kmer_dictionary(sequences, k)
    df = compute_document_frequency(sequences, k, kmer_to_index)
    idf = compute_idf(df, len(sequences))

    feature_list = []
    for seq in sequences:
        fv = compute_feature_vector(seq, k, kmer_to_index,idf)
        feature_list.append(fv)
    features = np.vstack(feature_list)
    return features, kmer_to_index

def spectrum_kernel_matrix(features):
    return features @ features.T

#reverse complement & canonical k-mers
def reverse_complement(seq):
    mapping = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(mapping[base] for base in reversed(seq))

def canonical_kmer(kmer):
    rc = reverse_complement(kmer)
    return min(kmer, rc)

# k-mer counting with TF--IDF weighting
def build_kmer_counts(seqs, k=5, use_canonical=True):
    counters = []
    for s in seqs:
        c = Counter()
        for i in range(len(s) - k + 1):
            kmer = s[i:i+k]
            if use_canonical:
                kmer = canonical_kmer(kmer)
            c[kmer] += 1
        counters.append(c)
    return counters

def apply_tfidf(counters):
    N = len(counters)
    df = Counter()
    for c in counters:
        for key in c:
            df[key] += 1
    idf = {key: math.log((N+1)/(df[key]+1)) + 1 for key in df}
    tfidf_counters = []
    for c in counters:
        new_c = Counter({key: count * idf[key] for key, count in c.items()})
        tfidf_counters.append(new_c)
    return tfidf_counters

def build_improved_kmer_counts(seqs, k=5, use_tfidf=True):
    counters = build_kmer_counts(seqs, k=k, use_canonical=True)
    if use_tfidf:
        counters = apply_tfidf(counters)
    return counters

#kernel computation and normalization
def compute_kernel_matrix(counters_train, counters_test=None):
    if counters_test is None:
        n = len(counters_train)
        K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                common = counters_train[i].keys() & counters_train[j].keys()
                val = sum(counters_train[i][key] * counters_train[j][key] for key in common)
                K[i, j] = val
                K[j, i] = val
        return K
    else:
        n_test = len(counters_test)
        n_train = len(counters_train)
        K = np.zeros((n_test, n_train), dtype=np.float64)
        for i in range(n_test):
            for j in range(n_train):
                common = counters_test[i].keys() & counters_train[j].keys()
                K[i, j] = sum(counters_test[i][key] * counters_train[j][key] for key in common)
        return K

def normalize_kernel(K, diag_train=None, diag_test=None):
    epsilon = 1e-8
    if diag_train is None and diag_test is None:
        diag = np.sqrt(np.diag(K))
        diag[diag < epsilon] = epsilon
        return K / (diag[:, None] * diag[None, :])
    else:
        diag_train = np.array(diag_train)
        diag_test = np.array(diag_test)
        diag_train[diag_train < epsilon] = epsilon
        diag_test[diag_test < epsilon] = epsilon
        return K / (diag_test[:, None] * diag_train[None, :])

# Prediction function

def predict(alpha,K):
    f = np.dot(K,alpha)
    predictions = np.sign(f)
    return f, predictions

def compute_feature_matrix_with_dict(sequences, k, kmer_to_index):
    feature_list = []
    for seq in sequences:
        fv = compute_feature_vector(seq, k, kmer_to_index)
        feature_list.append(fv)
    features = np.vstack(feature_list)
    return features

def center_kernel(K):
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n

##############################################
# LDA
##############################################

def compute_lda_parameters(X, y):
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    n_features = X.shape[1]
    mean_vectors = {c: np.mean(X[y == c], axis=0) for c in classes}
    S_W = np.zeros((n_features, n_features))
    for c in classes:
        X_c = X[y == c]
        S_W += np.cov(X_c, rowvar=False, bias=True) * X_c.shape[0]
    S_W /= X.shape[0]
    return mean_vectors, S_W

def train_lda(X, y):
    y = np.asarray(y).ravel()
    if X.shape[0] != len(y):
        X = X.T
    classes = np.unique(y)
    mean_vectors = {c: np.mean(X[y == c], axis=0) for c in classes}
    n, d = X.shape
    Sw = np.zeros((d, d))
    for c in classes:
        X_c = X[y == c]
        cov_c = np.cov(X_c, rowvar=False, bias=True)
        Sw += cov_c * X_c.shape[0]
    Sw /= n
    w = np.linalg.pinv(Sw) @ (mean_vectors[1] - mean_vectors[-1])
    b = -0.5 * (mean_vectors[1] + mean_vectors[-1]).dot(w)
    return w, b

def predict_lda(X, w, b):
    f = X.dot(w) + b
    return np.sign(f)

def train_regularized_lda(X, y, lambda_reg=0.1):
    y = np.asarray(y).ravel()
    if X.shape[0] != len(y):
        X = X.T
    classes = np.unique(y)

    mean_vectors = {c: np.mean(X[y == c], axis=0) for c in classes}
    n, d = X.shape
    Sw = np.zeros((d, d))
    for c in classes:
        X_c = X[y == c]
        cov_c = np.cov(X_c, rowvar=False, bias=True)
        Sw += cov_c * X_c.shape[0]
    Sw /= n
    Sw_reg = Sw + lambda_reg * np.eye(d)

    c0, c1 = np.sort(classes)
    w = np.linalg.pinv(Sw_reg) @ (mean_vectors[c1] - mean_vectors[c0])
    b = -0.5 * (mean_vectors[c1] + mean_vectors[c0]).dot(w)
    return w, b

##############################################
# Substring Kernel
##############################################

def extract_substrings(sequence, k):
    substrings = []
    L = len(sequence)
    for i in range(L - k + 1):
        substrings.append(sequence[i:i+k])
    return substrings

def build_substring_dictionary(sequences, k):
    substring_set = set()
    for seq in sequences:
        subs = extract_substrings(seq, k)
        for s in subs:
            substring_set.add(s)
    return {sub: i for i, sub in enumerate(sorted(substring_set))}

def compute_document_frequency(sequences, k, substring_to_index):
    n_subs = len(substring_to_index)
    df = np.zeros(n_subs)
    for seq in sequences:
        subs_in_seq = set(extract_substrings(seq, k))
        for s in subs_in_seq:
            if s in substring_to_index:
                df[substring_to_index[s]] += 1
    return df

def compute_tfidf_feature_vector(sequence, k, substring_to_index, idf):
    n_subs = len(substring_to_index)
    feature_vec = np.zeros(n_subs)
    subs = extract_substrings(sequence, k)
    for s in subs:
        if s in substring_to_index:
            feature_vec[substring_to_index[s]] += 1
    feature_vec *= idf
    return feature_vec

def compute_tfidf_feature_matrix(sequences, k, min_df=4, max_df_ratio=0.9, sublinear_tf=True):
    substring_to_index = build_substring_dictionary(sequences, k)
    n_subs = len(substring_to_index)

    df = compute_document_frequency(sequences, k, substring_to_index)
    n_seq = len(sequences)

    keep_mask = (df >= min_df) & (df <= max_df_ratio * n_seq)

    old_to_new = {}
    idx = 0
    for old_i in range(n_subs):
        if keep_mask[old_i]:
            old_to_new[old_i] = idx
            idx += 1
    
    df_filtered = df[keep_mask]
    idf = np.log((n_seq + 1) / (df_filtered + 1)) + 1.0
    
    feature_list = []
    for seq in sequences:
        fv = np.zeros(idx, dtype=np.float64)
        subs = extract_substrings(seq, k)
        sub_counts = {}
        for s in subs:
            old_sub_idx = substring_to_index.get(s, None)
            if old_sub_idx is not None and old_sub_idx in old_to_new:
                new_sub_idx = old_to_new[old_sub_idx]
                sub_counts[new_sub_idx] = sub_counts.get(new_sub_idx, 0) + 1
        for new_i, c in sub_counts.items():
            tf = 1 + np.log(c) if (sublinear_tf and c > 0) else c
            fv[new_i] = tf
        fv *= idf
        feature_list.append(fv)
    X = np.vstack(feature_list)
    
    return X, substring_to_index, old_to_new, idf


def compute_tfidf_feature_matrix_with_dict(sequences, k, substring_to_index, old_to_new, idf,sublinear_tf=True):
    n_subs_final = len(idf)
    feature_list = []
    for seq in sequences:
        fv = np.zeros(n_subs_final, dtype=np.float64)
        subs = extract_substrings(seq, k)
        sub_counts = {}
        for s in subs:
            old_idx = substring_to_index.get(s, None)
            if old_idx is not None and old_idx in old_to_new:
                new_idx = old_to_new[old_idx]
                sub_counts[new_idx] = sub_counts.get(new_idx, 0) + 1
        for new_i, c in sub_counts.items():
            tf = 1 + np.log(c) if (sublinear_tf and c > 0) else c
            fv[new_i] = tf
        fv *= idf
        feature_list.append(fv)
    return np.vstack(feature_list)

def substring_tfidf_kernel_matrix(features):
    return features @ features.T

########################################
# Logistic Regression functions
########################################

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(w, X, y, reg=0.0):
    logits = X.dot(w)
    eps = 1e-10
    loss = -np.mean(y*np.log(np.clip(sigmoid(logits), eps, 1-eps)) + (1-y)*np.log(np.clip(1-sigmoid(logits), eps, 1-eps)))
    loss += 0.5 * reg * np.sum(w**2)
    return loss

def logistic_grad(w, X, y, reg=0.0):
    preds = sigmoid(X.dot(w))
    grad = X.T.dot(preds - y) / X.shape[0] + reg * w
    return grad

def train_logistic_regression(X, y, lr=0.1, num_iter=1000, reg=1e-4):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    for i in range(num_iter):
        grad = logistic_grad(w, X, y, reg)
        w -= lr * grad
        if i % 100 == 0:
            loss = logistic_loss(w, X, y, reg)
            #print(f"Iteration {i}, loss = {loss:.4f}")
    return w

def predict_logistic_regression(w, X, threshold=0.5):
    probs = sigmoid(X.dot(w))
    preds = (probs >= threshold).astype(int)
    return preds