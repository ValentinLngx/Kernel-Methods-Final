import numpy as np
from itertools import product

def build_kmer_index(k, alphabet=('A', 'C', 'G', 'T')):
    all_kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    kmer_to_idx = {kmer: idx for idx, kmer in enumerate(all_kmers)}
    return all_kmers, kmer_to_idx

def generate_neighbors_with_distances(kmer, m, alphabet=('A','C','G','T')):
    neighbors = []
    def backtrack(prefix, idx, mismatches):
        if idx == len(kmer):
            neighbors.append((prefix, mismatches))
            return
        backtrack(prefix + kmer[idx], idx + 1, mismatches)
        if mismatches < m:
            for c in alphabet:
                if c != kmer[idx]:
                    backtrack(prefix + c, idx + 1, mismatches + 1)
    backtrack('', 0, 0)
    return neighbors

def build_kmer_neighbor_map(all_kmers, m, alphabet=('A','C','G','T')):
    nb_map = {}
    for km in all_kmers:
        nb_map[km] = generate_neighbors_with_distances(km, m, alphabet)
    return nb_map

def build_weighted_feature_vector(seq, k, kmer_to_idx, nb_map, alpha=0.5):
    L = len(seq)
    dim = len(kmer_to_idx)
    feature_vector = np.zeros(dim)

    for start in range(L - k + 1):
        sub = seq[start:start + k]
        if sub in nb_map:
            for nb, dist in nb_map[sub]:
                idx = kmer_to_idx[nb]
                feature_vector[idx] += alpha ** dist

    return feature_vector

def mismatch_kernel_matrix(sequences, k, m, alpha=0.5, alphabet=('A','C','G','T')):
    all_kmers, kmer_to_idx = build_kmer_index(k, alphabet)
    nb_map = build_kmer_neighbor_map(all_kmers, m, alphabet)

    n = len(sequences)
    features = np.zeros((n, len(all_kmers)))
    for i, seq in enumerate(sequences):
        features[i] = build_weighted_feature_vector(seq, k, kmer_to_idx, nb_map, alpha)

    return features @ features.T
