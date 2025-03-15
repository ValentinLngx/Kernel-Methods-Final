import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

# load data
from loader import Xtr0, Xtr1, Xtr2, Xte0, Xte1, Xte2
from loader import Ytr0, Ytr1, Ytr2

# convert training labels from {0,1} to {-1,1}
Ytr0 = (2 * Ytr0 - 1).flatten()
Ytr1 = (2 * Ytr1 - 1).flatten()
Ytr2 = (2 * Ytr2 - 1).flatten()


# Part 1: Context Tree Kernel Model for Dataset 0

from context_tree_utils import (compute_context_tree_feature_matrix_weighted,
                                context_tree_kernel_matrix,
                                compute_context_tree_feature_matrix_with_dict_weighted,
                                predict_context)

def train_and_predict_context_tree_kernel(Xtr_df, Ytr, Xte_df, max_depth, lambda_val, gamma):
    # Get training sequences and compute context tree features
    sequences_train = Xtr_df['seq'].tolist()
    labels_train = Ytr
    features_train, context_dict = compute_context_tree_feature_matrix_weighted(sequences_train, max_depth, gamma=gamma)
    
    # Compute training kernel matrix
    K_train = context_tree_kernel_matrix(features_train)
    n = K_train.shape[0]
    
    # Formulate and solve the QP problem via cvxopt
    P = matrix(2.0 * K_train.astype(np.double))
    q = matrix(-2.0 * labels_train.astype(np.double))
    G1 = -np.diag(labels_train)
    G2 = np.diag(labels_train)
    G = matrix(np.vstack((G1, G2)))
    h1 = np.zeros(n)
    h2 = np.ones(n) / (2.0 * lambda_val * n)
    h = matrix(np.hstack((h1, h2)).astype(np.double))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    alpha = np.array(sol['x']).flatten()
    
    #compute test features and test kernel matrix
    sequences_test = Xte_df['seq'].tolist()
    features_test = compute_context_tree_feature_matrix_with_dict_weighted(sequences_test, max_depth, context_dict, gamma=gamma)
    K_test = features_test.dot(features_train.T)
    f_test, preds_test = predict_context(alpha, K_test)
    
    #convert predictions from {-1, +1} to {0,1}
    preds_test_binary = np.where(preds_test == -1, 0, 1)
    return preds_test_binary

# get predictions
print("Computing predictions for Dataset 0...")
preds0 = train_and_predict_context_tree_kernel(Xtr0, Ytr0, Xte0, max_depth=12, lambda_val=0.2, gamma=0.8)
ids0 = np.arange(len(preds0))
sub0 = pd.DataFrame({'Id': ids0, 'Bound': preds0})
print("Dataset 0 predictions computed.")


# Part 2: Ensemble Spectrum Kernel SVM for Datasets 1 and 2

from utils import build_improved_kmer_counts, compute_kernel_matrix, normalize_kernel

#function to train SVM and predict test labels
def train_and_predict_multi_k_kernel(Xtr_df, Ytr, Xte_df, kset, C, use_tfidf=True):
    #get training and test sequences
    seqs_train = Xtr_df["seq"].tolist()
    seqs_test  = Xte_df["seq"].tolist()
    
    # Build improved k-mer counters for each k in the ensemble
    train_counters = {k: build_improved_kmer_counts(seqs_train, k=k, use_tfidf=use_tfidf) for k in kset}
    test_counters  = {k: build_improved_kmer_counts(seqs_test, k=k, use_tfidf=use_tfidf) for k in kset}
    
    #compute the ensemble training kernel
    K_train = None
    for k in kset:
        K_k = compute_kernel_matrix(train_counters[k], None)
        if K_train is None:
            K_train = K_k
        else:
            K_train += K_k
    K_train = normalize_kernel(K_train)
    diag_train = np.diag(K_train)
    
    #train SVM using cvxopt QP formulation
    n = K_train.shape[0]
    P = matrix(2.0 * K_train.astype(np.double))
    q = matrix(-2.0 * Ytr.astype(np.double))
    G1 = -np.diag(Ytr)
    G2 = np.diag(Ytr)
    # set the upper bound on dual variables using the relation: 1/(2*lambda*n) = C  => lambda = 1/(2*C*n)
    lambda_val_new = 1.0 / (2 * C * n)
    G = matrix(np.vstack((G1, G2)))
    h1 = np.zeros(n)
    h2 = np.ones(n) / (2.0 * lambda_val_new * n)
    h = matrix(np.hstack((h1, h2)).astype(np.double))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    alpha = np.array(sol['x']).flatten()
    
    #compute the ensemble test kernel (cross-kernel)
    K_test = None
    for k in kset:
        K_k_test = compute_kernel_matrix(train_counters[k], test_counters[k])
        if K_test is None:
            K_test = K_k_test
        else:
            K_test += K_k_test
    #for normalization, compute symmetric test kernel
    K_test_sym = None
    for k in kset:
        K_k_test_sym = compute_kernel_matrix(test_counters[k], None)
        if K_test_sym is None:
            K_test_sym = K_k_test_sym
        else:
            K_test_sym += K_k_test_sym
    K_test_sym = normalize_kernel(K_test_sym)
    diag_test = np.diag(K_test_sym)
    K_test = normalize_kernel(K_test, diag_train=diag_train, diag_test=diag_test)
    
    #compute decision function f = K_test^T * alpha, then prediction = sign(f)
    f_test = K_test.dot(alpha)
    y_pred = np.where(f_test >= 0, 1, -1)
    return y_pred

print("Computing predictions for Dataset 1...")
pred1 = train_and_predict_multi_k_kernel(Xtr1, Ytr1, Xte1, kset=(6, 9), C=1.0, use_tfidf=True)
print("Computing predictions for Dataset 2...")
pred2 = train_and_predict_multi_k_kernel(Xtr2, Ytr2, Xte2, kset=(4, 9), C=10.0, use_tfidf=True)

# Convert predictions from {-1, +1} to {0,1}
conv_pred1 = np.where(pred1 == -1, 0, 1)
conv_pred2 = np.where(pred2 == -1, 0, 1)

print("Datasets 1 and 2 predictions computed.")


# Part 3: Combine predictions from all three datasets and create submission file

N0 = sub0.shape[0]        
N1 = len(conv_pred1)       
N2 = len(conv_pred2)       

sub1 = pd.DataFrame({
    'Id': np.arange(N0, N0 + N1),
    'Bound': conv_pred1
})
sub2 = pd.DataFrame({
    'Id': np.arange(N0 + N1, N0 + N1 + N2),
    'Bound': conv_pred2
})

submission = pd.concat([sub0, sub1, sub2], axis=0).reset_index(drop=True)
submission = submission.sort_values(by='Id').reset_index(drop=True)

submission.to_csv("submission_final.csv", index=False)
print("Submission file 'submission_final.csv' created successfully!")
