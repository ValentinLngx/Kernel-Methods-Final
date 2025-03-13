import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

# Load training and test data for dataset 0
from loader import Xtr0, Ytr0, Xte0
from context_tree import (compute_context_tree_feature_matrix_weighted,context_tree_kernel_matrix,compute_context_tree_feature_matrix_with_dict_weighted, predict_context)


Ytr0 = (2 * Ytr0 - 1).flatten()

sequences_train = Xtr0['seq'].tolist()
labels_train = Ytr0

max_depth = 12
lambda_val = 0.2
gamma = 0.8       # weight factor for context tree
features_train, context_dict = compute_context_tree_feature_matrix_weighted(sequences_train, max_depth, gamma=gamma)

K_train = context_tree_kernel_matrix(features_train) # Compute kernel matrix

n = K_train.shape[0]
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

f_train, preds_train = predict_context(alpha, K_train) # Evaluate training accuracy
train_acc = np.mean(preds_train == labels_train)
print("Training accuracy:", train_acc) 


sequences_test = Xte0['seq'].tolist()
features_test = compute_context_tree_feature_matrix_with_dict_weighted(sequences_test, max_depth, context_dict, gamma=gamma)

K_test = features_test.dot(features_train.T)
f_test, preds_test = predict_context(alpha, K_test)

preds_test_binary = np.where(preds_test == -1, 0, 1) # Convert predictions from {-1, +1} to {0, 1}

ids = np.arange(len(preds_test_binary))
submission = pd.DataFrame({'Id': ids, 'Bound': preds_test_binary})
submission.to_csv("submission_dataset0.csv", index=False)
print("Submission file created as 'submission_dataset0.csv'.")
