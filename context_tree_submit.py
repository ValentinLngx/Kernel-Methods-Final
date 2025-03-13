import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

from loader import Xtr0, Xtr1, Xtr2, Ytr0, Ytr1, Ytr2, Xte0, Xte1, Xte2

from context_tree import (compute_context_tree_feature_matrix,compute_context_tree_feature_matrix_with_dict,context_tree_kernel_matrix)

Ytr0 = (2 * Ytr0 - 1).flatten()
Ytr1 = (2 * Ytr1 - 1).flatten()
Ytr2 = (2 * Ytr2 - 1).flatten()

"""
All best parameters per fold:
Fold 0 => {'max_depth': 11, 'lambda_val': 0.10604918078027928, 'bias': 0.03667642361893528, 'ridge_eps': 1.0648826087041697e-09}
Fold 1 => {'max_depth': 11, 'lambda_val': 0.0030346107160643516, 'bias': 0.019409078399341757, 'ridge_eps': 5.986539761760399e-07}
Fold 2 => {'max_depth': 12, 'lambda_val': 0.017385345138507237, 'bias': 0.05129817159203445, 'ridge_eps': 1.1058153376304655e-08}
"""

best_params = {
    0: {"max_depth":  10,"lambda_val": 0.08867968726299837,"bias":       0.008668091350122475,"ridge_eps":  8.088811673669314e-06},
    1: {"max_depth":  10,"lambda_val": 0.09692143205763142,"bias":       0.09034048928393573,"ridge_eps":  4.493531901462385e-05},
    2: {"max_depth":  12,"lambda_val": 0.017385345138507237,"bias":       0.05129817159203445,"ridge_eps":  1.1058153376304655e-08}}

def solve_svm_qp(K_train, y_train, lambda_val):
    n = K_train.shape[0]

    P = matrix((2.0 * K_train).astype(np.double))
    q = matrix((-2.0 * y_train).astype(np.double))

    G1 = -np.diag(y_train)
    G2 =  np.diag(y_train)
    G  = matrix(np.vstack((G1, G2)).astype(np.double))

    h1 = np.zeros(n)
    h2 = np.ones(n) / (2.0 * lambda_val * n)
    h  = matrix(np.hstack((h1, h2)).astype(np.double))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    alpha = np.array(sol['x']).flatten()
    return alpha

def run_context_tree_svm(X_train, y_train, X_test,max_depth, lambda_val, bias, ridge_eps):
    seqs_train = X_train["seq"].tolist()
    features_train, context_dict = compute_context_tree_feature_matrix(seqs_train, max_depth)
    K_train = context_tree_kernel_matrix(features_train)

    n = K_train.shape[0]
    K_train += ridge_eps * np.eye(n)

    alpha = solve_svm_qp(K_train, y_train, lambda_val)

    seqs_test = X_test["seq"].tolist()
    features_test = compute_context_tree_feature_matrix_with_dict(seqs_test, max_depth, context_dict)
    K_test = features_test.dot(features_train.T)

    f_test = K_test.dot(alpha) + bias
    preds = np.sign(f_test)
    return preds

preds_test0 = run_context_tree_svm(Xtr0, Ytr0, Xte0,best_params[0]["max_depth"],best_params[0]["lambda_val"],best_params[0]["bias"],best_params[0]["ridge_eps"])

preds_test1 = run_context_tree_svm(Xtr1, Ytr1, Xte1,best_params[1]["max_depth"],best_params[1]["lambda_val"],best_params[1]["bias"],best_params[1]["ridge_eps"])

preds_test2 = run_context_tree_svm(Xtr2, Ytr2, Xte2,best_params[2]["max_depth"],best_params[2]["lambda_val"],best_params[2]["bias"],best_params[2]["ridge_eps"])

preds_test0 = np.where(preds_test0 == -1, 0, 1)
preds_test1 = np.where(preds_test1 == -1, 0, 1)
preds_test2 = np.where(preds_test2 == -1, 0, 1)

final_predictions = np.concatenate([preds_test0, preds_test1, preds_test2])
ids = np.arange(len(final_predictions))

submission = pd.DataFrame({"Id": ids, "Bound": final_predictions})
submission.to_csv("submission_context_tree_bisrepetita.csv", index=False)

print("Submission file created as 'submission_context_tree.csv'.")
