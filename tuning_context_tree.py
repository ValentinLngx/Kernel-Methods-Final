import numpy as np
import optuna
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split # We use it to split the data into training and validation sets without doing a function from scratch

from loader import Ytr0, Ytr1, Ytr2
from loader import Xtr0, Xtr1, Xtr2

from context_tree_utils import (compute_context_tree_feature_matrix,context_tree_kernel_matrix,compute_context_tree_feature_matrix_with_dict,predict_context)

Ytr0 = (2 * Ytr0 - 1).flatten()
Ytr1 = (2 * Ytr1 - 1).flatten()
Ytr2 = (2 * Ytr2 - 1).flatten()


Xtr_list = [Xtr0, Xtr1, Xtr2]
Ytr_list = [Ytr0, Ytr1, Ytr2]
Xval_list = [None, None, None]
Yval_list = [None, None, None]

for i in range(3):
    Xtr_list[i], Xval_list[i], Ytr_list[i], Yval_list[i] = train_test_split(Xtr_list[i],Ytr_list[i],test_size=0.2, random_state=1)

def objective(trial, fold):
    max_depth  = trial.suggest_int("max_depth", 9, 12)
    lambda_val = trial.suggest_float("lambda_val", 1e-4, .2, log=True)
    bias       = trial.suggest_float("bias", -0.1, 0.1)
    eps = trial.suggest_float("ridge_eps", 1e-10, 1e-5, log=True)

    X_train  = Xtr_list[fold]
    Y_train  = Ytr_list[fold]
    seqs_train = X_train["seq"].tolist()

    features_train, context_dict = compute_context_tree_feature_matrix(seqs_train, max_depth)
    K_train = context_tree_kernel_matrix(features_train)  # NxN

    n = K_train.shape[0]
    K_train = K_train + eps * np.eye(n) 

    P = matrix((2.0 * K_train).astype(np.double))
    q = matrix((-2.0 * Y_train).astype(np.double))

    G1 = -np.diag(Y_train)
    G2 =  np.diag(Y_train)
    G  = matrix(np.vstack((G1, G2)).astype(np.double))

    h1 = np.zeros(n)
    h2 = np.ones(n) / (2.0 * lambda_val * n)
    h  = matrix(np.hstack((h1, h2)).astype(np.double))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    alpha = np.array(sol['x']).flatten()

    X_val   = Xval_list[fold]
    Y_val   = Yval_list[fold]
    seqs_val = X_val["seq"].tolist()

    features_val = compute_context_tree_feature_matrix_with_dict(seqs_val, max_depth,context_dict)
    K_val = features_val.dot(features_train.T)

    f_val, preds_val = predict_context(alpha, K_val)
    f_val = f_val + bias
    preds_val = np.sign(f_val)

    val_acc = np.mean(preds_val == Y_val)
    return 1.0 - val_acc

n_trials = 30 
best_params = {}

for fold_idx in [0, 1, 2]:
    print(f"Optimizing for fold {fold_idx}...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda tr: objective(tr, fold_idx), n_trials=n_trials, n_jobs=6)

    best_fold_params = study.best_trial.params
    best_fold_val_acc = 1.0 - study.best_trial.value
    best_params[fold_idx] = best_fold_params

    print(f"Fold {fold_idx}: Best Validation Accuracy = {best_fold_val_acc:.4f}")
    print("Best hyperparameters:", best_fold_params)
    print()

print("All best parameters per fold:")
for fold_idx in best_params:
    print(f"Fold {fold_idx} => {best_params[fold_idx]}")
