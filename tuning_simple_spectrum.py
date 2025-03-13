import numpy as np
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split
import optuna

from loader import Ytr0, Ytr1, Ytr2
from loader import Xtr0, Xtr1, Xtr2
from utils import compute_feature_matrix, compute_feature_matrix_with_dict, spectrum_kernel_matrix

Ytr0 = 2 * Ytr0 - 1
Ytr1 = 2 * Ytr1 - 1
Ytr2 = 2 * Ytr2 - 1

Ytr0 = Ytr0.flatten()
Ytr1 = Ytr1.flatten()
Ytr2 = Ytr2.flatten()

Xtr_list = [Xtr0, Xtr1, Xtr2]
Xval_list = [None, None, None]
Ytr_list = [Ytr0, Ytr1, Ytr2]
Yval_list = [None, None, None]

for i in range(3):
    Xtr_list[i], Xval_list[i], Ytr_list[i], Yval_list[i] = train_test_split(Xtr_list[i], Ytr_list[i], test_size=0.2, random_state=1)

def objective(trial, fold):
    lambda_val = trial.suggest_float("lambda", 0.01, 1.0, log=True)
    k_val = trial.suggest_int("k", 4, 8)
    eps = trial.suggest_float("eps", 1e-6, 1e-3, log=True)
    bias = trial.suggest_float("bias", -.5, .5)

    sequences_train = Xtr_list[fold]['seq'].tolist()
    features, kmer_to_index = compute_feature_matrix(sequences_train, k_val)
    K = spectrum_kernel_matrix(features)
    n = K.shape[0]

    K_reg = K + eps * np.eye(n)
    
    P = matrix(2 * K_reg.astype(np.double))
    q = matrix(-2 * Ytr_list[fold].astype(np.double))

    G1 = -np.diag(Ytr_list[fold])
    G2 = np.diag(Ytr_list[fold])
    G = matrix(np.vstack((G1, G2)))
    h1 = np.zeros(n)
    h2 = np.ones(n) / (2.0 * lambda_val * n)
    h = matrix(np.hstack((h1, h2)).astype(np.double))
    
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    alpha = np.array(sol['x']).flatten()

    sequences_val = Xval_list[fold]['seq'].tolist()
    features_val = compute_feature_matrix_with_dict(sequences_val, k_val, kmer_to_index)
    K_val = features_val @ features.T

    f_val = K_val.dot(alpha) + bias
    predicted_labels_val = np.sign(f_val)
    val_acc = np.mean(predicted_labels_val == Yval_list[fold])

    return 1.0 - val_acc

n_trials = 100 
best_params = {}

for fold in [0, 1, 2]:
    print(f"Optimizing for fold {fold}...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, fold), n_trials=n_trials, n_jobs=-1)
    best_params[fold] = study.best_trial.params
    best_val_acc = 1.0 - study.best_trial.value
    print(f"Fold {fold}: Best Validation Accuracy = {best_val_acc:.4f}")
    print(f"Best hyperparameters: {study.best_trial.params}\n")

print("All best parameters per fold:")
print(best_params)