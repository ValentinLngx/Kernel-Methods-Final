import numpy as np
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split
import optuna

from loader import Xtr0, Ytr0, Xtr1, Ytr1, Xtr2, Ytr2

from utils import (compute_feature_matrix,compute_feature_matrix_with_dict,spectrum_kernel_matrix,
    compute_tfidf_feature_matrix,compute_tfidf_feature_matrix_with_dict,substring_tfidf_kernel_matrix)
from context_tree_utils import (compute_context_tree_feature_matrix_weighted,context_tree_kernel_matrix,compute_context_tree_feature_matrix_with_dict_weighted)
import mismatch_kernel_utils as mismatch_kernel


for dataset_index in [0, 1, 2]:
    print(f"=== Dataset {dataset_index} ===")
    if dataset_index == 0:
        X = Xtr0
        Y = Ytr0
    elif dataset_index == 1:
        X = Xtr1
        Y = Ytr1
    else:
        X = Xtr2
        Y = Ytr2

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    train_seqs = X_train['seq'].tolist()
    test_seqs  = X_test['seq'].tolist()

    def objective(trial: optuna.Trial) -> float:
        m1 = trial.suggest_int("m1", 0, 3)
        m2 = trial.suggest_int("m2", 0, 3)
        alpha_kernel = trial.suggest_float("alpha_kernel", 0.1, 1.0, step=0.1)

        max_depth = trial.suggest_int("max_depth", 3, 10)
        gamma_ctx = trial.suggest_float("gamma_ctx", 0.5, 0.99, step=0.01)

        k_spectrum = trial.suggest_int("k_spectrum", 3, 10)

        k_substring = trial.suggest_int("k_substring", 3, 8)
        min_df = trial.suggest_int("min_df", 1, 6)
        max_df_ratio = trial.suggest_float("max_df_ratio", 0.6, 0.99, step=0.01)

        w_mismatch1  = trial.suggest_float("w_mismatch1", 0.0, 1.0)
        w_mismatch2  = trial.suggest_float("w_mismatch2", 0.0, 1.0)
        w_context    = trial.suggest_float("w_context",   0.0, 1.0)
        w_spectrum   = trial.suggest_float("w_spectrum",  0.0, 1.0)
        w_substring  = trial.suggest_float("w_substring", 0.0, 1.0)

        lambda_ = trial.suggest_float("lambda_", 1e-3, 10.0, log=True)

        K_mismatch1 = mismatch_kernel.mismatch_kernel_matrix(train_seqs, k_spectrum, m1, alpha_kernel)
        K_mismatch2 = mismatch_kernel.mismatch_kernel_matrix(train_seqs, k_spectrum, m2, alpha_kernel)

        features_context, context_dict = compute_context_tree_feature_matrix_weighted(train_seqs, max_depth, gamma=gamma_ctx)
        K_context = context_tree_kernel_matrix(features_context)

        features_spectrum, kmer_to_index = compute_feature_matrix(train_seqs, k_spectrum)
        K_spectrum_mat = spectrum_kernel_matrix(features_spectrum)

        features_tfidf, substring_to_index, old_to_new, idf = compute_tfidf_feature_matrix(train_seqs,k=k_substring,
            min_df=min_df,max_df_ratio=max_df_ratio,sublinear_tf=True)
        K_substring = substring_tfidf_kernel_matrix(features_tfidf)

        K_train = (w_mismatch1 * K_mismatch1 + w_mismatch2 * K_mismatch2 +w_context   * K_context + w_spectrum  * K_spectrum_mat + w_substring * K_substring)

        n = K_train.shape[0]
        P = matrix((2.0 * K_train).astype(np.double))
        q = matrix((-2.0 * y_train).astype(np.double))

        G1 = -np.diag(y_train)
        G2 =  np.diag(y_train)
        G = matrix(np.vstack((G1, G2)).astype(np.double))

        h1 = np.zeros(n)
        C = 1.0 / (2.0 * lambda_ * n)
        h2 = np.ones(n) * C
        h = matrix(np.hstack((h1, h2)).astype(np.double))

        solvers.options["show_progress"] = False
        sol = solvers.qp(P, q, G, h)
        alpha = np.array(sol["x"]).flatten()

        K_mismatch1_test = mismatch_kernel.mismatch_kernel_matrix_between(test_seqs, train_seqs, k_spectrum, m1, alpha_kernel)
        K_mismatch2_test = mismatch_kernel.mismatch_kernel_matrix_between(test_seqs, train_seqs, k_spectrum, m2, alpha_kernel)

        features_context_test = compute_context_tree_feature_matrix_with_dict_weighted(test_seqs, max_depth, context_dict, gamma=gamma_ctx)
        K_context_test = features_context_test.dot(features_context.T)

        features_spectrum_test = compute_feature_matrix_with_dict(test_seqs, k_spectrum, kmer_to_index)
        K_spectrum_test = features_spectrum_test @ features_spectrum.T

        features_tfidf_test = compute_tfidf_feature_matrix_with_dict(test_seqs, k=k_substring,substring_to_index=substring_to_index,
                                                                     old_to_new=old_to_new,idf=idf,sublinear_tf=True)
        K_substring_test = features_tfidf_test @ features_tfidf.T

        K_test = (w_mismatch1 * K_mismatch1_test +w_mismatch2 * K_mismatch2_test +w_context   * K_context_test +w_spectrum  * K_spectrum_test +w_substring * K_substring_test)

        f_test = K_test.dot(alpha)
        preds_test = np.sign(f_test)
        test_acc = np.mean(preds_test == y_test)

        return test_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True, n_jobs=1)

    print("\nBest trial for dataset", dataset_index)
    print(f"  Value (test accuracy): {study.best_value}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

