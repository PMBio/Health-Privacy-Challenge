from typing import Optional, Tuple, Dict, Any
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, auc)

from sklearn.utils import check_random_state
from scipy.special import logsumexp


src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from mia.utils.prepare_data import MIADataLoader
from mia.models.base import BaseMIAModel
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
from domias.baselines import ( LOGAN_D1,
                              GAN_leaks_cal) #GAN_leaks, MC,

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Adapted from https://github.com/holarissun/DOMIAS/blob/main/src/domias/baselines.py


class DOMIASBaselineModels(BaseMIAModel):
    def __init__(self, 
                 config: Dict[str, Any], 
                 synthetic_file: str,
                 membership_test_file: str,
                 membership_lbl_file: str,
                 mia_experiment_name:str,
                 reference_file:str = None,
                 test_on_real:bool = False): ## test this as baseline.. 
        super().__init__(config, 
                         synthetic_file, 
                         membership_test_file, 
                         membership_lbl_file,
                         mia_experiment_name,
                         reference_file)
        
        self.test_on_real = test_on_real
        self.random_seed = config["dataset_config"]["random_seed"]

    def run_attack(self):
        data_loader = MIADataLoader(
            synthetic_file=self.synthetic_file,
            membership_test_file=self.membership_test_file,
            membership_lbl_file = self.membership_lbl_file,
            membership_label_col=self.membership_label_col,
            generator_model=self.generator_model,
            reference_file=self.reference_file
        )

        if self.test_on_real:
            save_dir = os.path.join(self.home_dir, self.config["dir_list"]["data_splits"])
            synthetic_data, synthetic_labels = data_loader.load_original_data(save_dir,  
                                                            self.dataset_name)
        else:
            synthetic_data, synthetic_labels = data_loader.load_synthetic_data()
            ## need synthetic labels 
            
                    # Decide whether to align real data

        print("inside domaias baselines")
        print(synthetic_data.shape[1])
        if synthetic_data.shape[1] < 978:  # or X_train_real.shape[1] if known
            align_to_synthetic = synthetic_data
        else:
            align_to_synthetic = None

        print(synthetic_data.shape[1])
        X_test = data_loader.load_membership_dataset(align_to_synthetic)
        y_test = data_loader.load_membership_labels()

        if y_test is not None:
            assert len(X_test) == len(y_test), "mismatch in test data and label lengths."
        
        reference = data_loader.load_reference_data(align_to_synthetic)

        scores = run_baselines(X_test, synthetic_data, synthetic_labels, reference, reference, None)

        return scores, y_test
    
    
    

    def run_attack_(self, n_bootstrap=3):
        """
        Runs the membership inference attack with balanced bootstrapping.
        Returns concatenated scores and labels for evaluation, plus a bootstrap summary.
        """
        data_loader = MIADataLoader(
            synthetic_file=self.synthetic_file,
            membership_test_file=self.membership_test_file,
            membership_lbl_file=self.membership_lbl_file,
            membership_label_col=self.membership_label_col,
            generator_model=self.generator_model,
            reference_file=self.reference_file
        )

        # Load synthetic and membership data
        if self.test_on_real:
            save_dir = os.path.join(self.home_dir, self.config["dir_list"]["data_splits"])
            synthetic_data = data_loader.load_original_data(save_dir, self.dataset_name)
        else:
            synthetic_data, synthetic_labels = data_loader.load_synthetic_data()
            
        X_test = np.array(data_loader.load_membership_dataset())
        y_test = np.array(data_loader.load_membership_labels())
        assert len(X_test) == len(y_test), "Mismatch in test data and label lengths."

        reference = data_loader.load_reference_data()
        rng = np.random.RandomState(self.andom_seed)

        # Indices for members and non-members
        member_idx = np.where(y_test == 1)[0]
        non_member_idx = np.where(y_test == 0)[0]

        all_scores_dict = {key: [] for key in run_baselines(X_test[:1], synthetic_data, reference, reference, None).keys()}
        all_labels = []
        # Determine balanced sample size
        n_sample = min(len(member_idx), len(non_member_idx))

        for _ in range(n_bootstrap):
            sampled_member_idx = rng.choice(member_idx, size=n_sample, replace=True)
            # Use all non-members
            sampled_non_member_idx = non_member_idx  # no replacement
            #sampled_non_member_idx = rng.choice(non_member_idx, size=n_sample, replace=False)

            combined_idx = np.concatenate([sampled_member_idx, sampled_non_member_idx])
            rng.shuffle(combined_idx)

            X_eval = X_test[combined_idx]
            y_eval = y_test[combined_idx]

            # Run baseline attack (GAN_leaks, Likelihood-Ratio, etc.)
            scores = run_baselines(X_eval, synthetic_data, reference, reference, None)

            # Append scores for each attack model
            for key in scores.keys():
                all_scores_dict[key].append(scores[key])
            all_labels.append(y_eval)


        # Concatenate all bootstraps for evaluation function
        scores_concat = {key: np.concatenate(all_scores_dict[key], axis=0) for key in all_scores_dict}
        labels_concat = np.concatenate(all_labels, axis=0)



        return scores_concat, labels_concat



def d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if len(X.shape) == 1:
        return np.sum((X - Y) ** 2, axis=1)
    else:
        res = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            res[i] = d(x, Y)

        return res


def d_min(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.min(d(X, Y))


def GAN_leaks(X_test: np.ndarray, X_G: np.ndarray) -> np.ndarray:
    print("=== GAN_leaks debug ===")
    scores = np.zeros(X_test.shape[0])
    for i, x in enumerate(X_test):
        scores[i] = np.exp(-d_min(x, X_G))
        print("d_min:", d_min(x, X_G))
        assert not np.isinf(scores[i]), f"Found inf: -d_min = {-d_min(x, X_G)}"

    
    return scores


def GAN_leaks_modified(X_test: np.ndarray, X_G: np.ndarray) -> np.ndarray:
    print("=== GAN_leaks debug ===")
    
    # Step 1: compute all d_min values
    d_min_vals = np.array([d_min(x, X_G) for x in X_test])
    print(f"d_min stats: mean  {d_min_vals.mean()}, median {np.median(d_min_vals)}, min {d_min_vals.min()}, max {d_min_vals.max()}")
    
    # Step 2: scale distances by median to avoid underflow
    scale = np.median(d_min_vals)
    
    # Step 3: compute scores
    scores = np.exp(-d_min_vals / scale)
    
    return scores




def MC(X_test: np.ndarray, X_G: np.ndarray) -> np.ndarray:

    print("=== MC debug ===")
    scores = np.zeros(X_test.shape[0])
    distances = np.zeros((X_test.shape[0], X_G.shape[0]))
    print(f"distances shape: {distances.shape}")
    for i, x in enumerate(X_test):
        distances[i] = d(x, X_G)
    # median heuristic (Eq. 4 of Hilprecht)
    min_dist = np.min(distances, 1)
    print(f"min_dist stats: mean {np.mean(min_dist)} median {np.median(min_dist)} min {np.min(min_dist)} max {np.max(min_dist)}")
    assert min_dist.size == X_test.shape[0]
    epsilon = np.percentile(min_dist, 10) #np.median(min_dist)

    print(f"epsilon: {epsilon}")
    print("distances mean:", np.mean(distances))
    print("first 10 min_dist: {min_dist[:10]}")

    for i, x in enumerate(X_test):
        scores[i] = np.sum(distances[i] < epsilon)
    scores = scores / X_G.shape[0]
    return scores




def downstream_confidence_attack(X_candidates, X_synth, y_synth,
                                 model_type='lr',
                                 random_state=42,
                                 lr_C=1.0,
                                 rf_n_estimators=200,
                                 rf_max_depth=10):
    """
    Train a downstream classifier on synthetic data and score candidates by confidence.

    Scoring rule:
      score[i] = max_k P_model(y=k | x_i)   (higher -> more likely member)

    Parameters
    ----------
    X_candidates : array-like, shape (n_candidates, n_features)
      Features of candidate records to score.
    X_synth : array-like, shape (n_train, n_features)
      Synthetic training features.
    y_synth : array-like, shape (n_train,)
      Synthetic training labels (integers 0..K-1 or binary).
    model_type : {'lr','rf'}
      Model used for downstream classifier.
    random_state, lr_C, rf_* : model hyperparameters as before.
    return_proba_for_debug : bool
      If True, also return the full predicted-probability matrix (useful for analysis).

    Returns
    -------
    scores : ndarray, shape (n_candidates,)
      Confidence scores in [0,1] where larger means more confident (thus more likely a member).
    clf : trained sklearn-like classifier
    proba (optional) : ndarray, shape (n_candidates, n_classes)
      Returned only if return_proba_for_debug == True.
    """
    rng = check_random_state(random_state)

    if model_type == 'lr':
        clf = LogisticRegression(C=lr_C, solver='lbfgs', max_iter=1000,
                                 random_state=random_state)
    elif model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=rf_n_estimators,
                                     max_depth=rf_max_depth,
                                     n_jobs=-1, random_state=random_state)
    else:
        raise ValueError("model_type must be 'lr' or 'rf'")

    clf.fit(X_synth, y_synth)

    # get predicted probabilities in a robust way
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_candidates)   # shape (n, n_classes)
    else:
        # fallback for classifiers with decision_function
        dec = clf.decision_function(X_candidates)
        if dec.ndim == 1:
            # binary decision_function -> two-class logits convention: [-dec, +dec]
            logits = np.vstack([-dec, dec]).T
        else:
            logits = dec
        logits = logits - logsumexp(logits, axis=1, keepdims=True)
        probs = np.exp(logits)

    # confidence-based score: maximum class probability
    scores = probs.max(axis=1)

    return scores


def run_baselines(
    X_test: np.ndarray,
    #Y_test: np.ndarray,
    X_G: np.ndarray, #syn data
    y_G: np.ndarray, #syn label
    X_ref: np.ndarray,
    X_ref_GLC: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[dict, dict]:
    score = {}
   
    score["MC"] = MC(X_test, X_G)
    score["gan_leaks"] = GAN_leaks_modified(X_test, X_G)
    score["conf_lr"] = downstream_confidence_attack(X_test, X_G, y_G, model_type='lr')
    score["conf_rf"] = downstream_confidence_attack(X_test, X_G, y_G, model_type='rf')
  
    if X_ref is not None:
        score["LOGAN_D1"] = LOGAN_D1(X_test, X_G, X_ref)
        score["gan_leaks_cal"] = GAN_leaks_cal(X_test, X_G, X_ref_GLC)
    
        ### apply PCA 
        pca_ref = perform_pca(X_ref, n_components=150)
        pca_test = perform_pca(X_test, n_components=150)
        pca_synth = perform_pca(X_G, n_components=150)

        score["domias_kde"] = kde_domias(pca_test, pca_synth, pca_ref)
        #score["domias_bnaf"] = kde_domias(pca_test, pca_synth, pca_ref, "bnaf")


    return score


def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    #data = StandardScaler().fit_transform(data)
    principal_components = pca.fit_transform(data)
    print(np.sum(pca.explained_variance_ratio_))

    return principal_components

def kde_domias(
            X_test: np.ndarray,
            synth_set: np.ndarray,
            reference_set: np.ndarray,
            density_estimator:str = "kde"):
  
    # BNAF was memory intensive and couldnt be tested 
    # BNAF for pG
    if density_estimator == "bnaf":
        _, p_G_model = density_estimator_trainer(
                synth_set, #values
                None,
                None,
            )
        _, p_R_model = density_estimator_trainer(reference_set)
        p_G_evaluated = np.exp(
                compute_log_p_x(p_G_model, torch.as_tensor(X_test).float().to(DEVICE))
                .cpu()
                .detach()
                .numpy()
            )
        p_R_evaluated = np.exp(
                compute_log_p_x(p_R_model, torch.as_tensor(X_test).float().to(DEVICE))
                .cpu()
                .detach()
                .numpy()
            )
            # KDE for pG
    elif density_estimator == "kde":
            density_gen = stats.gaussian_kde(synth_set.transpose(1, 0)) #values
            density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
            p_G_evaluated = density_gen(X_test.transpose(1, 0))
            p_R_evaluated = density_data(X_test.transpose(1, 0))

    p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

    return p_rel


