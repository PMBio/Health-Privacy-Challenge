import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.metrics import (accuracy_score, roc_curve, f1_score,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, auc)



def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


class BaseMIAModel(ABC):
    def __init__(self, 
                config: Dict[str, Any], 
                synthetic_file: str,
                membership_test_file: str,
                membership_lbl_file: str,
                mia_experiment_name:str,
                reference_file:str = None):
        self.config = config
        self.home_dir = config["dir_list"]["home"]
        #self.results_dir = config["dir_list"]["mia_files"]
        self.generator_model = self.config["generator_config"]["name"]
        self.experiment_name = self.config["generator_config"]["experiment_name"]
        self.attack_model =  self.config["attack_model"]
        self.dataset_config = config["dataset_config"]
        self.dataset_name = self.dataset_config["name"]
        self.membership_label_col = self.dataset_config["membership_label_col"]
        self.synthetic_file = synthetic_file
        self.reference_file = reference_file
        self.membership_test_file = membership_test_file
        self.membership_lbl_file = membership_lbl_file
        self.results_save_dir = os.path.join(
                                        self.home_dir, 
                                        config["dir_list"]["mia_files"], 
                                        self.dataset_name, 
                                        self.attack_model,
                                        self.generator_model,
                                        self.experiment_name,
                                        mia_experiment_name
                                    )
        check_folder(self.results_save_dir)
        config_key = f"{self.attack_model}_config"

        if config_key in config:
            self.mia_config = config[config_key]
        else:
            raise ValueError(f"No config found for attack model: {self.attack_model}")


    @abstractmethod
    def run_attack(self) -> Dict[str, np.ndarray]:
        pass


    def evaluate_attack(self, 
                        scores: Dict[str, np.ndarray], 
                        labels: np.ndarray, 
                        file_name: str):

        eval_results = {}
        if isinstance(scores, np.ndarray):
            scores = {self.attack_model: scores}

        for method_name, score in scores.items():
            # compute metrics for each method
            #acc, fpr, tpr, threshold, auc, ap = self._compute_metrics(score, labels)
            print(method_name)
            (acc_median, acc_best, fpr, tpr, threshold, auc, ap, pr_auc,
            f1_median, f1_best, tpr_at_fpr_001, tpr_at_fpr_01, 
            precision_5pct, k, tp_count) = self._compute_metrics(score, labels)


            
            eval_results[method_name] = {
                #"accuracy": acc,
                #"aucroc": auc,
                #"average_precision": ap,
                "accuracy_median": acc_median,
                "accuracy_best": acc_best,
                "aucroc": auc,
                "average_precision": ap,
                "pr_auc": pr_auc,
                "f1_median": f1_median,
                "f1_best": f1_best,
                "tpr_at_fpr_01": tpr_at_fpr_01,
                "tpr_at_fpr_001": tpr_at_fpr_001,
                "precision@5pt": precision_5pct,
                "tp_count": tp_count
                #"fpr": fpr.tolist(),
                #"tpr": tpr.tolist(),
                #"threshold": threshold.tolist() 
            }

            # plot ROC curve for each method
            self._plot_roc_curve(method_name, fpr, tpr, auc, ap)

        self._save_eval_results(eval_results, file_name)



    def save_predictions(self, scores: Dict[str, np.ndarray]):
        for key, value in scores.items():
            df = pd.DataFrame(data = value, columns= [self.membership_label_col])
            df.to_csv(os.path.join(
                self.results_save_dir, 
                f"{key}_predictions.csv"), index=False)
            

    def load_predictions(self) -> Dict[str, np.ndarray]:
        predictions = {}
        for filename in os.listdir(self.results_save_dir):
            if filename.endswith("_predictions.csv"):
                key = filename.replace("_predictions.csv", "")
                df = pd.read_csv(os.path.join(self.results_save_dir, filename))
                predictions[key] = df.to_numpy()
        return predictions


    @staticmethod
    def compute_precision_top_percent(y_true, scores, top_percent=5):
        """
        Compute Precision@top_percent.
        - y_true: 1=member, 0=non-member
        - scores: continuous attack scores (higher -> more likely member)
        - top_percent: percentage of candidates to consider
        Returns: precision (fraction of true members in top candidates)
        """
        n_candidates = len(scores)
        k = max(1, int(np.ceil(n_candidates * top_percent / 100)))  # ensure at least 1
        top_idx = np.argsort(scores)[-k:][::-1]  # indices of top-k scores
        top_members = y_true[top_idx].sum()
        precision = top_members / k
        return precision, k, top_members


    @staticmethod
    def _compute_metrics(
                y_scores: np.ndarray, 
                y_true: np.ndarray, 
                sample_weight: Optional[np.ndarray] = None):
        y_pred_median = y_scores > np.median(y_scores)

        # compute F1 for multiple thresholds
        thresholds = np.sort(np.unique(y_scores))
        if len(thresholds) >= 2:
            f1_scores = [f1_score(y_true, y_scores > t, sample_weight=sample_weight) 
                     for t in thresholds]
            best_threshold = thresholds[np.argmax(f1_scores)]
            y_pred_best = y_scores > best_threshold
        else:
            y_pred_best = y_scores > np.median(y_scores)
            

        ## compare the accuracy and f1 computed with best_threshold vs median 
        acc_median = accuracy_score(y_true, y_pred_median, sample_weight=sample_weight)
        acc_best = accuracy_score(y_true, y_pred_best, sample_weight=sample_weight)

        auc_sc = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)
        ap = average_precision_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        # compute PR-AUC
        pr_auc = auc(recall, precision)

        fpr, tpr, threshold = roc_curve(y_true, y_scores, pos_label=1)

        # Get TPR at specific FPR thresholds
        tpr_at_fpr_001 = tpr[(fpr >= 0.01).argmax()]
        tpr_at_fpr_01 = tpr[(fpr >= 0.1).argmax()]

        idx = (fpr >= 0.01).argmax()
        threshold_at_fpr = threshold[idx]
        print("Threshold used for FPR=0.01:", threshold_at_fpr)

        pred_labels = (y_scores >= threshold_at_fpr).astype(int)
        tp_indices = np.where((pred_labels == 1) & (y_true == 1))[0]
        fp_indices = np.where((pred_labels == 1) & (y_true == 0))[0]
        print("TP count:", len(tp_indices), "FP count:", len(fp_indices))

        precision_5pct, k, tp_count = BaseMIAModel.compute_precision_top_percent(
                                            y_true, 
                                            y_scores, 
                                            top_percent=5)


        f1_median = f1_score(y_true, y_pred_median, sample_weight=sample_weight) 
        f1_best= f1_score(y_true, y_pred_best, sample_weight=sample_weight) 

        #return acc, fpr, tpr, threshold, auc, ap
        return (acc_median, acc_best, fpr, tpr, threshold, 
                auc_sc, ap, pr_auc, f1_median, f1_best, tpr_at_fpr_001, tpr_at_fpr_01,
                precision_5pct, k, tp_count)
    


    def _plot_roc_curve(self, name:str, fpr: np.ndarray, tpr: np.ndarray, auc: float, ap: float):
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} attack, auc={auc:.3f}, ap={ap:.3f}')
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(os.path.join(self.results_save_dir, f'roc_plot_{name}.png'))
        plt.close()

    

    def _save_eval_results(self, 
                          eval_results: Dict[str, Dict[str, float]], 
                          file_name: str):
        results_list = []
        for method, metrics in eval_results.items():
            metrics["method"] = method
            results_list.append(metrics)
        
        results_df = pd.DataFrame(results_list)
        save_path = os.path.join(self.results_save_dir, file_name)
        results_df.to_csv(save_path, index=False)
        print(f"Evaluation results saved to {save_path}")