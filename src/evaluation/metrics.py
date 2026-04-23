from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluationMetrics:
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob):
        metrics = {
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_true, y_prob),
            'PR-AUC': average_precision_score(y_true, y_prob)
        }
        return metrics

    @staticmethod
    def print_metrics(metrics):
        print("--- Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("--------------------------")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fraud'], 
                    yticklabels=['Normal', 'Fraud'])
        plt.title(title)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        return plt
