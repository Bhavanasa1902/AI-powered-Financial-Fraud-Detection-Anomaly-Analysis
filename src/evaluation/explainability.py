import shap
import matplotlib.pyplot as plt
import pandas as pd

class ExplainabilityModule:
    def __init__(self, model):
        """
        Currently built primarily around tree models using TreeExplainer.
        Can be extended to DeepExplainer for Neural Networks.
        """
        self.model = model
        self.explainer = None

    def initialize_explainer(self, background_data=None):
        """
        Initializes the SHAP TreeExplainer.
        """
        # For XGBoost or Random Forest
        self.explainer = shap.TreeExplainer(self.model)
        print("Initialized SHAP TreeExplainer.")

    def get_shap_values(self, X):
        if self.explainer is None:
            self.initialize_explainer()
        return self.explainer.shap_values(X)

    def plot_summary(self, X_df, save_path=None):
        """
        Plots global feature importance based on SHAP values across a dataset.
        """
        shap_values = self.get_shap_values(X_df)
        plt.figure()
        shap.summary_plot(shap_values, X_df, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        return plt

    def explain_instance(self, instance_df, max_display=10, save_path=None):
        """
        Provides local explanation (waterfall) for a specific flagged transaction.
        """
        if self.explainer is None:
            self.initialize_explainer()
            
        shap_values = self.explainer(instance_df)
        
        if len(shap_values.shape) == 3:
            shap_val_instance = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            shap_val_instance = shap_values[0]
        else:
            shap_val_instance = shap_values
            
        plt.figure()
        shap.plots.waterfall(shap_val_instance, max_display=max_display, show=False)
        
        if save_path:
            plt.gcf().savefig(save_path, bbox_inches='tight')
            plt.close()
        return plt
