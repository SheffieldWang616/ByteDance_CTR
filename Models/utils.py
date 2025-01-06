import os
import random
import numpy as np
import datetime
import torch
import matplotlib.pyplot as plt


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data splitting helpers
def split_features_and_target(data):
    """
    Splits the data into features (X) and target (y).
    """
    X = data.drop(columns=["treatment", "visit", "conversion"])
    y = data["conversion"]
    return X, y


def filter_data_by_treatment(data, treatment_value):
    """
    Filters the data based on treatment value (0 or 1).
    """
    return data[data["treatment"] == treatment_value]

def normalize_features(data):
    return (data - data.mean()) / data.std()

def get_field_indices(data):
    """
    Returns the field indices for each feature in the data. For the Cretio dataset, there are 12 input features
    ,a treatment col, a visit col, a conversion col and a exposure col.
    For this dataset, we will use all the 12 input features and one exposure column as fields. 
    Treatment was dropped, visit holds critical prediction information so it's dropped, conversion was applied as the output label.
    So in total, we have (12+4-3) = 13 fields.
    """
    field_indices = []
    return list(range(13))

# Plot metrics and save the plots
def plot_metrics(train_losses, val_losses, auc_scores, precision_scores, recall_scores, save_dir, grp_name, model_name):
    """
    Plot and save the metrics.

    """
    plot_dir = os.path.join(save_dir, model_name)
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(auc_scores, label="Validation AUC", color="orange")
    plt.title("AUC over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(precision_scores, label="Precision", color="green")
    plt.title("Precision over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(recall_scores, label="Recall", color="red")
    plt.title("Recall over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()

    plt.tight_layout()

    # Save plot
    if grp_name == "treatment":
        plt_path = os.path.join(plot_dir, "metrics_treatment.png")
    elif grp_name == "control":
        plt_path = os.path.join(plot_dir, "metrics_control.png")
    else:
        plt_path = os.path.join(plot_dir, "metrics.png")

    plt.savefig(plt_path)
    plt.close()
    print(f"Metrics plot saved at {plt_path}")
