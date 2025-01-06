import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Import utilities
import utils


class FieldAwareFactorizationMachineModel(nn.Module):
    def __init__(self, num_fields, input_dim, k):
        super(FieldAwareFactorizationMachineModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.v = nn.Parameter(torch.randn(num_fields, input_dim, k) * 0.01)  # Field-aware latent factors

    def forward(self, x, field_indices):
        linear_part = self.linear(x)

        # Compute interaction part for field-aware factorization
        interaction_part = 0

        # print("xshape is",x.shape[1])

        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                vi = self.v[field_indices[i], i]
                vj = self.v[field_indices[j], j]
                interaction_part += (torch.sum(vi * vj) * x[:, i] * x[:, j])

        interaction_part = interaction_part.unsqueeze(1)  # Adjust dimensions
        return torch.sigmoid(linear_part + interaction_part)


def train_and_evaluate(X_train, y_train, X_val, y_val, field_indices, epochs, lr, save_dir, grp_name, batch_size, k, device, session_name):
    input_dim = X_train.shape[1]
    num_fields = len(np.unique(field_indices))  # Number of unique fields
    model = FieldAwareFactorizationMachineModel(num_fields, input_dim, k).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)

    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []
    auc_scores, precision_scores, recall_scores = [], [], []

    for epoch in range(epochs):
        print(f"**************** Epoch [{epoch+1}/{epochs}] ****************")
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(batch_X, field_indices)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)

        model.eval()
        epoch_val_loss = 0
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
                outputs = model(batch_X, field_indices)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        epoch_val_loss /= len(val_loader)

        all_outputs, all_targets = np.array(all_outputs), np.array(all_targets)
        val_preds = (all_outputs > 0.5).astype(float)
        auc = roc_auc_score(all_targets, all_outputs)
        precision = precision_score(all_targets, val_preds, zero_division=0)
        recall = recall_score(all_targets, val_preds, zero_division=0)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        auc_scores.append(auc)
        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Pass model_name to plot_metrics
    utils.plot_metrics(train_losses, val_losses, auc_scores, precision_scores, recall_scores, save_dir, grp_name, session_name)

    model_save_path = os.path.join(save_dir, session_name, f"{grp_name}_ffm.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Field-aware Factorization Machine Model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--save_dir", type=str, default="./Results", help="Directory to save model checkpoints and results")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--k", type=int, default=10, help="Latent factor size for FFM interactions")
    parser.add_argument("--treatment", action="store_true", help="Separate treatment/control groups")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA if available")
    args = parser.parse_args()

    utils.set_seed(42)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_data = pd.read_csv("./Data/processed/train.csv")
    val_data = pd.read_csv("./Data/processed/val.csv")
    print("Data loaded, start training...")

    exp_info = f"{utils.get_timestamp}_batch{args.batch}_epoch{args.epoch}_lr{args.lr}_k{args.k}"
    session_name = f"{exp_info}_{args.model_name}"

    # Assign fields manually for this example
    field_indices = utils.get_field_indices(train_data)

    def train_model_for_group(train_group, val_group, group_name):
        print(f"Training model for {group_name} group...")
        X_train, y_train = utils.split_features_and_target(train_group)
        X_val, y_val = utils.split_features_and_target(val_group)
        return train_and_evaluate(X_train, y_train, X_val, y_val, field_indices, args.epoch, args.lr,
                                      args.save_dir, group_name, args.batch, args.k, device, session_name)

    if not args.treatment:
        X_train, y_train = utils.split_features_and_target(train_data)
        X_val, y_val = utils.split_features_and_target(val_data)
        train_and_evaluate(X_train, y_train, X_val, y_val, field_indices, args.epoch, args.lr,
                               args.save_dir, "combined", args.batch, args.k, device, session_name)
    else:
        train_treatment = utils.filter_data_by_treatment(train_data, 1)
        val_treatment = utils.filter_data_by_treatment(val_data, 1)
        train_model_for_group(train_treatment, val_treatment, "treatment")

        train_control = utils.filter_data_by_treatment(train_data, 0)
        val_control = utils.filter_data_by_treatment(val_data, 0)
        train_model_for_group(train_control, val_control, "control")
