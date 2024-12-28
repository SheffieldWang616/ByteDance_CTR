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


class WideAndDeep(nn.Module):
    def __init__(self, input_dim_wide, input_dim_deep, deep_hidden_units):
        super(WideAndDeep, self).__init__()
        # Wide part: Linear layer
        self.wide = nn.Linear(input_dim_wide, 1)
        
        # Deep part: Feedforward neural network
        layers = []
        for i in range(len(deep_hidden_units) - 1):
            layers.append(nn.Linear(deep_hidden_units[i], deep_hidden_units[i + 1]))
            layers.append(nn.ReLU())
        self.deep = nn.Sequential(*layers)
        
        # Final output layer (combiner)
        self.output = nn.Linear(deep_hidden_units[-1] + 1, 1)  # Wide + Deep combined

    def forward(self, wide_input, deep_input):
        wide_out = self.wide(wide_input)  # Wide part
        deep_out = self.deep(deep_input)  # Deep part
        combined = torch.cat([wide_out, deep_out], dim=1)  # Combine outputs
        return torch.sigmoid(self.output(combined))


def train_and_evaluate(X_train, y_train, X_val, y_val, epochs, lr, save_dir, grp_name, batch_size, device, session_name, input_dim):
    
    # define hidden units for deep part
    deep_hidden_units = [input_dim, 64, 32]
    model = WideAndDeep(input_dim_wide=input_dim, input_dim_deep=input_dim, deep_hidden_units=deep_hidden_units).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(X_train.values, dtype=torch.float32),  # Deep input (can duplicate wide input)
        torch.tensor(y_train.values, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(X_val.values, dtype=torch.float32),  # Deep input
        torch.tensor(y_val.values, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses, auc_scores, precision_scores, recall_scores = [], [], [], [], []

    for epoch in range(epochs):
        print(f"**************** Epoch [{epoch+1}/{epochs}] ****************")
        model.train()
        epoch_train_loss = 0
        for wide_input, deep_input, target in train_loader:
            wide_input, deep_input, target = wide_input.to(device), deep_input.to(device), target.to(device).view(-1,1)
            optimizer.zero_grad()
            outputs = model(wide_input, deep_input)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)

        # Validation
        model.eval()
        epoch_val_loss = 0
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for wide_input, deep_input, target in val_loader:
                wide_input, deep_input, target = wide_input.to(device), deep_input.to(device), target.to(device).view(-1,1)
                outputs = model(wide_input, deep_input)
                loss = criterion(outputs, target)
                epoch_val_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
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

    model_save_path = os.path.join(save_dir, session_name, f"{grp_name}_wide_deep.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Wide & Deep Learning Model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--save_dir", type=str, default="./Results", help="Directory to save model checkpoints and results")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
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

    exp_info = f"batch{args.batch}_epoch{args.epoch}_lr{args.lr}"
    session_name = f"{args.model_name}_{exp_info}"

    def train_model_for_group(train_group, val_group, group_name):
        print(f"Training model for {group_name} group...")
        X_train, y_train = utils.split_features_and_target(train_group)
        X_val, y_val = utils.split_features_and_target(val_group)
        input_dim = X_train.shape[1]
        return train_and_evaluate(X_train, y_train, X_val, y_val, args.epoch, args.lr, 
                                  args.save_dir, group_name, args.batch, device, session_name, input_dim)

    if not args.treatment:
        X_train, y_train = utils.split_features_and_target(train_data)
        X_val, y_val = utils.split_features_and_target(val_data)
        input_dim = X_train.shape[1]
        train_and_evaluate(X_train, y_train, X_val, y_val, args.epoch, args.lr, 
                           args.save_dir, "combined", args.batch, device, session_name, input_dim)
    else:
        train_treatment = utils.filter_data_by_treatment(train_data, 1)
        val_treatment = utils.filter_data_by_treatment(val_data, 1)
        train_model_for_group(train_treatment, val_treatment, "treatment")

        train_control = utils.filter_data_by_treatment(train_data, 0)
        val_control = utils.filter_data_by_treatment(val_data, 0)
        train_model_for_group(train_control, val_control, "control")
