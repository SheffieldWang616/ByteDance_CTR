import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from torch.utils.tensorboard import SummaryWriter

# Import utilities
import utils

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


from torch.utils.data import DataLoader, TensorDataset

def train_and_evaluate(X_train, y_train, X_val, y_val, epochs, lr, reg, save_dir, grp_name=None, model_name=None, batch_size=1024):
    input_dim = X_train.shape[1]
    model = LogisticRegressionModel(input_dim).to(device)  # Move model to the correct device
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=reg)

    log_dir = os.path.join(save_dir, "LR_event", model_name, grp_name)
    writer = SummaryWriter(log_dir)

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32)
    )

    # DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    auc_scores = []
    precision_scores = []
    recall_scores = []

    for epoch in range(epochs):
        print(f"******************** Epoch [{epoch+1}/{epochs}] ********************")

        # Training
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            # Move batch to the same device as the model
            batch_X = batch_X.to(device)
            batch_y = batch_y.view(-1, 1).to(device)

            optimizer.zero_grad()
            train_outputs = model(batch_X)
            train_loss = criterion(train_outputs, batch_y)
            train_loss.backward()
            optimizer.step()
            epoch_train_loss += train_loss.item()

        epoch_train_loss /= len(train_loader)

        # Validation
        model.eval()
        epoch_val_loss = 0
        all_val_outputs = []
        all_val_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Move batch to the same device as the model
                batch_X = batch_X.to(device)
                batch_y = batch_y.view(-1, 1).to(device)

                val_outputs = model(batch_X)
                val_loss = criterion(val_outputs, batch_y)
                epoch_val_loss += val_loss.item()

                # Collect outputs and targets for metric calculation
                all_val_outputs.extend(val_outputs.cpu().numpy())  # Move to CPU
                all_val_targets.extend(batch_y.cpu().numpy())      # Move to CPU

        epoch_val_loss /= len(val_loader)

        # Calculate metrics
        all_val_outputs = np.array(all_val_outputs)
        all_val_targets = np.array(all_val_targets)
        val_preds = (all_val_outputs > 0.5).astype(float)

        auc = roc_auc_score(all_val_targets, all_val_outputs)
        precision = precision_score(all_val_targets, val_preds, zero_division=0)
        recall = recall_score(all_val_targets, val_preds, zero_division=0)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        auc_scores.append(auc)
        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        writer.add_scalar("Loss_Train", epoch_train_loss, epoch)
        writer.add_scalar("Loss_Validation", epoch_val_loss, epoch)
        writer.add_scalar("AUC_Validation", auc, epoch)
        writer.add_scalar("Precision_Validation", precision, epoch)
        writer.add_scalar("Recall_Validation", recall, epoch)

    writer.close()

    # Plot metrics
    utils.plot_metrics(train_losses, val_losses, auc_scores, precision_scores, recall_scores, save_dir, grp_name, model_name)
    model_save_path = os.path.join(save_dir, model_name, f"{grp_name}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Logistic Regression Model")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--save_dir", type=str, default="./Results", help="Directory to save model checkpoints and results")
    parser.add_argument("--batch", type=int, default=1024, help="Input batch size for batched training")
    parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument("--reg", type =float,default = 1e-4, help="Regularization Strength")
    parser.add_argument("--treatment", action="store_true", help="Whether to include treatment as a feature")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA training")
    args = parser.parse_args()

    
    utils.set_seed(42)  # Ensure reproducibility

    # Check device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Data...")
    train_data = pd.read_csv("./Data/processed/train.csv")
    val_data = pd.read_csv("./Data/processed/val.csv")
    print("Data Loaded, start training...")

    stamp = utils.get_timestamp()
    exp_info = f"{stamp}_batch{args.batch}_epoch{args.epoch}_lr{args.lr}_reg{args.reg}"
    session_name = f"{exp_info}_{args.model_name}"
    

    def train_model_for_group(train_group, val_group, group_name):
        print(f"Training model for {group_name} group...")
        X_train, y_train = utils.split_features_and_target(train_group)
        X_val, y_val = utils.split_features_and_target(val_group)
        return train_and_evaluate(X_train, y_train, X_val, y_val, epochs=args.epoch, lr=args.lr, reg = args.reg, save_dir=args.save_dir, grp_name=group_name, model_name=session_name, batch_size=args.batch)

    if not args.treatment:
        # Combined group training
        X_train, y_train = utils.split_features_and_target(train_data)
        X_val, y_val = utils.split_features_and_target(val_data)
        train_and_evaluate(X_train, y_train, X_val, y_val, epochs=args.epoch, lr=args.lr, reg = args.reg, save_dir=args.save_dir, grp_name="combined", model_name=session_name, batch_size=args.batch)
    else:
        # Treatment group
        train_treatment = utils.filter_data_by_treatment(train_data, 1)
        val_treatment = utils.filter_data_by_treatment(val_data, 1)
        train_model_for_group(train_treatment, val_treatment, "treatment")

        # Control group
        train_control = utils.filter_data_by_treatment(train_data, 0)
        val_control = utils.filter_data_by_treatment(val_data, 0)
        train_model_for_group(train_control, val_control, "control")
