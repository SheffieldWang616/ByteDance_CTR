import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

from torch.utils.tensorboard import SummaryWriter

# Import utilities
import utils


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.cross_weights = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.cross_biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for i in range(self.num_layers):
            xw = self.cross_weights[i](x)  # (batch_size, 1)
            x = x0 * xw + self.cross_biases[i] + x  # (batch_size, input_dim)
        return x



class DeepCrossNetwork(nn.Module):
    def __init__(self, input_dim, deep_hidden_units, cross_num_layers):
        super(DeepCrossNetwork, self).__init__()
        # Cross Network
        self.cross_network = CrossNetwork(input_dim, cross_num_layers)

        # Deep Network
        deep_layers = []
        for i in range(len(deep_hidden_units) - 1):
            deep_layers.append(nn.Linear(deep_hidden_units[i], deep_hidden_units[i + 1]))
            deep_layers.append(nn.ReLU())
        self.deep_network = nn.Sequential(*deep_layers)

        # Final output layer
        combined_dim = input_dim + deep_hidden_units[-1]  # Cross + Deep
        self.output_layer = nn.Linear(combined_dim, 1)

    def forward(self, x):
        # Cross Network
        cross_out = self.cross_network(x)

        # Deep Network
        deep_out = self.deep_network(x)

        # Combine outputs
        combined_out = torch.cat([cross_out, deep_out], dim=1)

        # Final prediction
        output = torch.sigmoid(self.output_layer(combined_out))
        return output


def train_and_evaluate(X_train, y_train, X_val, y_val, deep_hidden_units, cross_num_layers, epochs, lr, save_dir, grp_name, batch_size, device, session_name):
    input_dim = X_train.shape[1]

    # Initialize deep hidden units
    deep_hidden_units = [input_dim] + deep_hidden_unit

    # Initialize model
    model = DeepCrossNetwork(input_dim, deep_hidden_units, cross_num_layers).to(device)
    print(model)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.reg)

    log_dir = os.path.join(save_dir, "DCN_event", session_name, grp_name)
    writer = SummaryWriter(log_dir)

    # Prepare data loaders
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
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss = epoch_train_loss +  loss.item()
        epoch_train_loss /= len(train_loader)

        # Validation
        model.eval()
        epoch_val_loss = 0
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        epoch_val_loss /= len(val_loader)

        # Calculate metrics
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

        writer.add_scalar("Loss_Train", epoch_train_loss, epoch)
        writer.add_scalar("Loss_Validation", epoch_val_loss, epoch)
        writer.add_scalar("AUC_Validation", auc, epoch)
        writer.add_scalar("Precision_Validation", precision, epoch)
        writer.add_scalar("Recall_Validation", recall, epoch)

    writer.close()

    # Plot metrics
    utils.plot_metrics(train_losses, val_losses, auc_scores, precision_scores, recall_scores, save_dir, grp_name, session_name)

    # Save model
    model_save_path = os.path.join(save_dir, session_name, f"{grp_name}_dcn.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Deep Cross Network (DCN) Model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--save_dir", type=str, default="./Results", help="Directory to save model checkpoints and results")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--reg", type =float,default = 1e-4, help="Regularization Strength")
    parser.add_argument("--hidden_units", type=str, default="512,64,32", help="Comma-separated list of hidden units for the deep network")
    parser.add_argument("--cross_layers", type=int, default=3, help="Number of cross layers in the Cross Network")
    parser.add_argument("--treatment", action="store_true", help="Separate treatment/control groups")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA if available")
    args = parser.parse_args()

    # Parse deep hidden units
    deep_hidden_unit = list(map(int, args.hidden_units.split(',')))

    utils.set_seed(42)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_data_raw = pd.read_csv("./Data/processed/train.csv")
    val_data_raw = pd.read_csv("./Data/processed/val.csv")
    
    scaler = MinMaxScaler()
    train_data = pd.DataFrame(scaler.fit_transform(train_data_raw), columns=train_data_raw.columns)
    val_data = pd.DataFrame(scaler.transform(val_data_raw), columns=val_data_raw.columns)
    print("Data loaded, start training...")

    # Prepare session name
    stamp = utils.get_timestamp()
    exp_info = f"{stamp}_batch{args.batch}_epoch{args.epoch}_lr{args.lr}_reg{args.reg}_cross{args.cross_layers}"
    session_name = f"{exp_info}_{args.model_name}"

    def train_model_for_group(train_group, val_group, group_name):
        print(f"Training model for {group_name} group...")
        X_train, y_train = utils.split_features_and_target(train_group)
        X_val, y_val = utils.split_features_and_target(val_group)
        return train_and_evaluate(X_train, y_train, X_val, y_val, deep_hidden_unit, args.cross_layers, args.epoch, args.lr,
                                  args.save_dir, group_name, args.batch, device, session_name)

    if not args.treatment:
        X_train, y_train = utils.split_features_and_target(train_data)
        X_val, y_val = utils.split_features_and_target(val_data)
        train_and_evaluate(X_train, y_train, X_val, y_val, deep_hidden_unit, args.cross_layers, args.epoch, args.lr,
                           args.save_dir, "combined", args.batch, device, session_name)
    else:
        train_treatment = utils.filter_data_by_treatment(train_data, 1)
        val_treatment = utils.filter_data_by_treatment(val_data, 1)
        train_model_for_group(train_treatment, val_treatment, "treatment")

        train_control = utils.filter_data_by_treatment(train_data, 0)
        val_control = utils.filter_data_by_treatment(val_data, 0)
        train_model_for_group(train_control, val_control, "control")
