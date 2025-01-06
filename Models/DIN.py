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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



class AttentionUnit(nn.Module):
    """Attention mechanism to calculate interest scores."""
    def __init__(self, input_dim):
        super(AttentionUnit, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, query, keys):
        queries = query.unsqueeze(1).expand_as(keys)  # (batch_size, seq_len, input_dim)

        interactions = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        attention_scores = self.attention_layer(interactions).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Normalize scores
        return attention_weights


class DeepInterestNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, deep_hidden_units):
        super(DeepInterestNetwork, self).__init__()

        # Embedding layer
        self.embedding_layer = nn.Embedding(input_dim, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_layer.weight)
        
        self.attention = AttentionUnit(embedding_dim)

        # print(f"Deep Hidden Units: {deep_hidden_units[0]}")
        # print(f"Embedding Dim: {embedding_dim}")
        assert deep_hidden_units[0] == embedding_dim, "The first hidden layer must match the embedding_dim."

        # Deep network
        deep_layers = []
        for i in range(len(deep_hidden_units) - 1):
            deep_layers.append(nn.Linear(deep_hidden_units[i], deep_hidden_units[i + 1]))
            deep_layers.append(nn.ReLU())
        self.deep_network = nn.Sequential(*deep_layers)

        # Output layer
        self.output_layer = nn.Linear(deep_hidden_units[-1], 1)

    def forward(self, query, keys):
        query_embedding = self.embedding_layer(query)
        keys_embedding = self.embedding_layer(keys)

        # print(f"Query Shape: {query_embedding.shape}, Keys Shape: {keys_embedding.shape}")

        attention_weights = self.attention(query_embedding, keys_embedding)
        # print(f"Attention Weights Shape: {attention_weights.shape}")

        interest_representation = torch.sum(attention_weights.unsqueeze(-1) * keys_embedding, dim=1)
        # print(f"Interest Representation Shape: {interest_representation.shape}")

        deep_out = self.deep_network(interest_representation)
        # print(f"Deep Out Shape: {deep_out.shape}")

        output = torch.sigmoid(self.output_layer(deep_out))
        # print(f"Output Shape: {output.shape}")
        
        return output


def train_and_evaluate(X_train, y_train, X_val, y_val, item_col, seq_cols, embedding_dim, deep_hidden_units, epochs, lr, save_dir, grp_name, batch_size, device, session_name):
    input_dim = X_train[item_col].nunique() + 1

    model = DeepInterestNetwork(input_dim, embedding_dim, deep_hidden_units).to(device)
    print(model)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # debug
    # print(f"Unique values in item_col (train): {X_train[item_col].nunique()}")
    # print(f"Expected input_dim: {input_dim}")
    # print(f"Max index in item_col (train): {X_train[item_col].max()}")
    # print(f"Max index in seq_cols (train): {X_train[seq_cols].max().max()}")
    # Clamp indices to ensure they are within the embedding range
    X_train[seq_cols] = X_train[seq_cols].clip(0, input_dim - 1)
    X_train[item_col] = X_train[item_col].clip(0, input_dim - 1)
    X_val[seq_cols] = X_val[seq_cols].clip(0, input_dim - 1)
    X_val[item_col] = X_val[item_col].clip(0, input_dim - 1)


    train_dataset = TensorDataset(
        torch.tensor(X_train[item_col].values, dtype=torch.long),
        torch.tensor(X_train[seq_cols].values, dtype=torch.long),
        torch.tensor(y_train.values, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val[item_col].values, dtype=torch.long),
        torch.tensor(X_val[seq_cols].values, dtype=torch.long),
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
        for batch_item, batch_seq, batch_y in train_loader:
            batch_item, batch_seq, batch_y = batch_item.to(device), batch_seq.to(device), batch_y.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(batch_item, batch_seq)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)

        model.eval()
        epoch_val_loss = 0
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for batch_item, batch_seq, batch_y in val_loader:
                batch_item, batch_seq, batch_y = batch_item.to(device), batch_seq.to(device), batch_y.to(device).view(-1, 1)
                outputs = model(batch_item, batch_seq)
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

    utils.plot_metrics(train_losses, val_losses, auc_scores, precision_scores, recall_scores, save_dir, grp_name, session_name)

    model_save_path = os.path.join(save_dir, session_name, f"{grp_name}_din.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Deep Interest Network (DIN) Model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--save_dir", type=str, default="./Results", help="Directory to save model checkpoints and results")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=str, default="64,32", help="Comma-separated list of hidden units for the deep network")
    parser.add_argument("--item_col", type=str, default = "exposure", help="Column name for the target item")
    parser.add_argument("--seq_cols", type=str, default= "f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11", help="Comma-separated column names for the sequence")
    parser.add_argument("--treatment", action="store_true", help="Separate treatment/control groups")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA if available")
    args = parser.parse_args()

    deep_hidden_unit = list(map(int, args.hidden_units.split(',')))
    seq_cols = args.seq_cols.split(',')
    embedding_dim = args.hidden_units[0]

    utils.set_seed(42)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_data = pd.read_csv("./Data/processed/train.csv")
    val_data = pd.read_csv("./Data/processed/val.csv")
    print("Data loaded, start training...")

    stamp = utils.get_timestamp()
    exp_info = f"{stamp}_batch{args.batch}_epoch{args.epoch}_lr{args.lr}_emb{embedding_dim}"
    session_name = f"{exp_info}_{args.model_name}"
    

    def train_model_for_group(train_group, val_group, group_name):
        print(f"Training model for {group_name} group...")
        X_train, y_train = utils.split_features_and_target(train_group)
        X_val, y_val = utils.split_features_and_target(val_group)
        return train_and_evaluate(
            X_train, y_train, X_val, y_val, args.item_col, seq_cols,
            embedding_dim, deep_hidden_unit, args.epoch, args.lr,
            args.save_dir, group_name, args.batch, device, session_name
        )

    if not args.treatment:
        X_train, y_train = utils.split_features_and_target(train_data)
        X_val, y_val = utils.split_features_and_target(val_data)
        train_and_evaluate(
            X_train, y_train, X_val, y_val, args.item_col, seq_cols,
            embedding_dim, deep_hidden_unit, args.epoch, args.lr,
            args.save_dir, "combined", args.batch, device, session_name
        )
    else:
        train_treatment = utils.filter_data_by_treatment(train_data, 1)
        val_treatment = utils.filter_data_by_treatment(val_data, 1)
        train_model_for_group(train_treatment, val_treatment, "treatment")

        train_control = utils.filter_data_by_treatment(train_data, 0)
        val_control = utils.filter_data_by_treatment(val_data, 0)
        train_model_for_group(train_control, val_control, "control")
