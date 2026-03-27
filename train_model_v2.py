#!/usr/bin/env python3
"""
train_angle_model.py

Expect CSV with headers:
x_1,y_1,z_1,x_2,y_2,z_2,angle_joint
"""

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Config
# ----------------------------
output_scale = 90

feature_xy = True
feature_z = True
feature_magnitude = True
feature_angle = True
feature_diff_xy = True
feature_diff_mag = True
feature_dot_cross = True

NUM_SENSORS = 1 # max available is 3 for MCP, 2 for PIP

NAME = "sim_pcb2_exp3_2_1"
TRAINING_NUM = 3

# CSV_PATH = "logs/Experiment 9/"+NAME+".csv" # replace with your path
#CSV_PATH = "C:/Users/laure/PycharmProjects/MT_joint_angle_sensing/classic_data/pcb2_mcp_exp1_2deg_2.csv"
CSV_PATH = "C:/Users/laure/PycharmProjects/MT_magnet/training_data/sim_pcb2_exp3_2_1.csv"
RANDOM_SEED = 729 # whatever you like
TEST_SIZE = 0.1  # 0.1 for both seems optimal
VAL_SIZE = 0.1  # proportion of full dataset used for validation
BATCH_SIZE = 256  # 64
LR = 2e-3  # 1e-3
N_EPOCHS = 200
WEIGHT_DECAY = 1e-5
PATIENCE = 9  # early stopping patience on validation loss
DROPOUT = 0.1
HIDDEN_DIMS = (256, 512, 256)

NOISE_FACTOR = 0.00

MODEL_SAVE_PATH = "models/" + NAME + "_" + str(TRAINING_NUM) + "/model.pth"
FULL_MODEL_SAVE_PATH = "models/" + NAME + "_" + str(TRAINING_NUM) + "/model_full.pth"
CONFIG_SAVE_PATH = "models/" + NAME + "_" + str(TRAINING_NUM) + "/model_config.pth"
SCALER_SAVE_PATH = "models/" + NAME + "_" + str(TRAINING_NUM) + "/scalar.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ----------------------------
# Dataset / DataLoader
# ----------------------------
class AngleDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------
# Model (MLP)
# ----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dims: Tuple[int, ...] = HIDDEN_DIMS,
                 dropout: float = DROPOUT, use_batchnorm: bool = True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.LeakyReLU(0.01,inplace=True))
            # layers.append(nn.GELU())
            # layers.append(nn.Sigmoid())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # output single angle value
        # layers.append(nn.Sigmoid()) maybe slight improvement
        self.net = nn.Sequential(*layers)

        # Store architecture config
        self.config = dict(
            input_dim=input_dim,
            hidden_dims=tuple(hidden_dims),
            dropout=dropout,
            use_batchnorm=use_batchnorm
        )

    def forward(self, x):
        return self.net(x)



# ----------------------------
# Utilities
# ----------------------------
def load_data(csv_path: str):

    df = pd.read_csv(csv_path)
    expected_cols = ['x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'angle_joint']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"CSV must contain columns: {expected_cols}. Found: {list(df.columns)}")

    # Base Magnitudes (XY plane)
    mag_1 = np.sqrt(df['x_1'] ** 2 + df['y_1'] ** 2)
    mag_2 = np.sqrt(df['x_2'] ** 2 + df['y_2'] ** 2)

    # Base Angles (XY plane)
    angle_xy_1 = np.arctan2(df['y_1'], df['x_1'])
    angle_xy_2 = np.arctan2(df['y_2'], df['x_2'])

    sin_1, cos_1 = np.sin(angle_xy_1), np.cos(angle_xy_1)
    sin_2, cos_2 = np.sin(angle_xy_2), np.cos(angle_xy_2)

    features = []

    if NUM_SENSORS == 2:
        # --- Standard Features ---
        if feature_xy:
            features.extend([df['x_1'].values, df['y_1'].values,
                             df['x_2'].values, df['y_2'].values])
        if feature_z:
            features.extend([df['z_1'].values, df['z_2'].values])
        if feature_magnitude:
            features.extend([mag_1, mag_2])
        if feature_angle:
            features.extend([sin_1, cos_1, sin_2, cos_2])

        # --- Engineered Relational Features ---
        if feature_diff_xy:
            features.extend([
                (df['x_2'] - df['x_1']).values,
                (df['y_2'] - df['y_1']).values
            ])
        if feature_diff_mag:
            features.extend([mag_2 - mag_1])
        if feature_dot_cross:
            # 2D Dot and Cross products in the XY plane
            features.extend([
                (df['x_1'] * df['x_2'] + df['y_1'] * df['y_2']).values,  # Dot product
                (df['x_1'] * df['y_2'] - df['y_1'] * df['x_2']).values  # Cross product
            ])
    else:
        # --- Single Sensor Fallback ---
        # Relational flags (diff, dot_cross) are inherently ignored here
        if feature_xy:
            features.extend([df['x_1'].values, df['y_1'].values])
        if feature_z:
            features.extend([df['z_1'].values])
        if feature_magnitude:
            features.extend([mag_1])
        if feature_angle:
            features.extend([sin_1, cos_1])

    X = np.column_stack(features)
    y = df['angle_joint'].values

    Y = y / output_scale  # normalize

    return X, Y


def load_data_triple(csv_path: str):
    df = pd.read_csv(csv_path)
    expected_cols = ['x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3', 'angle_joint']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"CSV must contain columns: {expected_cols}. Found: {list(df.columns)}")

    # Base Magnitudes
    mag_1 = np.sqrt(df['x_1'] ** 2 + df['y_1'] ** 2)
    mag_2 = np.sqrt(df['x_2'] ** 2 + df['y_2'] ** 2)
    mag_3 = np.sqrt(df['x_3'] ** 2 + df['y_3'] ** 2)

    # Base Angles
    angle_xy_1 = np.arctan2(df['y_1'], df['x_1'])
    angle_xy_2 = np.arctan2(df['y_2'], df['x_2'])
    angle_xy_3 = np.arctan2(df['y_3'], df['x_3'])

    sin_1, cos_1 = np.sin(angle_xy_1), np.cos(angle_xy_1)
    sin_2, cos_2 = np.sin(angle_xy_2), np.cos(angle_xy_2)
    sin_3, cos_3 = np.sin(angle_xy_3), np.cos(angle_xy_3)

    features = []

    # --- Original Features ---
    if feature_xy:
        features.extend([df['x_1'].values, df['y_1'].values,
                         df['x_2'].values, df['y_2'].values,
                         df['x_3'].values, df['y_3'].values])
    if feature_z:
        features.extend([df['z_1'].values, df['z_2'].values, df['z_3'].values])

    if feature_magnitude:
        features.extend([mag_1, mag_2, mag_3])

    if feature_angle:
        features.extend([sin_1, cos_1, sin_2, cos_2, sin_3, cos_3])

    # --- Engineered Features ---

    # 1. Spatial Differences in XY (Gradient of the field)
    if feature_diff_xy:
        features.extend([
            (df['x_2'] - df['x_1']).values, (df['y_2'] - df['y_1']).values,
            (df['x_3'] - df['x_2']).values, (df['y_3'] - df['y_2']).values,
            (df['x_3'] - df['x_1']).values, (df['y_3'] - df['y_1']).values
        ])

    # 2. Differences in Magnitude (Field drop-off profile)
    if feature_diff_mag:
        features.extend([
            mag_2 - mag_1,
            mag_3 - mag_2,
            mag_3 - mag_1
        ])

    # 3. Dot and Cross Products in XY (Relative orientation and alignment)
    if feature_dot_cross:
        # Between Sensor 1 and 2
        features.extend([
            (df['x_1'] * df['x_2'] + df['y_1'] * df['y_2']).values,  # Dot 1-2
            (df['x_1'] * df['y_2'] - df['y_1'] * df['x_2']).values  # Cross 1-2
        ])
        # Between Sensor 2 and 3
        features.extend([
            (df['x_2'] * df['x_3'] + df['y_2'] * df['y_3']).values,  # Dot 2-3
            (df['x_2'] * df['y_3'] - df['y_2'] * df['x_3']).values  # Cross 2-3
        ])
        # Between Sensor 1 and 3
        features.extend([
            (df['x_1'] * df['x_3'] + df['y_1'] * df['y_3']).values,  # Dot 1-3
            (df['x_1'] * df['y_3'] - df['y_1'] * df['x_3']).values  # Cross 1-3
        ])

    X = np.column_stack(features)
    y = df['angle_joint'].values / output_scale

    return X, y


def split_data(X, y, test_size=0.15, val_size=0.15, seed=42):
    # First split off test set
    X_rem, X_test, y_rem, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)
    # Now split remaining into train and val (val_size is relative to full dataset; convert)
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_rem, y_rem, test_size=val_relative, random_state=seed,
                                                      shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if NOISE_FACTOR != 0:
            noise = torch.randn_like(xb) * NOISE_FACTOR
            xb = xb + noise
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = xb.shape[0]
        total_loss += loss.item() * batch_size
        n += batch_size
    return total_loss / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            batch_size = xb.shape[0]
            total_loss += loss.item() * batch_size
            n += batch_size
    return total_loss / n


# ----------------------------
# Train function
# ----------------------------
def run_training(csv_path: str):
    print(f"Device: {DEVICE}")
    if NUM_SENSORS == 3:
        X, y = load_data_triple(csv_path)
    else:
        X, y = load_data(csv_path)
    print(f"Input Dims: {X.shape[1]}")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE,
                                                                seed=RANDOM_SEED)
    print("Sizes -> train: {}, val: {}, test: {}".format(len(X_train), len(X_val), len(X_test)))

    # Standardize inputs (fit on training data only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for inference later
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

    # Datasets & loaders
    train_ds = AngleDataset(X_train_scaled, y_train)
    val_ds = AngleDataset(X_val_scaled, y_val)
    test_ds = AngleDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model: input_dim=6
    # model = MLPRegressor(input_dim=X.shape[1], hidden_dims=(256, 128, 64, 32), dropout=0.2, use_batchnorm=True).to(DEVICE)
    model = MLPRegressor(input_dim=X.shape[1]).to(DEVICE)  # set params in class so that inference runs correctly
    torch.save(model.config, CONFIG_SAVE_PATH)
    criterion = nn.MSELoss()  # regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        train_loss_out = train_loss
        val_loss_out = val_loss
        train_loss_out *= output_scale * output_scale
        val_loss_out *= output_scale * output_scale

        print(f"Epoch {epoch:03d} | Train Loss (MSE) = {train_loss_out:.6f} | Val Loss (MSE) = {val_loss_out:.6f}")

        # Early stopping logic
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # save best model checkpoint

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch},
                       MODEL_SAVE_PATH)
            # keep a separate best copy too
            torch.save(model.state_dict(), MODEL_SAVE_PATH + ".best")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered (no improvement in {PATIENCE} epochs).")
                break

    print("Training finished. Loading best model for testing...")
    # load best model weights
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH + ".best", map_location=DEVICE))
    except Exception:
        print("Warning: unable to load best weights file, using last weights from training checkpoint.")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    model.eval()
    ys = []
    yps = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().numpy().reshape(-1)
            y_true = yb.numpy().reshape(-1)
            yps.append(preds)
            ys.append(y_true)

    y_pred = np.concatenate(yps)
    y_true = np.concatenate(ys)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae *= output_scale
    rmse *= output_scale
    r2 = r2_score(y_true, y_pred)

    print("Test results:")
    print(f"  MAE = {mae:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  R^2 = {r2:.6f}")

    # Save final model (state_dict)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model state_dict saved to {MODEL_SAVE_PATH}")
    torch.save(model, FULL_MODEL_SAVE_PATH)
    print(f"Full final model saved to {FULL_MODEL_SAVE_PATH}")

    return {
        "model": model,
        "scaler_path": SCALER_SAVE_PATH,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2}
    }


# ----------------------------
# If run as script
# ----------------------------
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"CSV file not found at {CSV_PATH}. Please edit CSV_PATH at top of script.")
    else:
        os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(CONFIG_SAVE_PATH), exist_ok=True)
        results = run_training(CSV_PATH)
