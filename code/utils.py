# =======================
# Imported libraries
# =======================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import os
# =======================


# =======================
# Data loading functions
# =======================
def load_data(
        path_data, path_gt, test_size=0.2, 
        val_size=0, pca_size=0, random_state=42):
    """Load and preprocess hyperspectral data.
    
    Parameters
    ----------
    path_data : str
        Path to hyperspectral data file (.npy)
    path_gt : str
        Path to ground truth labels file (.npy)
    test_size : float, optional
        Proportion of test data (default: 0.2)
    val_size : float, optional
        Proportion of validation data (default: 0)
    pca_size : int, optional
        Number of PCA components (0 means no PCA, default: 0)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns
    -------
    list
        List containing train/val/test data in format:
        [X_train, y_train, X_val, y_val, X_test, y_test] if val_size > 0
        or [X_train, y_train, X_test, y_test] if val_size = 0
    """
    
    data = np.load(path_data)
    gt = np.load(path_gt)
    
    _, _, bands = data.shape
    X = data.reshape(-1, bands)
    y = gt.flatten()
    X = X[y > 0]
    y = y[y > 0] - 1
    X = (X - X.mean(0)) / X.std(0)
    
    if pca_size > 0:
        pca = PCA(n_components=pca_size)
        X = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size+val_size, 
        random_state=random_state)
    
    if val_size > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, 
            test_size=test_size, 
            random_state=random_state)

    if val_size > 0:
        return [X_train, y_train, X_val, y_val, X_test, y_test]
    return [X_train, y_train, X_test, y_test]
# =============

# =============
def load_models_architectures(path):
    """Load model architectures from a text file.
    
    Reads a file containing model architecture specifications where each line
    represents a model architecture as hyphen-separated integers (e.g., '64-32-16').
    Converts each line to a list of integers representing layer sizes.

    Parameters
    ----------
    path : str
        Path to the text file containing model architectures.
        Each line should contain hyphen-separated integers representing layer sizes.
        Example file content:
            64-32-16
            128-64-32-16
            256-128-64

    Returns
    -------
    list of lists
        A list where each element is a list of integers representing a model architecture.
        Example return value:
            [[64, 32, 16], [128, 64, 32, 16], [256, 128, 64]]

    Examples
    --------
    >>> architectures = load_models_architectures('model_architectures.txt')
    >>> print(architectures)
    [[64, 32, 16], [128, 64, 32, 16], [256, 128, 64]]
    """
    
    series = []
    with open(path, 'r') as f:
        for model in f:
            series.append(list(map(int, model.split('-'))))
    return series
# =======================


# =======================
# File I/O functions
# =======================
def save_params(path, model_name, args):
    """Save model parameters to args.yaml file.
    
    Parameters
    ----------
    path : str
        Directory path to save the file
    model_name : str
        Model name to write in the file
    args : dict
        Dictionary of parameters to save
    """
    
    os.makedirs(path, exist_ok=True)
    filename = 'args.yaml'
    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        f.write(f'model_name: {model_name}\n')
        for k, v in args.items():
            f.write(f'{k}: {v}\n')
# =============

# =============
def save_res(
        data, path='', 
        rewrite=True, file_name='results'):
    """Save results to CSV file with append/overwrite options.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to save
    path : str, optional
        Directory path to save file (default: current directory)
    rewrite : bool, optional
        Overwrite existing file (True) or append (False) (default: True)
    file_name : str, optional
        Filename without extension (default: 'results')
    """
    
    filename = f'{file_name}.csv'
    filepath = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    if os.path.exists(filepath) and not rewrite:
        with open(filepath, 'r') as f:
            df = pd.read_csv(f)
        data = pd.concat([df, data])
    data.to_csv(filepath, index=False)
# =======================


# =======================
# Model evaluation
# =======================
def evaluate_model(
        model, loader, 
        f1_average='weighted', 
        device='cuda'):
    """Evaluate model on loader data using accuracy and F1-score metrics.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate
    loader : torch.utils.data.DataLoader
        DataLoader with evaluation data
    f1_average : str, optional
        F1-score averaging method ('weighted', 'micro', 'macro', etc.)
    device : str, optional
        Device to use for evaluation (default: 'cuda')
    
    Returns
    -------
    tuple (float, float)
        Accuracy and F1-score values
    """
    
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average=f1_average))
# =======================


# =======================
# Loss functions
# =======================
def lasso_loss(
        model, criterion, 
        outputs, targets, l1_lambda=1e-4):
    """Compute loss with L1 regularization (Lasso).
    
    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters will be regularized
    criterion : torch.nn.modules.loss
        Loss function (e.g., nn.CrossEntropyLoss)
    outputs : torch.Tensor
        Model outputs
    targets : torch.Tensor
        Ground truth labels
    l1_lambda : float, optional
        L1 regularization coefficient (default: 1e-4)
    
    Returns
    -------
    torch.Tensor
        Loss value with L1 regularization
    """
    
    loss = criterion(outputs, targets)
    L1_norm = sum(torch.abs(param).sum() for param in model.parameters())
    return loss + l1_lambda*L1_norm
# =============

# =============
def ridge_loss(
        model, criterion, 
        outputs, targets, l2_lambda=1e-4):
    """Compute loss with L2 regularization (Ridge).
    
    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters will be regularized
    criterion : torch.nn.modules.loss
        Loss function (e.g., nn.CrossEntropyLoss)
    outputs : torch.Tensor
        Model outputs
    targets : torch.Tensor
        Ground truth labels
    l2_lambda : float, optional
        L2 regularization coefficient (default: 1e-4)
    
    Returns
    -------
    torch.Tensor
        Loss value with L2 regularization
    """
    
    loss = criterion(outputs, targets)
    L2_norm = sum(torch.norm(param, 2) for param in model.parameters())
    return loss + l2_lambda*L2_norm
# =============

# =============
def elastic_net_loss(
        model, criterion, outputs, 
        targets, l1_lambda=1e-4, l2_lambda=1e-4):
    """Compute loss with combined L1 and L2 regularization (Elastic Net).
    
    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters will be regularized
    criterion : torch.nn.modules.loss
        Loss function (e.g., nn.CrossEntropyLoss)
    outputs : torch.Tensor
        Model outputs
    targets : torch.Tensor
        Ground truth labels
    l1_lambda : float, optional
        L1 regularization coefficient (default: 1e-4)
    l2_lambda : float, optional
        L2 regularization coefficient (default: 1e-4)
    
    Returns
    -------
    torch.Tensor
        Loss value with Elastic Net regularization
    """
    
    loss = criterion(outputs, targets)
    L1_norm = sum(torch.abs(param).sum() for param in model.parameters())
    L2_norm = sum(torch.norm(param, 2) for param in model.parameters())
    return loss + l1_lambda*L1_norm + l2_lambda*L2_norm
# =======================


# =======================
# Plotting functions
# =======================
def plot_res(path_csv, path_png='', name_png='metrics'): 
    """Plot and save model comparison by Accuracy and F1-score metrics.
    
    Parameters
    ----------
    path_csv : str
        Path to CSV file with results (must contain 'Accuracy' and 'F1_score' columns)
    path_png : str, optional
        Directory to save plot (default: current directory)
    name_png : str, optional
        Output filename without extension (default: 'metrics')
    """
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    if not os.path.exists(path_csv):
        print(f"Файл {path_csv} не найден")
    else:
        df = pd.read_csv(path_csv)
        df = df.sort_values('Accuracy', ascending=False)
        
        plt.figure(figsize=(16, 10))
        
        x = range(len(df))
        bar_width = 0.35
        
        bars1 = plt.bar(x, 
                        df['Accuracy'], 
                        width=bar_width, 
                        label='Accuracy', 
                        color='#3498db')
        bars2 = plt.bar([i + bar_width for i in x], 
                        df['F1_score'], 
                        width=bar_width, 
                        label='F1_score', 
                        color='#2ecc71')
        
        plt.xlabel('Модели')
        plt.ylabel('Значение метрики')
        plt.title('Сравнение моделей по Accuracy и F1_score')
        plt.xticks([i + bar_width/2 for i in x], df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        os.makedirs(path_png, exist_ok=True)
        plt.savefig(os.path.join(path_png, name_png + '.png'), dpi=300)
        plt.show()
        plt.close()
# =============

# =============
def plot_train_val_loss(
        path_csv, path_png='', 
        name_png='train_val_loss', model_name=''):
    """Plot and save training and validation loss curves.
    
    Parameters
    ----------
    path_csv : str
        Path to CSV file with loss data (must contain 'epochs', 'train_loss', 'val_loss' columns)
    path_png : str, optional
        Directory to save plot (default: current directory)
    name_png : str, optional
        Output filename without extension (default: 'train_val_loss')
    model_name : str, optional
        Model name to display in plot title
    """
    
    os.makedirs(path_png, exist_ok=True)
    
    df = pd.read_csv(path_csv)
    plt.figure(figsize=(16, 10))
    
    plt.plot(df['epochs'], df['train_loss'], 
             label='Training Loss', 
             linewidth=2,
             color='#3498db')
    plt.plot(df['epochs'], df['val_loss'], 
             label='Validation Loss', 
             linewidth=2, 
             linestyle='--',
             color='#e74c3c')
    
    plt.xlabel('Эпоха')
    plt.ylabel('Значение ошибки')
    if model_name == '':
        plt.title(f'Сравнение train_loss и val_loss')
    else:
        plt.title(f'Сравнение train_loss и val_loss для модели {model_name}')

    plt.legend()
    plt.tight_layout()
   
    plt.savefig(os.path.join(path_png, name_png + '.png'), dpi=300)
    plt.close()
# =======================