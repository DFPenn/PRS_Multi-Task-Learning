import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")

# Load data
data_labels = pd.read_csv("E://table//PRS/all_labels.csv")
file_path = 'E://table//PRS/Only_PGS.pkl'
with open(file_path, 'rb') as file:
    data_prs_updated = pickle.load(file)


# Data Preparation
class PRSDataset(Dataset):
    def __init__(self, prs_data, labels):
        self.prs_data = prs_data
        self.labels = labels

    def __len__(self):
        return len(self.prs_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.prs_data.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float32)
        return x, y


# Preprocessing
# Normalizing the PRS data
scaler = StandardScaler()
data_prs_updated.iloc[:, 1:] = scaler.fit_transform(
    data_prs_updated.iloc[:, 1:])  # Normalize PRS features (excluding ID column)

# Merging labels and PRS datasets by ID
merged_data = data_prs_updated.merge(data_labels, left_on='FID', right_on='eid')
prs_data = merged_data.iloc[:, 1:81]  # Extract PRS columns
disease_labels = merged_data.iloc[:, 82:]  # Extract labels columns

# Aggregating labels by disease class
label_classes = disease_labels.columns.str.extract(r'Class_([A-Z]+)')[0].unique()
aggr_labels = pd.DataFrame()
for label_class in label_classes:
    class_columns = disease_labels.filter(regex=f'Class_{label_class}').columns
    aggr_labels[f'Class_{label_class}'] = disease_labels[class_columns].max(
        axis=1)  # Aggregate labels by taking max (if any disease is present, mark as 1)

# Train-test split
prs_train, prs_test, labels_train, labels_test = train_test_split(prs_data, aggr_labels, test_size=0.3, random_state=42)
train_dataset = PRSDataset(prs_train, labels_train)
test_dataset = PRSDataset(prs_test, labels_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model Definition
input_dim = 80
latent_dim = 128  # Increased latent dimension to increase model complexity
hidden_dim = 256  # Added to increase hidden layers
task_num = aggr_labels.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Increase model complexity by adding more layers and neurons
class SharedEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(SharedEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)


class MultiTaskAttention(nn.Module):
    def __init__(self, latent_dim, task_num):
        super(MultiTaskAttention, self).__init__()
        self.task_num = task_num
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, task_num),
            nn.Softmax(dim=-1)
        )

    def forward(self, latent_features):
        attn_weights = self.attention(latent_features)  # (batch_size, task_num)
        attended_features = torch.einsum('bt,bf->btf', attn_weights,
                                         latent_features)  # (batch_size, task_num, latent_dim)
        return attended_features


class TaskSpecificDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(TaskSpecificDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.decoder(x)


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, task_num):
        super(MultiTaskModel, self).__init__()
        self.shared_encoder = SharedEncoder(input_dim, latent_dim, hidden_dim)
        self.attention = MultiTaskAttention(latent_dim, task_num)
        self.decoders = nn.ModuleList([TaskSpecificDecoder(latent_dim) for _ in range(task_num)])

    def forward(self, x):
        latent_features = self.shared_encoder(x)  # (batch_size, latent_dim)
        attended_features = self.attention(latent_features)  # (batch_size, task_num, latent_dim)
        outputs = [self.decoders[i](attended_features[:, i, :]) for i in range(self.attention.task_num)]
        return torch.cat(outputs, dim=-1)


model = MultiTaskModel(input_dim, latent_dim, hidden_dim, task_num).to(device)
pos_weight = torch.tensor([15.0], device=device)  # Increased pos_weight to better handle class imbalance
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Assuming binary labels for each task
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)  # Reduced learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


# Functions
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_labels.append(y.cpu().numpy())
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Calculate AUC for training set
    auc_scores = []
    for i in range(task_num):
        if len(np.unique(all_labels[:, i])) > 1:
            auc_scores.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
        else:
            auc_scores.append(np.nan)
    avg_train_auc = np.nanmean(auc_scores)

    return total_loss / len(dataloader), avg_train_auc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            all_labels.append(y.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Calculate metrics for each task
    accuracies = []
    f1_scores = []
    auc_scores = []
    for i in range(task_num):
        if len(np.unique(all_labels[:, i])) == 1:
            # Skip this task if only one class is present in the true labels
            accuracies.append(np.nan)
            f1_scores.append(np.nan)
            auc_scores.append(np.nan)
        else:
            accuracies.append(accuracy_score(all_labels[:, i], (all_preds[:, i] > 0.5).astype(int)))
            f1_scores.append(f1_score(all_labels[:, i], (all_preds[:, i] > 0.5).astype(int), zero_division=0))
            auc_scores.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))

    # Compute average metrics, ignoring NaNs
    avg_accuracy = np.nanmean(accuracies)
    avg_f1_score = np.nanmean(f1_scores)
    avg_auc_score = np.nanmean(auc_scores)

    return total_loss / len(dataloader), avg_accuracy, avg_f1_score, avg_auc_score


# Training Loop
num_epochs = 10  # Increased number of epochs to 50
train_losses = []
train_aucs = []
test_losses = []
test_accuracies = []
test_f1_scores = []
test_auc_scores = []

for epoch in range(num_epochs):
    train_loss, avg_train_auc = train(model, train_loader, criterion, optimizer, device)
    test_loss, avg_accuracy, avg_f1_score, avg_auc_score = evaluate(model, test_loader, criterion, device)
    train_losses.append(train_loss)
    train_aucs.append(avg_train_auc)
    test_losses.append(test_loss)
    test_accuracies.append(avg_accuracy)
    test_f1_scores.append(avg_f1_score)
    test_auc_scores.append(avg_auc_score)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train AUC: {avg_train_auc:.4f}, Test Loss: {test_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}, Avg F1 Score: {avg_f1_score:.4f}, Avg AUC: {avg_auc_score:.4f}")
    scheduler.step(test_loss)

# Check for overfitting
epochs = range(1, num_epochs + 1)

# Plot train and test loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Test Loss')
plt.legend()
plt.show()

# Plot accuracy, F1 score, and AUC over epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_aucs, label='Train AUC')
plt.plot(epochs, test_auc_scores, label='Test AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('Train vs Test AUC')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.plot(epochs, test_f1_scores, label='Test F1 Score')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.title('Test Metrics Over Epochs')
plt.legend()
plt.show()
