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
prs_data = merged_data.iloc[:, 1:81]
disease_labels = merged_data.iloc[:, 82:]

# Aggregating labels by disease class
label_classes = disease_labels.columns.str.extract(r'Class_([A-Z]+)')[0].unique()
aggr_labels = pd.DataFrame()
for label_class in label_classes:
    class_columns = disease_labels.filter(regex=f'Class_{label_class}').columns
    aggr_labels[f'Class_{label_class}'] = disease_labels[class_columns].max(
        axis=1)

# Data Balance Check and SMOTE
for col in aggr_labels.columns:
    print(f"Class {col}: {aggr_labels[col].sum()/len(aggr_labels)} even rate")

----------------------------------------------------------------------------------------------------------------------------------------------
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

resampled_datasets = []

for col in aggr_labels.columns:

    current_labels = aggr_labels[col]
    current_data = prs_data.copy()

    current_data['label'] = current_labels

    print(f"Before sampling - Class {col} distribution: {current_data['label'].value_counts().to_dict()}")

    # Use SMOTE for a few class oversampling first, increased a few classes to 50%
    smote = SMOTE(sampling_strategy=0.5, random_state=2024)
    try:
        smote_data, smote_labels = smote.fit_resample(current_data.iloc[:, :-1], current_data['label'])
    except ValueError as e:
        print(f"Skipping SMOTE for class {col} due to: {str(e)}")
        resampled_datasets.append(current_data)
        continue

    smote_data = pd.DataFrame(smote_data, columns=current_data.columns[:-1])
    smote_data['label'] = smote_labels

    # Undersampling is performed to balance most classes,Undersampling most classes, reserving 80%
    rus = RandomUnderSampler(sampling_strategy=0.8, random_state=1234)
    try:
        resampled_data, resampled_labels = rus.fit_resample(smote_data.iloc[:, :-1], smote_data['label'])
    except ValueError as e:
        print(f"Skipping undersampling for class {col} due to: {str(e)}")
        resampled_datasets.append(smote_data)
        continue

    print(f"After sampling - Class {col} distribution: {pd.Series(resampled_labels).value_counts().to_dict()}")

    resampled_data = pd.DataFrame(resampled_data, columns=current_data.columns[:-1])
    resampled_data['label'] = resampled_labels

    resampled_datasets.append(resampled_data)

balanced_dataset = pd.concat(resampled_datasets, axis=0).drop_duplicates().reset_index(drop=True)

prs_data_resampled = balanced_dataset.iloc[:, :-1]
labels_resampled = pd.DataFrame()

for col in aggr_labels.columns:
    labels_resampled[col] = balanced_dataset['label'].where(balanced_dataset['label'] == 1, 0)

prs_train, prs_test, labels_train, labels_test = train_test_split(prs_data_resampled, labels_resampled, test_size=0.2,
                                                                  random_state=1232)
train_dataset = PRSDataset(prs_train, labels_train)
test_dataset = PRSDataset(prs_test, labels_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

--------------------------------------------------------------------------------------------------------------------------------------------------------
# Model Definitions
class SharedEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SharedEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)  # (batch_size, latent_dim)

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
        attended_features = torch.einsum('bt,bf->btf', attn_weights, latent_features)  # (batch_size, task_num, latent_dim)
        return attended_features

class TaskSpecificDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(TaskSpecificDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.decoder(x)  # (batch_size, 1)

class AttentionMappingModel(nn.Module):
    def __init__(self, num_general_categories, num_sub_categories):
        super(AttentionMappingModel, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(num_general_categories, num_sub_categories))

    def forward(self, general_probs):
        attention_scores = torch.matmul(general_probs, self.attention_weights)  # (batch_size, num_sub_categories)
        attention_distribution = torch.softmax(attention_scores, dim=1)
        sub_category_probs = attention_distribution
        return sub_category_probs

class EnhancedMultiTaskModel(nn.Module):
    def __init__(self, input_dim, latent_dim, task_num, num_sub_categories):
        super(EnhancedMultiTaskModel, self).__init__()
        self.task_num = task_num
        self.shared_encoder = SharedEncoder(input_dim, latent_dim)
        self.attention = MultiTaskAttention(latent_dim, task_num)
        self.decoders = nn.ModuleList([TaskSpecificDecoder(latent_dim) for _ in range(task_num)])
        self.attention_mapping = AttentionMappingModel(task_num, num_sub_categories)

    def forward(self, x):
        latent_features = self.shared_encoder(x)  # (batch_size, latent_dim)
        attended_features = self.attention(latent_features)  # (batch_size, task_num, latent_dim)
        general_outputs = [self.decoders[i](attended_features[:, i, :]) for i in range(self.task_num)]
        general_probs = torch.cat(general_outputs, dim=-1).view(x.size(0), -1)  # (batch_size, task_num)
        sub_category_probs = self.attention_mapping(torch.sigmoid(general_probs))  # (batch_size, num_sub_categories)
        return general_probs, sub_category_probs

# Training setting
input_dim = 80
latent_dim = 32
task_num = aggr_labels.shape[1]
num_sub_categories = disease_labels.shape[1]

model = EnhancedMultiTaskModel(input_dim, latent_dim, task_num, num_sub_categories)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        general_probs, _ = model(x)
        loss = criterion(general_probs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def enhanced_evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_general_preds = []
    all_sub_preds = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            general_outputs, sub_category_probs = model(x)
            loss = criterion(general_outputs, y)
            total_loss += loss.item()
            all_labels.append(y.cpu().numpy())
            all_general_preds.append(torch.sigmoid(general_outputs).cpu().numpy())
            all_sub_preds.append(sub_category_probs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_general_preds = np.vstack(all_general_preds)
    all_sub_preds = np.vstack(all_sub_preds)

    accuracies, f1_scores, auc_scores = [], [], []
    for i in range(task_num):
        if len(np.unique(all_labels[:, i])) == 1:
            accuracies.append(np.nan)
            f1_scores.append(np.nan)
            auc_scores.append(np.nan)
        else:
            accuracies.append(accuracy_score(all_labels[:, i], (all_general_preds[:, i] > 0.5).astype(int)))
            f1_scores.append(f1_score(all_labels[:, i], (all_general_preds[:, i] > 0.5).astype(int), zero_division=0))
            auc_scores.append(roc_auc_score(all_labels[:, i], all_general_preds[:, i]))

    avg_accuracy = np.nanmean(accuracies)
    avg_f1_score = np.nanmean(f1_scores)
    avg_auc_score = np.nanmean(auc_scores)

    return total_loss / len(dataloader), avg_accuracy, avg_f1_score, avg_auc_score, all_general_preds, all_sub_preds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
all_predictions = []
all_sub_category_predictions = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, avg_accuracy, avg_f1_score, avg_auc_score, general_preds, sub_preds = enhanced_evaluate(model, test_loader, criterion, device)

    # Store general and sub-category predictions
    all_predictions.append((general_preds, sub_preds))
    all_sub_category_predictions.append(sub_preds)  # Store sub-category predictions separately

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}, Avg F1 Score: {avg_f1_score:.4f}, Avg AUC: {avg_auc_score:.4f}")



# Save sub-category predictions for further evaluation
sub_category_predictions = np.vstack(all_sub_category_predictions)

# Filter only original patient records (first 336979 records)
original_sub_category_predictions = sub_category_predictions[:336979, :]