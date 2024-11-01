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
prs_data = merged_data.iloc[:, 1:81]  # Extract PRS columns
disease_labels = merged_data.iloc[:, 82:]  # Extract labels columns

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
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

resampled_datasets = []

for col in aggr_labels.columns:

    current_labels = aggr_labels[col]
    current_data = prs_data.copy()

    current_data['label'] = current_labels

    print(f"Before sampling - Class {col} distribution: {current_data['label'].value_counts().to_dict()}")

    # Use SMOTE for a few class oversampling first,increased a few classes to 50%
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

# Train-test split
prs_train, prs_test, labels_train, labels_test = train_test_split(prs_data_resampled, labels_resampled, test_size=0.2,
                                                                  random_state=42)
train_dataset = PRSDataset(prs_train, labels_train)
test_dataset = PRSDataset(prs_test, labels_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

--------------------------------------------------------------------------------------------------------------------------------------------------------
# Model Definition
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
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.decoder(x)


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, latent_dim, task_num):
        super(MultiTaskModel, self).__init__()
        self.shared_encoder = SharedEncoder(input_dim, latent_dim)
        self.attention = MultiTaskAttention(latent_dim, task_num)
        self.decoders = nn.ModuleList([TaskSpecificDecoder(latent_dim) for _ in range(task_num)])

    def forward(self, x):
        latent_features = self.shared_encoder(x)  # (batch_size, latent_dim)
        attended_features = self.attention(latent_features)  # (batch_size, task_num, latent_dim)
        outputs = [self.decoders[i](attended_features[:, i, :]) for i in range(self.attention.task_num)]
        return torch.cat(outputs, dim=-1)


# Training Setup
input_dim = 80
latent_dim = 32
task_num = aggr_labels.shape[1]

model = MultiTaskModel(input_dim, latent_dim, task_num)
criterion = nn.BCEWithLogitsLoss()  # Assuming binary labels for each task
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# Modified evaluate function to return predictions
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
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())  # Use sigmoid to get probabilities

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

    return total_loss / len(dataloader), avg_accuracy, avg_f1_score, avg_auc_score, all_preds


# Training Loop with extraction of probabilities
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
all_predictions = []  # To store predictions for each epoch

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, avg_accuracy, avg_f1_score, avg_auc_score, test_preds = evaluate(model, test_loader, criterion, device)

    # Store the predictions for further use (e.g., Bayesian analysis)
    all_predictions.append(test_preds)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}, Avg F1 Score: {avg_f1_score:.4f}, Avg AUC: {avg_auc_score:.4f}")


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# After training, extract the probabilities for Bayesian Analysis

# Get predictions for all samples (train and test)
all_predictions = []
model.eval()
with torch.no_grad():
    for x, _ in DataLoader(train_dataset, batch_size=32, shuffle=False):
        x = x.to(device)
        outputs = torch.sigmoid(model(x))
        all_predictions.append(outputs.cpu().numpy())
    for x, _ in test_loader:
        x = x.to(device)
        outputs = torch.sigmoid(model(x))
        all_predictions.append(outputs.cpu().numpy())

predicted_large_class_probs = np.vstack(all_predictions)  # (total_samples, num_classes)


# Function to compute initial conditional probabilities
def compute_initial_conditional_prob(aggr_labels, disease_labels):
    conditional_probs = {}
    for large_class in aggr_labels.columns:
        conditional_probs[large_class] = {}
        large_class_indices = aggr_labels[aggr_labels[large_class] == 1].index
        for small_class in disease_labels.columns:
            if len(large_class_indices) > 0:
                conditional_probs[large_class][small_class] = disease_labels.loc[large_class_indices, small_class].mean()
            else:
                conditional_probs[large_class][small_class] = 0  # Handle case where no data is available
    return conditional_probs


# Initialize conditional probabilities from training data frequencies
conditional_probs = compute_initial_conditional_prob(aggr_labels, disease_labels)

# Function to calculate posterior probabilities P(B) in E step
def compute_posterior_probs(large_class_probs, conditional_probs, aggr_labels, disease_labels):
    posterior_probs = []
    for person_probs in large_class_probs:
        individual_probs = {}
        for small_class in disease_labels.columns:
            # Extract the large class prefix, e.g., "Class_A" from "Class_A00"
            large_class = '_'.join(small_class.split('_')[:2])
            # Get the probability for the large class
            P_A = person_probs[aggr_labels.columns.get_loc(large_class)]
            # Get the conditional probability P(B|A)
            P_B_given_A = conditional_probs.get(large_class, {}).get(small_class, 0)
            # Calculate P(B) = P(A) * P(B|A)
            individual_probs[small_class] = P_A * P_B_given_A
        posterior_probs.append(individual_probs)
    return posterior_probs

# Function to update conditional probabilities P(B|A) in M step
def compute_conditional_prob(posterior_probs, aggr_labels, disease_labels):
    conditional_probs = {}
    for large_class in aggr_labels.columns:
        conditional_probs[large_class] = {}
        # Filter samples where the large class is present
        large_class_indices = aggr_labels[aggr_labels[large_class] == 1].index
        for small_class in disease_labels.columns:
            if len(large_class_indices) > 0:
                # Calculate the mean posterior probability for the small class, given the large class
                conditional_probs[large_class][small_class] = np.mean([
                    posterior_probs[i][small_class] for i in large_class_indices if i < len(posterior_probs)
                ])
            else:
                conditional_probs[large_class][small_class] = 0
    return conditional_probs

# EM Iteration
max_iterations = 100
tolerance = 1e-4  # Convergence tolerance
for iteration in range(max_iterations):
    print(f"EM Iteration {iteration + 1}")

    # E Step: Calculate posterior probabilities P(B)
    posterior_probs = compute_posterior_probs(predicted_large_class_probs, conditional_probs, aggr_labels, disease_labels)

    # M Step: Update conditional probabilities P(B|A)
    new_conditional_probs = compute_conditional_prob(posterior_probs, aggr_labels, disease_labels)

    # Check for convergence
    diff = np.sum([
        abs(new_conditional_probs[large_class][small_class] - conditional_probs[large_class].get(small_class, 0))
        for large_class in new_conditional_probs for small_class in new_conditional_probs[large_class]
    ])

    if diff < tolerance:
        print(f"Convergence reached after {iteration + 1} iterations.")
        break

    # Update conditional probabilities for the next iteration
    conditional_probs = new_conditional_probs

