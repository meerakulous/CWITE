import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
import numpy as np
import os
from cwite_paths import data_path, output_path, output_dir
import torch
import torch.nn as nn
torch.backends.cudnn.enabled=False
import torch.optim as optim
from torch.utils.data import Dataset
import pickle
import joblib
from sklearn.metrics import mean_absolute_error 
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import subprocess
import argparse
import random
from sklearn.model_selection import KFold

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage
set_seed(42)

def random_train_val_split(X_combined, y_combined, n_train, n_val):
    n_total = n_train + n_val
    idx = np.random.permutation(n_total)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train_new = X_combined[train_idx]
    y_train_new = y_combined[train_idx]

    X_val_new = X_combined[val_idx]
    y_val_new = y_combined[val_idx]

    return X_train_new, y_train_new, X_val_new, y_val_new


# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Parse GPU and index range inputs.")

parser.add_argument("-gpu", type=int, required=True, help="GPU index to use.")

# Parse the arguments
args = parser.parse_args()

# Access arguments
gpu = args.gpu
# Example usage
print(f"Using GPU {gpu}")



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CurrDataset(Dataset):
    def __init__(self, name, data):
        super().__init__()
        self.name = name
        self.features = data['features']
        self.labels = data['labels']
        self.length = len(self.features)
        
    def __len__(self):
        """Return number of sequences."""
        return self.length

    def __getitem__(self, index):
        """Return sequence and label at index."""
        return self.features[index].float(), self.labels[index].long()


class DeepHit(nn.Module):
    def __init__(self, num_input, num_output, num_layers, hidden_size):
        super(DeepHit, self).__init__()
        self.num_output = num_output
        if num_layers == 1:
            self.feature_extractor = nn.Sequential(
                nn.Linear(num_input, num_output)
            )
        elif num_layers == 2:
            self.feature_extractor = nn.Sequential(
                nn.Linear(num_input, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_output)
            )
        elif num_layers == 3:
            self.feature_extractor = nn.Sequential(
                nn.Linear(num_input, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_output)
            )
                
    def forward(self, x, event_times):
        logits = self.feature_extractor(x)
        
        predictions = F.softmax(logits, dim=1)
        loss = self.total_loss(logits, event_times).mean()
                
        return torch.argmax(predictions, 1), loss
    
    def total_loss(self, logits, event_times):
        event_times = event_times.long()
        batch_size, num_times = logits.shape
        event_times = event_times.clamp(0, num_times - 1)
                
        predictions = F.softmax(logits, dim=1)
        log_probs = torch.log(predictions + 1e-8)

        event_loss = -log_probs[torch.arange(batch_size), event_times]        
        return event_loss

params = []
hospital_types = ['low_resource', 'moderate_resource', 'high_resource']
strategies = ['covariate']
thresholds = [20, 40, 60, 80]
for strategy in strategies:
    if strategy == 'hybrid':
        params.extend(['%s_T%d_%s'%(strategy, t, h) for t in thresholds for h in hospital_types])
    elif strategy == 'outcome':
        params.extend(['%s_T%d'%(strategy, t) for t in thresholds])
    elif strategy == 'covariate':
        params.extend(['%s_%s'%(strategy, h) for h in hospital_types])
    
f = open('oracle.txt', 'w')

print(params)
for suffix in params:
    X_test = joblib.load(data_path('support50_propbin_data', 'X_test_%s.joblib')%suffix)
    y_test = joblib.load(data_path('support50_propbin_data', 'orig_y_test_%s.joblib')%suffix)

    X_train = joblib.load(data_path('support50_propbin_data', 'X_train_%s.joblib')%suffix)
    y_train = joblib.load(data_path('support50_propbin_data', 'orig_y_train_%s.joblib')%suffix)

    X_val = joblib.load(data_path('support50_propbin_data', 'X_val_%s.joblib')%suffix)
    y_val = joblib.load(data_path('support50_propbin_data', 'orig_y_val_%s.joblib')%suffix)
    
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    
    grid_val_losses = []
    grid_params = []

    print(suffix, 'BEGIN GRIDSEARCH')
    for grididx in range(25):
        num_layers = np.random.choice([1, 2, 3], 1)[0]
        hidden_size = np.random.choice([16, 32, 64], 1)[0]
        batch_options = {
            64: [1e-3, 5e-4],
            128: [5e-4, 1e-4],
            256: [1e-4],
        }

        batch_size = int(np.random.choice(list(batch_options.keys()), 1)[0])
        lr = np.random.choice(batch_options[batch_size], 1)[0]
        
        wd = np.random.choice([1e-4, 1e-5, 1e-6], 1)[0]

        train_dataset = CurrDataset("train", {'features':torch.tensor(X_train), 'labels':torch.tensor(y_train)})
        train_loader = data_utils.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)


        val_dataset = CurrDataset("train", {'features':torch.tensor(X_val), 'labels':torch.tensor(y_val)})
        val_loader = data_utils.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)

        model = DeepHit(X_train.shape[1], max(y_train), num_layers, hidden_size).to(device)
        # model.apply(weights_init)


        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        optimizer.zero_grad()

        val_losses = []

        stop = -1
        for epoch in range(500):
            for batch_idx, (data, label) in enumerate(train_loader):
                optimizer.zero_grad()
                data, label = data.to(device), label.to(device).float().to(device)
                _, loss = model(data, label)
                loss.backward()
                # step
                optimizer.step()

            curr_val_losses = []
            for batch_idx, (data, label) in enumerate(val_loader):
                data, label = data.to(device), label.to(device).float().to(device)    
                output, loss = model(data, label)
                curr_val_losses.append(loss.detach().cpu().numpy())
            val_losses.append(np.mean(curr_val_losses))

            if val_losses[-1] > min(val_losses):
                stop += 1
            if val_losses[-1] == min(val_losses):
                stop = 0
            if stop == 15:
                break
        grid_val_losses.append(min(val_losses)) 
        grid_params.append([num_layers, hidden_size, batch_size, lr, wd])


    print(suffix, 'FINAL LAYER TRAINING')
    f_num_layers, f_hidden_size, f_batch_size, f_lr, f_wd = grid_params[np.argmin(grid_val_losses)]
    f.write('%s\n'%(str(grid_params[np.argmin(grid_val_losses)])))
    f.flush()