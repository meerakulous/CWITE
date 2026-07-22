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
from lifelines.utils import concordance_index
import argparse
import random
from sklearn.model_selection import StratifiedKFold
import ast

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Example usage

def random_train_val_split(X_combined, y_combined, orig_y_combined, binary_y_combined, n_train, n_val):
    n_total = n_train + n_val
    idx = np.random.permutation(n_total)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train_new = X_combined[train_idx]
    y_train_new = y_combined[train_idx]
    orig_y_train_new = orig_y_combined[train_idx]
    binary_y_train_new = binary_y_combined[train_idx]

    X_val_new = X_combined[val_idx]
    y_val_new = y_combined[val_idx]
    orig_y_val_new = orig_y_combined[val_idx]
    binary_y_val_new = binary_y_combined[val_idx]

    return X_train_new, y_train_new, orig_y_train_new, binary_y_train_new, \
           X_val_new, y_val_new, orig_y_val_new, binary_y_val_new



# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Parse GPU and index range inputs.")

parser.add_argument("--gpu", type=int, required=True, help="GPU index to use.")
parser.add_argument("--idxmin", type=int, default=0, help="Minimum index to process.")
parser.add_argument("--idxmax", type=int, default=-1, help="Maximum index to process (exclusive).")

# Parse the arguments
args = parser.parse_args()

# Access arguments
gpu = args.gpu
idxmin = args.idxmin
idxmax = args.idxmax

# Example usage
print(f"Using GPU {gpu}, processing indices from {idxmin} to {idxmax}")


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def safe_cindex(event_times, predictions, event_observed):
    """
    Computes the concordance index with a safeguard against invalid comparisons.
    """
    import numpy as np

    event_times = np.asarray(event_times)
    predictions = np.asarray(predictions)
    event_observed = np.asarray(event_observed)

    # Check for at least one uncensored event
    num_events = np.sum(event_observed)
    if num_events < 2:
        return 0  # or 0.5 as fallback — your choice

    try:
        return concordance_index(event_times, predictions, event_observed)
    except ValueError as e:
        return 0

class CurrDataset(Dataset):
    def __init__(self, name, data):
        super().__init__()
        self.name = name
        self.features = data['features']
        self.labels = data['labels']
        self.uci = data['uncensored_indicator']
        self.length = len(self.features)
        
    def __len__(self):
        """Return number of sequences."""
        return self.length

    def __getitem__(self, index):
        """Return sequence and label at index."""
        return self.features[index].float(), self.labels[index].long(), self.uci[index].long()


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
                
    def forward(self, x, event_times, event_indicator):
        logits = self.feature_extractor(x)
        
        predictions = F.softmax(logits, dim=1)
        loss = self.total_loss(logits, event_times, event_indicator)
        return torch.argmax(predictions, 1), loss
    
    def total_loss(self, logits, event_times, event_indicator):
        event_times = event_times.long()
        batch_size, num_times = logits.shape
        event_times = event_times.clamp(0, num_times - 1)
                
        predictions = F.softmax(logits, dim=1)
        
        observed_mask = event_indicator == 1
        censored_mask = event_indicator == 0

        log_probs = torch.log(predictions + 1e-8)
        event_loss = -log_probs[torch.arange(batch_size), event_times] * observed_mask

        # Loss for censored events: maximize probability for times >= event_times
        censored_loss = -torch.log(1 - predictions.cumsum(dim=1)[torch.arange(batch_size), event_times] + 1e-8) * censored_mask
        

        event_loss = torch.nan_to_num(event_loss, nan=0.0)  # Replace NaNs with 0
        censored_loss = torch.nan_to_num(censored_loss, nan=0.0)  # Replace NaNs with 0

        loss = event_loss.sum() + censored_loss.sum()
 
        return loss
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

# Load and parse each line
file_path = 'deephit.txt'
# Load and parse each line
hyps = []
with open(file_path, 'r') as f:
    for line in f:
        if len(line.split(',')) > 2:
            hyps.append(ast.literal_eval(line.strip()))

print(params)
suffixcount = -1
for suffix in params:
    suffixcount += 1
    X_test = joblib.load(data_path('support50_propbin_data', 'X_test_%s.joblib')%suffix)
    y_test = joblib.load(data_path('support50_propbin_data', 'orig_y_test_%s.joblib')%suffix)
    binary_y_test = joblib.load(data_path('support50_propbin_data', 'binary_y_test_%s.joblib')%suffix)

    X_train = joblib.load(data_path('support50_propbin_data', 'X_train_%s.joblib')%suffix)
    y_train = joblib.load(data_path('support50_propbin_data', 'y_train_%s.joblib')%suffix)
    orig_y_train = joblib.load(data_path('support50_propbin_data', 'orig_y_train_%s.joblib')%suffix)
    binary_y_train = joblib.load(data_path('support50_propbin_data', 'binary_y_train_%s.joblib')%suffix)

    X_val = joblib.load(data_path('support50_propbin_data', 'X_val_%s.joblib')%suffix)
    y_val = joblib.load(data_path('support50_propbin_data', 'y_val_%s.joblib')%suffix)
    orig_y_val = joblib.load(data_path('support50_propbin_data', 'orig_y_val_%s.joblib')%suffix)
    binary_y_val = joblib.load(data_path('support50_propbin_data', 'binary_y_val_%s.joblib')%suffix)
    
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    orig_y_combined = np.concatenate([orig_y_train, orig_y_val], axis=0)
    binary_y_combined = np.concatenate([binary_y_train, binary_y_val], axis=0)

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    
    
    f_num_layers, f_hidden_size, f_batch_size, f_lr, f_wd = hyps[suffixcount]
    
    for splitidx in range(idxmin, idxmax):
        set_seed(splitidx)
        # These are the updated train/val arrays you can plug into model training
        X_train_new, y_train_new, orig_y_train_new, binary_y_train_new, \
X_val_new, y_val_new, orig_y_val_new, binary_y_val_new = random_train_val_split(
    X_combined, y_combined, orig_y_combined, binary_y_combined, n_train, n_val)
  
        train_dataset = CurrDataset("train", {'features':torch.tensor(X_train_new), 'labels':torch.tensor(y_train_new), 'uncensored_indicator':torch.tensor(binary_y_train_new)})
        train_loader = data_utils.DataLoader(train_dataset,
                                             batch_size=f_batch_size,
                                             shuffle=True)


        val_dataset = CurrDataset("train", {'features':torch.tensor(X_val_new), 'labels':torch.tensor(y_val_new), 'uncensored_indicator':torch.tensor(binary_y_val_new)})
        val_loader = data_utils.DataLoader(val_dataset,
                                             batch_size=f_batch_size,
                                             shuffle=False)

        test_dataset = CurrDataset("train", {'features':torch.tensor(X_test), 'labels':torch.tensor(y_test), 'uncensored_indicator':torch.tensor(binary_y_test)})
        test_loader = data_utils.DataLoader(test_dataset,
                                             batch_size=f_batch_size,
                                             shuffle=False)


        model = DeepHit(X_train_new.shape[1], max(y_train_new), f_num_layers, f_hidden_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=f_lr, weight_decay=f_wd)
        optimizer.zero_grad()

        train_losses = []
        val_losses = []

        stop = -1
        for epoch in range(500):
            curr_train_losses = []
            for batch_idx, (data, label, uci) in enumerate(train_loader):
                optimizer.zero_grad()
                data, label, uci = data.to(device), label.to(device).float(), uci.to(device) 
                _, loss = model(data, label, uci)
                loss.backward()
                curr_train_losses.append(loss.detach().cpu().numpy())
                # step
                optimizer.step()
            train_losses.append(np.mean(curr_train_losses))

            curr_val_losses = []
            for batch_idx, (data, label, uci) in enumerate(val_loader):
                data, label, uci = data.to(device), label.to(device).float(), uci.to(device)     
                output, loss = model(data, label, uci)
                curr_val_losses.append(loss.detach().cpu().numpy())
            val_losses.append(np.mean(curr_val_losses))
            torch.save(model, output_path('support50_propbin_models', 'dhcind_epoch%d_idxmin%d')%(epoch, idxmin))


            if val_losses[-1] > min(val_losses):
                stop += 1
            if val_losses[-1] == min(val_losses):
                stop = 0
            if stop == 15:
                break
        
        best_epoch = np.argmin(val_losses)
        model = DeepHit(X_train_new.shape[1], max(y_train_new), f_num_layers, f_hidden_size).to(device)
        model = torch.load(output_path('support50_propbin_models', 'dhcind_epoch%d_idxmin%d')%(best_epoch, idxmin))

        test_preds = []

        for batch_idx, (data, label, uci) in enumerate(test_loader):
            data, label, uci = data.to(device), label.to(device).float(), uci.to(device)
            output, _ = model(data, label, uci)

            test_preds.extend(output.detach().cpu().numpy().reshape(-1))

        test_preds = np.array(test_preds)
        joblib.dump(test_preds, output_path('support50_propbin_test_preds', 'dhcind_y_test_pred_split%d_%s.joblib')%(splitidx, suffix))
        print(suffix, '%0.2f'%np.mean(np.abs(test_preds[binary_y_test == 1] - y_test[binary_y_test == 1])))

        folder_path = output_dir('support50_propbin_models')
        os.system(f"rm -f {folder_path}/*dhcind*idxmin{idxmin}*")