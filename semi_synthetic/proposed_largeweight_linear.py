import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
import numpy as np
import os
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
from sklearn.cluster import KMeans
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

def create_cluster_weights(binary_y, y, categories, func):
    curr_cluster_weights = []

    for i in range(len(y)):
        if len(y[(categories == categories[i]) & (binary_y == 0)]) > 0:
            max_cluster_y = max(1, max(y[(categories == categories[i]) & (binary_y == 0)]))
        else:
            max_cluster_y = max(1, max(y[(categories == categories[i])]))
        if func == 'linear':
            curr_cluster_weights.append((y[i])/(max_cluster_y))
        elif func == 'poly':
            curr_cluster_weights.append((y[i]**5)/(max_cluster_y**5))
        elif func == 'exp':
            curr_cluster_weights.append(np.exp(y[i])/np.exp(max_cluster_y))
    
    return np.array(curr_cluster_weights)

class CurrDataset(Dataset):
    def __init__(self, name, data):
        super().__init__()
        self.name = name
        self.features = data['features']
        self.labels = data['labels']
        self.uci = data['uncensored_indicator']
        self.propensity = data['propensity']
        self.cluster_weight = data['cluster_weight']
        self.length = len(self.features)
        
    def __len__(self):
        """Return number of sequences."""
        return self.length

    def __getitem__(self, index):
        """Return sequence and label at index."""
        return self.features[index].float(), self.labels[index].long(), self.uci[index].long(), self.propensity[index],  self.cluster_weight[index]



class IPCW(nn.Module):
    def __init__(self, num_input, num_output, num_layers, hidden_size):
        super(IPCW, self).__init__()
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
                
    def forward(self, x, event_times, event_indicator, propensity, cluster_weight, lambda_loss):
        logits = self.feature_extractor(x)
        
        predictions = F.softmax(logits, dim=1)
        
        loss = self.total_loss(logits, event_times, event_indicator)
        
        propensity = torch.clamp(propensity, 0.1, 0.9) 
        IPCW_loss = torch.dot(1/propensity[event_indicator == 1], loss[event_indicator == 1])
        cluster_loss = torch.dot(cluster_weight[event_indicator == 0], loss[event_indicator == 0])
        
        loss = IPCW_loss + lambda_loss*cluster_loss*len(event_indicator[event_indicator == 0])
        
        return torch.argmax(predictions, 1), loss
    
    def total_loss(self, logits, event_times, event_indicator):
        event_times = event_times.long()
        batch_size, num_times = logits.shape
        event_times = event_times.clamp(0, num_times - 1)
                
        predictions = F.softmax(logits, dim=1)
        log_probs = torch.log(predictions + 1e-8)

        event_loss = -log_probs[torch.arange(batch_size), event_times]
        event_loss = torch.nan_to_num(event_loss, nan=0.0)  # Replace NaNs with 0
        
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

# Load and parse each line
file_path = 'proposed.txt'
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
    X_test = joblib.load('/data4/meerak/support50_propbin_data/X_test_%s.joblib'%suffix)
    y_test = joblib.load('/data4/meerak/support50_propbin_data/orig_y_test_%s.joblib'%suffix)
    binary_y_test = joblib.load('/data4/meerak/support50_propbin_data/binary_y_test_%s.joblib'%suffix)

    X_train = joblib.load('/data4/meerak/support50_propbin_data/X_train_%s.joblib'%suffix)
    y_train = joblib.load('/data4/meerak/support50_propbin_data/y_train_%s.joblib'%suffix)
    orig_y_train = joblib.load('/data4/meerak/support50_propbin_data/orig_y_train_%s.joblib'%suffix)
    binary_y_train = joblib.load('/data4/meerak/support50_propbin_data/binary_y_train_%s.joblib'%suffix)

    X_val = joblib.load('/data4/meerak/support50_propbin_data/X_val_%s.joblib'%suffix)
    y_val = joblib.load('/data4/meerak/support50_propbin_data/y_val_%s.joblib'%suffix)
    orig_y_val = joblib.load('/data4/meerak/support50_propbin_data/orig_y_val_%s.joblib'%suffix)
    binary_y_val = joblib.load('/data4/meerak/support50_propbin_data/binary_y_val_%s.joblib'%suffix)
    
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    orig_y_combined = np.concatenate([orig_y_train, orig_y_val], axis=0)
    binary_y_combined = np.concatenate([binary_y_train, binary_y_val], axis=0)

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    

    bestC, elbow_k, f_num_layers, f_hidden_size, f_batch_size, f_lr, f_wd, _, _ = hyps[suffixcount]
    f_lambda_loss = 1
    f_func = 'linear'

    for splitidx in range(idxmin, idxmax):
        set_seed(splitidx)
        X_train_new, y_train_new, orig_y_train_new, binary_y_train_new, X_val_new, y_val_new, orig_y_val_new, binary_y_val_new = random_train_val_split(X_combined, y_combined, orig_y_combined, binary_y_combined, n_train, n_val)
        
        kmeans = KMeans(n_clusters=elbow_k, random_state = 42, n_init = 10)
        categories_train = kmeans.fit_predict(X_train_new)
        categories_val = kmeans.predict(X_val_new)
        categories_test = kmeans.predict(X_test)
        
        train_cluster_weight = create_cluster_weights(binary_y_train_new, y_train_new, categories_train, f_func)
        val_cluster_weight = create_cluster_weights(binary_y_val_new, y_val_new, categories_val, f_func)
        test_cluster_weight = create_cluster_weights(binary_y_test, y_test, categories_test, f_func)

        
        best_clf = LogisticRegression(random_state=0, C = bestC).fit(X_train_new, binary_y_train_new)
        propensity_train = best_clf.predict_proba(X_train_new)[:, 1]
        propensity_val = best_clf.predict_proba(X_val_new)[:, 1]
        propensity_test = best_clf.predict_proba(X_test)[:, 1]

        train_dataset = CurrDataset("train", {'features':torch.tensor(X_train_new), 'labels':torch.tensor(y_train_new), 'uncensored_indicator':torch.tensor(binary_y_train_new), 'propensity':torch.tensor(propensity_train), 'cluster_weight':torch.tensor(train_cluster_weight)})
        train_loader = data_utils.DataLoader(train_dataset,
                                             batch_size=f_batch_size,
                                             shuffle=True)


        val_dataset = CurrDataset("train", {'features':torch.tensor(X_val_new), 'labels':torch.tensor(y_val_new), 'uncensored_indicator':torch.tensor(binary_y_val_new), 'propensity':torch.tensor(propensity_val), 'cluster_weight':torch.tensor(val_cluster_weight)})
        val_loader = data_utils.DataLoader(val_dataset,
                                             batch_size=f_batch_size,
                                             shuffle=False)

        test_dataset = CurrDataset("train", {'features':torch.tensor(X_test), 'labels':torch.tensor(y_test), 'uncensored_indicator':torch.tensor(binary_y_test), 'propensity':torch.tensor(propensity_test), 'cluster_weight':torch.tensor(test_cluster_weight)})
        test_loader = data_utils.DataLoader(test_dataset,
                                             batch_size=f_batch_size,
                                             shuffle=False)


        prop = IPCW(X_train_new.shape[1], max(y_train_new), f_num_layers, f_hidden_size).to(device)
        optimizer = optim.Adam(prop.parameters(), lr=f_lr, weight_decay=f_wd)
        optimizer.zero_grad()

        val_losses = []
        train_losses = []
        stop = -1
        for epoch in range(500):
            
            curr_train_losses = []
            for batch_idx, (data, label, uci, propensity, cluster_weight) in enumerate(train_loader):
                optimizer.zero_grad()
                data, label, uci, propensity, cluster_weight = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float(), cluster_weight.to(device).float()    
                _, loss = prop(data, label, uci, propensity, cluster_weight, f_lambda_loss)
                curr_train_losses.append(loss.detach().cpu().numpy())
                
                loss.backward()
                # step
                optimizer.step()
                
            train_losses.append(np.mean(curr_train_losses))


            curr_val_losses = []
            for batch_idx, (data, label, uci, propensity, cluster_weight) in enumerate(val_loader):
                data, label, uci, propensity, cluster_weight = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float(), cluster_weight.to(device).float()    
                output, loss = prop(data, label, uci, propensity, cluster_weight, f_lambda_loss)
                curr_val_losses.append(loss.detach().cpu().numpy())
            val_losses.append(np.mean(curr_val_losses))
            torch.save(prop, '/data4/meerak/support50_propbin_models/proposed_largeweight_linear_epoch%d_idxmin%d'%(epoch, idxmin))


            if val_losses[-1] > min(val_losses):
                stop += 1
            if val_losses[-1] == min(val_losses):
                stop = 0
            if stop == 15:
                break

        best_epoch = np.argmin(val_losses)
        prop = IPCW(X_train_new.shape[1], max(y_train_new), f_num_layers, f_hidden_size).to(device)
        prop = torch.load('/data4/meerak/support50_propbin_models/proposed_largeweight_linear_epoch%d_idxmin%d'%(best_epoch, idxmin))


        test_preds = []

        for batch_idx, (data, label, uci, propensity, cluster_weight) in enumerate(test_loader):
            data, label, uci, propensity, cluster_weight = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float(), cluster_weight.to(device).float()            
            output, _ = prop(data, label, uci, propensity, cluster_weight, f_lambda_loss)

            test_preds.extend(output.detach().cpu().numpy().reshape(-1))

        test_preds = np.array(test_preds)
        joblib.dump(test_preds, '/data4/meerak/support50_propbin_test_preds/proposed_largeweight_linear_y_test_pred_split%d_%s.joblib'%(splitidx, suffix))
        print(suffix, '%0.2f'%np.mean(np.abs(test_preds[binary_y_test == 0] - y_test[binary_y_test == 0])))

        folder_path = "/data4/meerak/support50_propbin_models"
        os.system(f"rm -f {folder_path}/*proposed_largeweight_linear*idxmin{idxmin}*")
