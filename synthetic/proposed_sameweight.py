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
from lifelines.utils import concordance_index
import argparse
parser = argparse.ArgumentParser(description="Parse GPU core and index range inputs.")

parser.add_argument("-gpu", type=int, required=True, help="GPU")
parser.add_argument("-idxmin", type=int, required=True, help="Minimum index value")
parser.add_argument("-idxmax", type=int, required=True, help="Maximum index value")

# Parse the arguments
args = parser.parse_args()

# Process CPU core input
gpu = args.gpu

# Get index values
idx_min = args.idxmin
idx_max = args.idxmax

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


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
        
        loss = (IPCW_loss*len(event_indicator[event_indicator == 1]) + lambda_loss*cluster_loss*len(event_indicator[event_indicator == 0]))/len(event_indicator)
        
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
    
f = open('proposed_sameweight_hyp_%d_%d.txt'%(idx_min, idx_max), 'w')    

for COUNT in range(idx_min, idx_max):
    suffix = 'COUNT%d'%(COUNT)

    X_test = joblib.load('/data4/meerak/onevar_data/X_test_%s.joblib'%suffix).reshape(-1, 1)
    y_test = joblib.load('/data4/meerak/onevar_data/y_test_%s.joblib'%suffix)
    binary_y_test = joblib.load('/data4/meerak/onevar_data/actual_binary_y_test%s.joblib'%suffix)

    X_train = joblib.load('/data4/meerak/onevar_data/X_train_%s.joblib'%suffix).reshape(-1, 1)
    y_train = joblib.load('/data4/meerak/onevar_data/y_train_%s.joblib'%suffix)
    orig_y_train = joblib.load('/data4/meerak/onevar_data/orig_y_train_%s.joblib'%suffix)
    binary_y_train = joblib.load('/data4/meerak/onevar_data/binary_y_train_%s.joblib'%suffix)

    X_val = joblib.load('/data4/meerak/onevar_data/X_val_%s.joblib'%suffix).reshape(-1, 1)
    y_val = joblib.load('/data4/meerak/onevar_data/y_val_%s.joblib'%suffix)
    orig_y_val = joblib.load('/data4/meerak/onevar_data/orig_y_val_%s.joblib'%suffix)
    binary_y_val = joblib.load('/data4/meerak/onevar_data/binary_y_val_%s.joblib'%suffix)

    val_roc_scores = []
    for currC in [1, 1e-2, 1e-4, 1e-6]:
        clf = LogisticRegression(random_state=0, C = currC).fit(X_train, binary_y_train)
        y_pred = clf.predict_proba(X_val)[:, 1]
        val_roc_scores.append(roc_auc_score(binary_y_val, y_pred))

    bestC = np.array([1, 1e-2, 1e-4, 1e-6])[np.argmax(val_roc_scores)]

    best_clf = LogisticRegression(random_state=0, C = bestC).fit(X_train, binary_y_train)
    
    propensity_train = best_clf.predict_proba(X_train)[:, 1]
    propensity_val = best_clf.predict_proba(X_val)[:, 1]
    propensity_test = best_clf.predict_proba(X_test)[:, 1]


    categories_train = np.array([1]*len(y_train))
    categories_val = np.array([1]*len(y_val))
    categories_test = np.array([1]*len(y_test))

    
    print(suffix, 'AUROC propensity', roc_auc_score(binary_y_val, propensity_val))

    grid_val_losses = []
    grid_c_indices = []
    grid_params = []

    for grididx in range(10):
        print(suffix, 'PROPOSED GRIDSEARCH', grididx)
        num_layers = 3
        hidden_size = np.random.choice([16, 32, 64], 1)[0]
        batch_size = int(np.random.choice([128, 256, 512, 1024], 1)[0])
        lr = np.random.choice([1e-2, 1e-3], 1)[0]
        wd = np.random.choice([1e-4, 1e-5, 1e-6], 1)[0]
        lambda_loss = np.random.choice([1e-2, 0.05, 1e-1, 0.5, 1], 1)[0]
        func = np.random.choice(['linear', 'exp', 'poly'])
        
        train_cluster_weight = create_cluster_weights(binary_y_train, y_train, categories_train, func)
        val_cluster_weight = create_cluster_weights(binary_y_val, y_val, categories_val, func)
        test_cluster_weight = create_cluster_weights(binary_y_test, y_test, categories_test, func)

        train_dataset = CurrDataset("train", {'features':torch.tensor(X_train), 'labels':torch.tensor(y_train), 'uncensored_indicator':torch.tensor(binary_y_train), 'propensity':torch.tensor(propensity_train), 'cluster_weight':torch.tensor(train_cluster_weight)})
        train_loader = data_utils.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)


        val_dataset = CurrDataset("train", {'features':torch.tensor(X_val), 'labels':torch.tensor(y_val), 'uncensored_indicator':torch.tensor(binary_y_val), 'propensity':torch.tensor(propensity_val), 'cluster_weight':torch.tensor(val_cluster_weight)})
        val_loader = data_utils.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)

        prop = IPCW(X_train.shape[1], max(y_train), num_layers, hidden_size).to(device)
        optimizer = optim.Adam(prop.parameters(), lr=lr, weight_decay=wd)
        optimizer.zero_grad()

        val_losses = []
        c_indices = []
        stop = -1
        for epoch in range(500):
            for batch_idx, (data, label, uci, propensity, cluster_weight) in enumerate(train_loader):
                optimizer.zero_grad()
                data, label, uci, propensity, cluster_weight = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float(), cluster_weight.to(device).float()    
                _, loss = prop(data, label, uci, propensity, cluster_weight, lambda_loss)
                loss.backward()
                # step
                optimizer.step()

            curr_val_losses = []
            val_preds = []
            val_true = []
            val_obs = []
            for batch_idx, (data, label, uci, propensity, cluster_weight) in enumerate(val_loader):
                data, label, uci, propensity, cluster_weight = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float(), cluster_weight.to(device).float()    
                output, loss = prop(data, label, uci, propensity, cluster_weight, lambda_loss)
                curr_val_losses.append(loss.detach().cpu().numpy())
                val_preds.extend(output.detach().cpu().numpy())
                val_true.extend(label.detach().cpu().numpy())
                val_obs.extend(uci.detach().cpu().numpy())
                
            val_losses.append(np.mean(curr_val_losses))
            c_indices.append(concordance_index(np.array(val_true).reshape(-1), np.array(val_preds).reshape(-1), np.array(val_obs).reshape(-1)))
            
            if val_losses[-1] > min(val_losses):
                stop += 1
            if val_losses[-1] == min(val_losses):
                stop = 0
            if stop == 15:
                break
        grid_val_losses.append(min(val_losses)) 
        grid_c_indices.append(c_indices[np.argmin(val_losses)])
        grid_params.append([num_layers, hidden_size, batch_size, lr, wd, lambda_loss, func])
        print(grid_c_indices)
    
    print(grididx, suffix, 'FINAL LAYER TRAINING')
    f_num_layers, f_hidden_size, f_batch_size, f_lr, f_wd, f_lambda_loss, f_func = grid_params[np.argmax(grid_c_indices)]
    
    f.write(str(grid_params[np.argmax(grid_c_indices)]) + '\n')
    
    train_cluster_weight = create_cluster_weights(binary_y_train, y_train, categories_train, f_func)
    val_cluster_weight = create_cluster_weights(binary_y_val, y_val, categories_val, f_func)
    test_cluster_weight = create_cluster_weights(binary_y_test, y_test, categories_test, f_func)


    train_dataset = CurrDataset("train", {'features':torch.tensor(X_train), 'labels':torch.tensor(y_train), 'uncensored_indicator':torch.tensor(binary_y_train), 'propensity':torch.tensor(propensity_train), 'cluster_weight':torch.tensor(train_cluster_weight)})
    train_loader = data_utils.DataLoader(train_dataset,
                                         batch_size=f_batch_size,
                                         shuffle=True)


    val_dataset = CurrDataset("train", {'features':torch.tensor(X_val), 'labels':torch.tensor(y_val), 'uncensored_indicator':torch.tensor(binary_y_val), 'propensity':torch.tensor(propensity_val), 'cluster_weight':torch.tensor(val_cluster_weight)})
    val_loader = data_utils.DataLoader(val_dataset,
                                         batch_size=f_batch_size,
                                         shuffle=False)

    test_dataset = CurrDataset("train", {'features':torch.tensor(X_test), 'labels':torch.tensor(y_test), 'uncensored_indicator':torch.tensor(binary_y_test), 'propensity':torch.tensor(propensity_test), 'cluster_weight':torch.tensor(test_cluster_weight)})
    test_loader = data_utils.DataLoader(test_dataset,
                                         batch_size=f_batch_size,
                                         shuffle=False)


    prop = IPCW(X_train.shape[1], max(y_train), f_num_layers, f_hidden_size).to(device)
    optimizer = optim.Adam(prop.parameters(), lr=f_lr, weight_decay=f_wd)
    optimizer.zero_grad()

    val_losses = []

    stop = -1
    for epoch in range(500):
        for batch_idx, (data, label, uci, propensity, cluster_weight) in enumerate(train_loader):
            optimizer.zero_grad()
            data, label, uci, propensity, cluster_weight = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float(), cluster_weight.to(device).float()    
            _, loss = prop(data, label, uci, propensity, cluster_weight, f_lambda_loss)
            loss.backward()
            # step
            optimizer.step()

        curr_val_losses = []
        for batch_idx, (data, label, uci, propensity, cluster_weight) in enumerate(val_loader):
            optimizer.zero_grad()
            data, label, uci, propensity, cluster_weight = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float(), cluster_weight.to(device).float()    
            output, loss = prop(data, label, uci, propensity, cluster_weight, f_lambda_loss)
            curr_val_losses.append(loss.detach().cpu().numpy())
        val_losses.append(np.mean(curr_val_losses))
        torch.save(prop, '/data4/meerak/onevar_models/proposed_sameweight_epoch%d'%(epoch))


        if val_losses[-1] > min(val_losses):
            stop += 1
        if val_losses[-1] == min(val_losses):
            stop = 0
        if stop == 15:
            break


    best_epoch = np.argmin(val_losses)
    prop = IPCW(X_train.shape[1], max(y_train), f_num_layers, f_hidden_size).to(device)
    prop = torch.load('/data4/meerak/onevar_models/proposed_sameweight_epoch%d'%(best_epoch))


    test_preds = []

    for batch_idx, (data, label, uci, propensity, cluster_weight) in enumerate(test_loader):
        data, label, uci, propensity, cluster_weight = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float(), cluster_weight.to(device).float()            
        output, _ = prop(data, label, uci, propensity, cluster_weight, f_lambda_loss)

        test_preds.extend(output.detach().cpu().numpy().reshape(-1))

    test_preds = np.array(test_preds)
    joblib.dump(test_preds, '/data4/meerak/onevar_test_preds/proposed_sameweight_y_test_pred_%s.joblib'%(suffix))
    print(suffix, '%0.2f'%np.mean(np.abs(test_preds[binary_y_test == 0] - y_test[binary_y_test == 0])))
