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
os.makedirs('/data4/meerak/onevar_models_pmf_extract', exist_ok=True)
os.makedirs('/data4/meerak/onevar_test_preds_pmf_extract', exist_ok=True)

class CurrDataset(Dataset):
    def __init__(self, name, data):
        super().__init__()
        self.name = name
        self.features = data['features']
        self.labels = data['labels']
        self.uci = data['uncensored_indicator']
        self.propensity = data['propensity']
        self.length = len(self.features)
        
    def __len__(self):
        """Return number of sequences."""
        return self.length

    def __getitem__(self, index):
        """Return sequence and label at index."""
        return self.features[index].float(), self.labels[index].long(), self.uci[index].long(), self.propensity[index]


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
                
    def forward(self, x, event_times, event_indicator, propensity):
        logits = self.feature_extractor(x)
        
        predictions = F.softmax(logits, dim=1)
        loss = self.total_loss(logits, event_times, event_indicator)
        propensity = torch.clamp(propensity, 0.1, 0.9) 
        loss = torch.dot(1/propensity, loss)
                
        return torch.argmax(predictions, 1), loss
    
    def total_loss(self, logits, event_times, event_indicator):
        event_times = event_times.long()
        batch_size, num_times = logits.shape
        event_times = event_times.clamp(0, num_times - 1)
                
        predictions = F.softmax(logits, dim=1)
        log_probs = torch.log(predictions + 1e-8)

        event_loss = -log_probs[torch.arange(batch_size), event_times]
        event_loss[event_indicator == 0] = 0
        event_loss = torch.nan_to_num(event_loss, nan=0.0)  # Replace NaNs with 0
        
        return event_loss


f = open('pmf_extract_IPCW_hyp_%d_%d.txt'%(idx_min, idx_max), 'w')
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

    print(suffix, 'AUROC propensity', roc_auc_score(binary_y_val, propensity_val))

    f.write('%d, %0.3f\n'%(COUNT, roc_auc_score(binary_y_val, propensity_val)))
    f.flush()
    grid_val_losses = []
    grid_c_indices = []
    grid_params = []

    print(suffix, 'BEGIN GRIDSEARCH')
    for grididx in range(5):
        num_layers = 3
        hidden_size = np.random.choice([16, 32, 64], 1)[0]
        batch_size = int(np.random.choice([128, 256, 512, 1024], 1)[0])
        lr = np.random.choice([1e-2, 1e-3], 1)[0]
        wd = np.random.choice([1e-4, 1e-5, 1e-6], 1)[0]

        train_dataset = CurrDataset("train", {'features':torch.tensor(X_train), 'labels':torch.tensor(y_train), 'uncensored_indicator':torch.tensor(binary_y_train), 'propensity':torch.tensor(propensity_train)})
        train_loader = data_utils.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)


        val_dataset = CurrDataset("train", {'features':torch.tensor(X_val), 'labels':torch.tensor(y_val), 'uncensored_indicator':torch.tensor(binary_y_val), 'propensity':torch.tensor(propensity_val)})
        val_loader = data_utils.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)

        model = DeepHit(X_train.shape[1], max(y_train), num_layers, hidden_size).to(device)
        # model.apply(weights_init)


        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        optimizer.zero_grad()

        val_losses = []
        c_indices = []
        stop = -1
        for epoch in range(500):
            for batch_idx, (data, label, uci, propensity) in enumerate(train_loader):
                optimizer.zero_grad()
                data, label, uci, propensity = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float()    
                _, loss = model(data, label, uci, propensity)
                loss.backward()
                # step
                optimizer.step()

            curr_val_losses = []
            val_preds = []
            val_true = []
            val_obs = []
            for batch_idx, (data, label, uci, propensity) in enumerate(val_loader):
                data, label, uci, propensity = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float()  
                output, loss = model(data, label, uci, propensity)
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
        grid_params.append([num_layers, hidden_size, batch_size, lr, wd])


    print(suffix, 'FINAL LAYER TRAINING')
    f_num_layers, f_hidden_size, f_batch_size, f_lr, f_wd = grid_params[np.argmax(grid_c_indices)]
    f.write(str(grid_params[np.argmax(grid_c_indices)]) + '\n')

    train_dataset = CurrDataset("train", {'features':torch.tensor(X_train), 'labels':torch.tensor(y_train), 'uncensored_indicator':torch.tensor(binary_y_train), 'propensity':torch.tensor(propensity_train)})
    train_loader = data_utils.DataLoader(train_dataset,
                                         batch_size=f_batch_size,
                                         shuffle=True)


    val_dataset = CurrDataset("train", {'features':torch.tensor(X_val), 'labels':torch.tensor(y_val), 'uncensored_indicator':torch.tensor(binary_y_val), 'propensity':torch.tensor(propensity_val)})
    val_loader = data_utils.DataLoader(val_dataset,
                                         batch_size=f_batch_size,
                                         shuffle=False)

    test_dataset = CurrDataset("train", {'features':torch.tensor(X_test), 'labels':torch.tensor(y_test), 'uncensored_indicator':torch.tensor(binary_y_test), 'propensity':torch.tensor(propensity_test)})
    test_loader = data_utils.DataLoader(test_dataset,
                                         batch_size=f_batch_size,
                                         shuffle=False)


    model = DeepHit(X_train.shape[1], max(y_train), f_num_layers, f_hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=f_lr, weight_decay=f_wd)
    optimizer.zero_grad()

    val_losses = []

    stop = -1
    for epoch in range(500):
        for batch_idx, (data, label, uci, propensity) in enumerate(train_loader):
            optimizer.zero_grad()
            data, label, uci, propensity = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float()    
            _, loss = model(data, label, uci, propensity)
            loss.backward()
            # step
            optimizer.step()

        curr_val_losses = []
        for batch_idx, (data, label, uci, propensity) in enumerate(val_loader):
            data, label, uci, propensity = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float()       
            output, loss = model(data, label, uci, propensity)
            curr_val_losses.append(loss.detach().cpu().numpy())
        val_losses.append(np.mean(curr_val_losses))
        torch.save(model, '/data4/meerak/onevar_models_pmf_extract/icind_epoch%d'%epoch)


        if val_losses[-1] > min(val_losses):
            stop += 1
        if val_losses[-1] == min(val_losses):
            stop = 0
        if stop == 15:
            break


    best_epoch = np.argmin(val_losses)
    model = DeepHit(X_train.shape[1], max(y_train), f_num_layers, f_hidden_size).to(device)
    model = torch.load('/data4/meerak/onevar_models_pmf_extract/icind_epoch%d'%(best_epoch), weights_only=False)

    os.makedirs('/data4/meerak/onevar_test_pmfs', exist_ok=True)
    test_preds = []
    test_pmfs = []

    for batch_idx, (data, label, uci, propensity) in enumerate(test_loader):
        data, label, uci, propensity = data.to(device), label.to(device).float(), uci.to(device), propensity.to(device).float()
        with torch.no_grad():
            output, _ = model(data, label, uci, propensity)
            pmf = F.softmax(model.feature_extractor(data), dim=1)

        test_preds.extend(output.detach().cpu().numpy().reshape(-1))
        test_pmfs.extend(pmf.detach().cpu().numpy())

    test_preds = np.array(test_preds)
    test_pmfs = np.array(test_pmfs)
    joblib.dump(test_preds, '/data4/meerak/onevar_test_preds_pmf_extract/icind_y_test_pred_%s.joblib'%(suffix))
    joblib.dump(test_pmfs, '/data4/meerak/onevar_test_pmfs/icind_y_test_pmf_%s.joblib'%(suffix))
    print(suffix, '%0.2f'%np.mean(np.abs(test_preds[binary_y_test == 1] - y_test[binary_y_test == 1])))
