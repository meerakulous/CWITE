import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


METHODS = ["IPCW", "DeepHit", "Powell", "Proposed"]
LOG_FH = None
LOCK_FH = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


def set_log_file(path):
    global LOG_FH
    if LOG_FH is not None:
        LOG_FH.close()
    LOG_FH = open(path, "a", buffering=1)
    log(f"Persistent log file: {path}")


def acquire_out_dir_lock(out_dir):
    global LOCK_FH
    lock_path = Path(out_dir) / "run.lock"
    LOCK_FH = open(lock_path, "w")
    try:
        fcntl.flock(LOCK_FH.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(
            f"Another sweep process already holds {lock_path}. "
            "Use a different --out-dir, or stop the existing process before restarting."
        )
    LOCK_FH.write(f"pid={os.getpid()}\n")
    LOCK_FH.flush()
    os.fsync(LOCK_FH.fileno())
    log(f"Acquired output directory lock: {lock_path}")


def log(message):
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(line + "\n")
        LOG_FH.flush()
        os.fsync(LOG_FH.fileno())


def format_seconds(seconds):
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def log_leaderboard(best_by_method, low_margin=0.25, struggle_margin=0.75):
    if not best_by_method:
        return

    ranked = sorted(
        (
            (method, best["row"]["val_low_mae"], best["row"]["val_high_mae"], best["row"]["val_score"])
            for method, best in best_by_method.items()
        ),
        key=lambda item: item[1],
    )
    summary = "; ".join(
        f"{method}: low={low:.3f}, high={high:.3f}, score={score:.3f}"
        for method, low, high, score in ranked
    )
    log(f"RUNNING LEADERBOARD by low-bin val MAE: {summary}")

    if "Proposed" not in best_by_method:
        log("STATUS: CWITE has not run yet; too early to judge the CWITE story.")
        return

    proposed_low = best_by_method["Proposed"]["row"]["val_low_mae"]
    non_proposed = [(m, b["row"]["val_low_mae"]) for m, b in best_by_method.items() if m != "Proposed"]
    if not non_proposed:
        log("STATUS: CWITE is the only completed method so far; waiting for baselines before calling it.")
        return

    best_baseline, best_baseline_low = min(non_proposed, key=lambda item: item[1])
    gap = proposed_low - best_baseline_low
    if gap <= -low_margin:
        log(
            f"STATUS: looks like CWITE is going ahead in the least-uncensored bin "
            f"({proposed_low:.3f} vs {best_baseline} {best_baseline_low:.3f}; gap={gap:.3f})."
        )
    elif abs(gap) <= low_margin:
        log(
            f"STATUS: CWITE is basically tied with the best baseline in the least-uncensored bin "
            f"({proposed_low:.3f} vs {best_baseline} {best_baseline_low:.3f}; gap={gap:.3f})."
        )
    elif gap >= struggle_margin:
        log(
            f"STATUS: CWITE is really struggling; think we need to adjust course "
            f"({proposed_low:.3f} vs {best_baseline} {best_baseline_low:.3f}; gap={gap:.3f})."
        )
    else:
        log(
            f"STATUS: CWITE is behind but still within striking distance "
            f"({proposed_low:.3f} vs {best_baseline} {best_baseline_low:.3f}; gap={gap:.3f})."
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CurrDataset(Dataset):
    def __init__(self, x, y, e, propensity=None, cluster_weight=None):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).long()
        self.e = torch.tensor(e).long()
        self.propensity = torch.tensor(propensity if propensity is not None else np.ones(len(y))).float()
        self.cluster_weight = torch.tensor(cluster_weight if cluster_weight is not None else np.ones(len(y))).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.e[idx], self.propensity[idx], self.cluster_weight[idx]


class DiscreteTimeNet(nn.Module):
    def __init__(self, num_input, num_output, num_layers, hidden_size):
        super().__init__()
        self.num_output = int(num_output)
        if num_layers == 1:
            self.feature_extractor = nn.Linear(num_input, self.num_output)
        elif num_layers == 2:
            self.feature_extractor = nn.Sequential(
                nn.Linear(num_input, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_output),
            )
        elif num_layers == 3:
            self.feature_extractor = nn.Sequential(
                nn.Linear(num_input, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_output),
            )
        else:
            raise ValueError(f"Unsupported num_layers={num_layers}")

    def forward(self, x):
        logits = self.feature_extractor(x)
        probs = F.softmax(logits, dim=1)
        time_bins = torch.arange(self.num_output, device=logits.device).float()
        expected_time = torch.sum(probs * time_bins, dim=1)
        return logits, probs, expected_time


def load_split(data_dir, split):
    data_dir = Path(data_dir)
    log(f"Loading {split} split from {data_dir}")
    x = joblib.load(data_dir / f"real_labor_X_{split}.joblib")
    y = joblib.load(data_dir / f"real_labor_y_{split}.joblib").astype(int)
    e = joblib.load(data_dir / f"real_labor_binary_y_{split}.joblib").astype(int)
    log(
        f"Loaded {split}: X shape={x.shape}, y range=[{np.min(y)}, {np.max(y)}], "
        f"uncensored={int(np.sum(e))}/{len(e)} ({np.mean(e):.3f})"
    )
    return x, y, e


def fit_propensity(x_train, e_train, x_val, e_val, x_test):
    log("Fitting propensity model")
    cs = [1, 1e-2, 1e-4, 1e-6]
    val_scores = []
    for c in cs:
        start = time.time()
        clf = LogisticRegression(random_state=0, C=c, max_iter=1000).fit(x_train, e_train)
        score = roc_auc_score(e_val, clf.predict_proba(x_val)[:, 1])
        val_scores.append(score)
        log(f"  propensity C={c}: val AUROC={score:.4f} ({format_seconds(time.time() - start)})")
    best_c = cs[int(np.argmax(val_scores))]
    clf = LogisticRegression(random_state=0, C=best_c, max_iter=1000).fit(x_train, e_train)
    return (
        clf.predict_proba(x_train)[:, 1],
        clf.predict_proba(x_val)[:, 1],
        clf.predict_proba(x_test)[:, 1],
        best_c,
        float(max(val_scores)),
    )


def transform_features(kind, dim, x_train, y_train, e_train, x_val, x_test):
    start = time.time()
    log(f"Applying transform={kind}, dim={dim}")
    if kind == "raw":
        log(f"Transform raw complete: train shape={x_train.shape}")
        return x_train, x_val, x_test

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    if kind == "standard":
        log(f"Transform standard complete: train shape={x_train_s.shape} ({format_seconds(time.time() - start)})")
        return x_train_s, x_val_s, x_test_s
    if kind == "pca":
        n_components = min(dim, x_train_s.shape[1], x_train_s.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=0)
        out = pca.fit_transform(x_train_s), pca.transform(x_val_s), pca.transform(x_test_s)
        log(
            f"Transform PCA complete: n_components={n_components}, train shape={out[0].shape}, "
            f"explained_variance={float(np.sum(pca.explained_variance_ratio_)):.3f} "
            f"({format_seconds(time.time() - start)})"
        )
        return out
    if kind == "select":
        k = min(dim, x_train_s.shape[1])
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(x_train_s[e_train == 1], y_train[e_train == 1])
        out = selector.transform(x_train_s), selector.transform(x_val_s), selector.transform(x_test_s)
        log(f"Transform SelectKBest complete: k={k}, train shape={out[0].shape} ({format_seconds(time.time() - start)})")
        return out

    raise ValueError(f"Unknown transform kind={kind}")


def nll_per_sample(logits, y):
    y = y.long().clamp(0, logits.shape[1] - 1)
    log_probs = F.log_softmax(logits, dim=1)
    return -log_probs[torch.arange(logits.shape[0], device=logits.device), y]


def compute_loss(method, logits, probs, y, e, propensity, cluster_weight, lambda_loss, proposed_loss_scaling):
    per_nll = nll_per_sample(logits, y)
    batch_size = max(1, int(y.shape[0]))

    if method == "IPCW":
        propensity = torch.clamp(propensity, 0.1, 0.9)
        mask = e == 1
        return torch.sum((1.0 / propensity[mask]) * per_nll[mask]) / batch_size

    if method == "DeepHit":
        y_clamped = y.long().clamp(0, logits.shape[1] - 1)
        idx = torch.arange(logits.shape[0], device=logits.device)
        event_loss = per_nll * (e == 1)
        survival = 1.0 - probs.cumsum(dim=1)[idx, y_clamped]
        censored_loss = -torch.log(survival + 1e-8) * (e == 0)
        return torch.sum(torch.nan_to_num(event_loss + censored_loss, nan=0.0)) / batch_size

    if method == "Powell":
        pred_argmax = torch.argmax(probs, dim=1)
        loss = per_nll.clone()
        loss[(e == 0) & (pred_argmax > y)] = 0.0
        return torch.sum(torch.nan_to_num(loss, nan=0.0)) / batch_size

    if method == "Proposed":
        propensity = torch.clamp(propensity, 0.1, 0.9)
        uncensored = e == 1
        censored = e == 0
        ipcw_loss = torch.sum((1.0 / propensity[uncensored]) * per_nll[uncensored])
        cluster_loss = torch.sum(cluster_weight[censored] * per_nll[censored])
        if proposed_loss_scaling == "original":
            return ipcw_loss + lambda_loss * cluster_loss * torch.sum(censored)
        if proposed_loss_scaling == "normalized":
            return (ipcw_loss + lambda_loss * cluster_loss) / batch_size
        raise ValueError(f"Unknown proposed_loss_scaling={proposed_loss_scaling}")

    raise ValueError(f"Unknown method={method}")


def cluster_reference_weights(y_train, e_train, train_clusters, target_clusters, target_y, target_e, func):
    cluster_max = {}
    for c in np.unique(train_clusters):
        mask_cens = (train_clusters == c) & (e_train == 0)
        mask_all = train_clusters == c
        ref = np.max(y_train[mask_cens]) if np.any(mask_cens) else np.max(y_train[mask_all])
        cluster_max[int(c)] = max(1.0, float(ref))

    default_ref = max(1.0, float(np.max(y_train)))
    weights = np.ones(len(target_y), dtype=float)
    for i, c in enumerate(target_clusters):
        ref = cluster_max.get(int(c), default_ref)
        ratio = max(0.0, float(target_y[i])) / ref
        if func == "linear":
            weights[i] = ratio
        elif func == "sqrt":
            weights[i] = np.sqrt(ratio)
        elif func == "square":
            weights[i] = ratio**2
        elif func == "poly5":
            weights[i] = ratio**5
        elif func == "binary_top_half":
            weights[i] = 1.0 if ratio >= 0.5 else 0.0
        else:
            raise ValueError(f"Unknown weight func={func}")
    weights[target_e == 1] = 1.0
    return np.clip(weights, 0.0, 1.0)


def make_cluster_weights(config, x_train, y_train, e_train, x_val, y_val, e_val, x_test, y_test, e_test):
    if config["method"] != "Proposed":
        return np.ones(len(y_train)), np.ones(len(y_val)), np.ones(len(y_test))

    start = time.time()
    log(f"Fitting KMeans for Proposed: k={config['k']}, weight_func={config['weight_func']}")
    kmeans = KMeans(n_clusters=config["k"], random_state=config["seed"], n_init=10)
    train_clusters = kmeans.fit_predict(x_train)
    val_clusters = kmeans.predict(x_val)
    test_clusters = kmeans.predict(x_test)
    weights = (
        cluster_reference_weights(y_train, e_train, train_clusters, train_clusters, y_train, e_train, config["weight_func"]),
        cluster_reference_weights(y_train, e_train, train_clusters, val_clusters, y_val, e_val, config["weight_func"]),
        cluster_reference_weights(y_train, e_train, train_clusters, test_clusters, y_test, e_test, config["weight_func"]),
    )
    unique, counts = np.unique(train_clusters, return_counts=True)
    log(
        f"KMeans complete: clusters={len(unique)}, min_size={int(np.min(counts))}, "
        f"median_size={float(np.median(counts)):.1f}, max_size={int(np.max(counts))}, "
        f"train_censored_weight_mean={float(np.mean(weights[0][e_train == 0])):.3f} "
        f"({format_seconds(time.time() - start)})"
    )
    return weights


def predict(model, loader, device, prediction_mode):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _, _, _, _ in loader:
            logits, probs, expected = model(x.to(device))
            if prediction_mode == "expected":
                pred = expected
            elif prediction_mode == "argmax":
                pred = torch.argmax(probs, dim=1).float()
            else:
                raise ValueError(f"Unknown prediction_mode={prediction_mode}")
            preds.extend(pred.detach().cpu().numpy().reshape(-1))
    return np.asarray(preds)


def predict_distribution(model, loader, device):
    model.eval()
    probs_out = []
    with torch.no_grad():
        for x, _, _, _, _ in loader:
            _, probs, _ = model(x.to(device))
            probs_out.append(probs.detach().cpu().numpy())
    return np.vstack(probs_out)


def evaluate_loss(model, loader, device, config):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y, e, propensity, cluster_weight in loader:
            x = x.to(device)
            y = y.to(device)
            e = e.to(device)
            propensity = propensity.to(device)
            cluster_weight = cluster_weight.to(device)
            logits, probs, _ = model(x)
            loss = compute_loss(
                config["method"],
                logits,
                probs,
                y,
                e,
                propensity,
                cluster_weight,
                config["lambda_loss"],
                config["proposed_loss_scaling"],
            )
            losses.append(float(loss.detach().cpu().numpy()))
    return float(np.mean(losses))


def mae_on_uncensored(y, e, pred, mask):
    eval_mask = mask & (e == 1)
    if np.sum(eval_mask) == 0:
        return np.inf
    return float(np.mean(np.abs(y[eval_mask] - pred[eval_mask])))


def selection_score(y, e, pred, low_mask, high_mask, args):
    if args.selection_score == "overall":
        return mae_on_uncensored(y, e, pred, np.ones(len(y), dtype=bool))
    if args.selection_score == "low_high":
        low_mae = mae_on_uncensored(y, e, pred, low_mask)
        high_mae = mae_on_uncensored(y, e, pred, high_mask)
        return low_mae + args.high_penalty * high_mae
    if args.selection_score == "low_only":
        return mae_on_uncensored(y, e, pred, low_mask)
    raise ValueError(f"Unknown selection_score={args.selection_score}")


def prediction_diagnostic(name, y, e, pred, mask):
    eval_mask = mask & (e == 1)
    if np.sum(eval_mask) == 0:
        return f"{name}: no uncensored validation rows"

    y_eval = np.asarray(y[eval_mask], dtype=float)
    pred_eval = np.asarray(pred[eval_mask], dtype=float)
    err = pred_eval - y_eval
    pred_q = np.percentile(pred_eval, [0, 5, 25, 50, 75, 95, 100])
    y_q = np.percentile(y_eval, [0, 5, 25, 50, 75, 95, 100])

    rounded = np.rint(pred_eval).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    top_idx = np.argsort(counts)[-5:][::-1]
    top = ", ".join(f"{int(values[i])}:{int(counts[i])}" for i in top_idx)

    return (
        f"{name}: n={len(y_eval)} mae={np.mean(np.abs(err)):.3f} "
        f"mean_err={np.mean(err):+.3f} pred_mean={np.mean(pred_eval):.3f} y_mean={np.mean(y_eval):.3f} "
        f"pred_q=[{', '.join(f'{v:.1f}' for v in pred_q)}] "
        f"y_q=[{', '.join(f'{v:.1f}' for v in y_q)}] "
        f"top_rounded_pred={top}"
    )


def train_one(config, arrays, propensities, device, args):
    start = time.time()
    set_seed(config["seed"])
    x_train, y_train, e_train, x_val, y_val, e_val, x_test, y_test, e_test = arrays
    prop_train, prop_val, prop_test = propensities

    log(
        f"Training start: method={config['method']}, transform={config['transform']}, dim={config['dim']}, "
        f"X_train={x_train.shape}, device={device}, batch={config['batch_size']}, lr={config['lr']}, wd={config['wd']}"
    )
    w_train, w_val, w_test = make_cluster_weights(config, x_train, y_train, e_train, x_val, y_val, e_val, x_test, y_test, e_test)
    train_loader = data_utils.DataLoader(
        CurrDataset(x_train, y_train, e_train, prop_train, w_train), batch_size=config["batch_size"], shuffle=True
    )
    val_loader = data_utils.DataLoader(
        CurrDataset(x_val, y_val, e_val, prop_val, w_val), batch_size=config["batch_size"], shuffle=False
    )
    test_loader = data_utils.DataLoader(
        CurrDataset(x_test, y_test, e_test, prop_test, w_test), batch_size=config["batch_size"], shuffle=False
    )

    model = DiscreteTimeNet(x_train.shape[1], int(max(y_train) + 1), config["num_layers"], config["hidden_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    low_val_mask = prop_val < args.low_max
    high_val_mask = prop_val >= args.high_min
    log(
        f"Validation bins: low n={int(np.sum(low_val_mask))}, low uncensored={int(np.sum(e_val[low_val_mask]))}; "
        f"high n={int(np.sum(high_val_mask))}, high uncensored={int(np.sum(e_val[high_val_mask]))}"
    )

    best_val_loss = np.inf
    best_score_at_best_loss = np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        for x, y, e, propensity, cluster_weight in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            e = e.to(device)
            propensity = propensity.to(device)
            cluster_weight = cluster_weight.to(device)
            logits, probs, _ = model(x)
            loss = compute_loss(
                config["method"],
                logits,
                probs,
                y,
                e,
                propensity,
                cluster_weight,
                config["lambda_loss"],
                config["proposed_loss_scaling"],
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().numpy()))

        val_loss = evaluate_loss(model, val_loader, device, config)
        val_pred = predict(model, val_loader, device, args.prediction_mode)
        low_mae = mae_on_uncensored(y_val, e_val, val_pred, low_val_mask)
        high_mae = mae_on_uncensored(y_val, e_val, val_pred, high_val_mask)
        overall_mae = mae_on_uncensored(y_val, e_val, val_pred, np.ones(len(y_val), dtype=bool))
        score = selection_score(y_val, e_val, val_pred, low_val_mask, high_val_mask, args)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_score_at_best_loss = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if improved or epoch == 1 or epoch % args.log_every_epochs == 0 or stale >= args.patience:
            log(
                f"  epoch={epoch:03d} train_loss={float(np.mean(train_losses)):.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_overall_mae={overall_mae:.3f} val_low_mae={low_mae:.3f} "
                f"val_high_mae={high_mae:.3f} score={score:.3f} "
                f"best_val_loss={best_val_loss:.4f} "
                f"best_score_at_best_loss={best_score_at_best_loss:.3f} "
                f"stale={stale}/{args.patience} "
                f"{'NEW_BEST' if improved else ''} ({format_seconds(time.time() - epoch_start)})"
            )
            if config["method"] == "Proposed":
                log("    Proposed predictions | " + prediction_diagnostic("low-bin", y_val, e_val, val_pred, low_val_mask))
                log("    Proposed predictions | " + prediction_diagnostic("high-bin", y_val, e_val, val_pred, high_val_mask))
        if stale >= args.patience:
            log(
                f"Early stopping at epoch={epoch}; best_epoch={best_epoch}; "
                f"best_val_loss={best_val_loss:.4f}; best_score_at_best_loss={best_score_at_best_loss:.3f}"
            )
            break

    if best_state is None:
        raise RuntimeError("No best model state was saved; check validation masks and losses")
    model.load_state_dict(best_state)
    val_pred = predict(model, val_loader, device, args.prediction_mode)
    test_pred = predict(model, test_loader, device, args.prediction_mode)
    test_dist = predict_distribution(model, test_loader, device) if args.save_test_distribution else None
    overall_mae = mae_on_uncensored(y_val, e_val, val_pred, np.ones(len(y_val), dtype=bool))
    final_score = selection_score(y_val, e_val, val_pred, low_val_mask, high_val_mask, args)
    log(
        f"Training done in {format_seconds(time.time() - start)}; "
        f"best_epoch={best_epoch}; final_val_loss={best_val_loss:.4f}; final_score={final_score:.3f}"
    )
    return {
        "val_overall_mae": overall_mae,
        "val_low_mae": mae_on_uncensored(y_val, e_val, val_pred, low_val_mask),
        "val_high_mae": mae_on_uncensored(y_val, e_val, val_pred, high_val_mask),
        "val_score": final_score,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_pred": test_pred,
        "test_dist": test_dist,
    }


def build_configs(args):
    transforms = []
    if "raw" in args.transforms:
        transforms.append(("raw", 0))
    if "standard" in args.transforms:
        transforms.append(("standard", 0))
    if "pca" in args.transforms:
        transforms += [("pca", dim) for dim in args.pca_dims]
    if "select" in args.transforms:
        transforms += [("select", dim) for dim in args.select_dims]

    configs = []
    for method in args.methods:
        for transform, dim in transforms:
            common = {
                "method": method,
                "transform": transform,
                "dim": dim,
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "wd": args.wd,
                "seed": args.seed,
                "proposed_loss_scaling": args.proposed_loss_scaling,
            }
            if method == "Proposed":
                for k in args.k:
                    for weight_func in args.weight_func:
                        for lambda_loss in args.lambda_loss:
                            configs.append({**common, "k": k, "weight_func": weight_func, "lambda_loss": lambda_loss})
            else:
                configs.append({**common, "k": 0, "weight_func": "none", "lambda_loss": 0.0})
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", type=int, default=0)
    parser.add_argument("--data-dir", default="/data4/meerak/real_labor")
    parser.add_argument("--out-dir", default="/data4/meerak/cwite_realdata_final")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--low-max", type=float, default=0.10)
    parser.add_argument("--high-min", type=float, default=0.70)
    parser.add_argument("--high-penalty", type=float, default=0.25)
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--weight-func", nargs="+", default=["linear", "sqrt", "square", "poly5", "binary_top_half"])
    parser.add_argument("--lambda-loss", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--transforms", nargs="+", choices=["raw", "standard", "pca", "select"], default=["standard", "select"])
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--select-dims", type=int, nargs="+", default=[50, 100, 200, 500])
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--prediction-mode", choices=["expected", "argmax"], default="argmax")
    parser.add_argument("--proposed-loss-scaling", choices=["original", "normalized"], default="normalized")
    parser.add_argument("--selection-score", choices=["overall", "low_high", "low_only"], default="overall")
    parser.add_argument("--save-test-distribution", action="store_true", help="Save best model test PMF for D-calibration.")
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    run_start = time.time()
    set_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else out_dir / "realdata_overall_sweep.log.txt"
    set_log_file(log_path)
    acquire_out_dir_lock(out_dir)
    log(f"Arguments: {json.dumps(vars(args), sort_keys=True)}")
    log(f"Using device={device}; CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    log(f"Output directory: {out_dir}")

    x_train_raw, y_train, e_train = load_split(args.data_dir, "train")
    x_val_raw, y_val, e_val = load_split(args.data_dir, "val")
    x_test_raw, y_test, e_test = load_split(args.data_dir, "test")
    prop_train, prop_val, prop_test, best_c, val_auroc = fit_propensity(x_train_raw, e_train, x_val_raw, e_val, x_test_raw)
    log(f"Propensity best C={best_c}, val AUROC={val_auroc:.4f}")

    rows = []
    best_by_method = {}
    configs = build_configs(args)
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    log(f"Total configs to run: {len(configs)}")

    partial_csv = out_dir / "sweep_results_partial.csv"
    completed = 0
    failed = 0
    for i, config in enumerate(configs):
        config_start = time.time()
        log("=" * 80)
        log(f"SWEEP {i + 1}/{len(configs)} START: {json.dumps(config, sort_keys=True)}")
        try:
            x_train, x_val, x_test = transform_features(
                config["transform"], config["dim"], x_train_raw, y_train, e_train, x_val_raw, x_test_raw
            )
            result = train_one(
                config,
                (x_train, y_train, e_train, x_val, y_val, e_val, x_test, y_test, e_test),
                (prop_train, prop_val, prop_test),
                device,
                args,
            )
            row = {
                **config,
                "val_low_mae": result["val_low_mae"],
                "val_high_mae": result["val_high_mae"],
                "val_overall_mae": result["val_overall_mae"],
                "val_score": result["val_score"],
                "best_epoch": result["best_epoch"],
                "best_val_loss": result["best_val_loss"],
                "status": "ok",
                "error": "",
                "seconds": round(time.time() - config_start, 3),
            }
            rows.append(row)
            completed += 1
            log(
                f"SWEEP {i + 1}/{len(configs)} DONE: val_low_mae={row['val_low_mae']:.3f} "
                f"val_high_mae={row['val_high_mae']:.3f} score={row['val_score']:.3f} "
                f"({format_seconds(row['seconds'])})"
            )

            method = config["method"]
            if method not in best_by_method or row["val_score"] < best_by_method[method]["row"]["val_score"]:
                best_by_method[method] = {"row": row, "test_pred": result["test_pred"], "test_dist": result["test_dist"]}
                filename = f"{method.lower()}_feature_sweep_best_y_test_pred.joblib"
                pred_path = out_dir / filename
                config_path = out_dir / f"{method.lower()}_best_config.json"
                joblib.dump(result["test_pred"], pred_path)
                if args.save_test_distribution and result["test_dist"] is not None:
                    dist_path = out_dir / f"{method.lower()}_feature_sweep_best_y_test_dist.joblib"
                    joblib.dump(result["test_dist"], dist_path)
                    log(f"Saved test distribution for {method}: {dist_path}")
                config_path.write_text(json.dumps(row, indent=2, sort_keys=True))
                log(f"NEW METHOD BEST for {method}: saved {pred_path} and {config_path}")
                log_leaderboard(best_by_method)
        except Exception as exc:
            failed += 1
            tb = traceback.format_exc()
            row = {
                **config,
                "val_low_mae": np.nan,
                "val_high_mae": np.nan,
                "val_overall_mae": np.nan,
                "val_score": np.nan,
                "best_epoch": np.nan,
                "best_val_loss": np.nan,
                "status": "failed",
                "error": repr(exc),
                "seconds": round(time.time() - config_start, 3),
            }
            rows.append(row)
            log(f"SWEEP {i + 1}/{len(configs)} FAILED after {format_seconds(row['seconds'])}: {repr(exc)}")
            log(tb)
            (out_dir / f"failed_config_{i + 1}.json").write_text(json.dumps(row, indent=2, sort_keys=True))
            (out_dir / f"failed_config_{i + 1}.traceback.txt").write_text(tb)
            if args.fail_fast:
                raise

        log_leaderboard(best_by_method)
        if rows:
            with partial_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            elapsed = time.time() - run_start
            avg = elapsed / max(1, i + 1)
            remaining = avg * (len(configs) - i - 1)
            log(
                f"Progress: completed={completed}, failed={failed}, total_seen={i + 1}/{len(configs)}, "
                f"elapsed={format_seconds(elapsed)}, avg_per_config={format_seconds(avg)}, "
                f"ETA={format_seconds(remaining)}; partial CSV={partial_csv}"
            )

    with (out_dir / "sweep_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    log("BEST CONFIGS")
    for method, best in best_by_method.items():
        log(method + " " + json.dumps(best["row"], indent=2, sort_keys=True))
    log(f"Saved outputs to {out_dir}")
    log(f"Run finished in {format_seconds(time.time() - run_start)} with completed={completed}, failed={failed}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        log("FATAL ERROR: run terminated outside the per-config error handler")
        log(tb)
        raise
    finally:
        if LOG_FH is not None:
            LOG_FH.flush()
            os.fsync(LOG_FH.fileno())
            LOG_FH.close()
        if LOCK_FH is not None:
            fcntl.flock(LOCK_FH.fileno(), fcntl.LOCK_UN)
            LOCK_FH.close()
