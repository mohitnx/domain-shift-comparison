"""
ordinal severity grading and conformal prediction on coffee leaf disease images

The main question here is fairly simple: severity labels are ordered, and plain
softmax ignores that. This script checks two seperate things rather than assuming
the answer: whether an ordinal-regression model (coral) actually does better
than a plain softmax classifier on a real severity dataset, and whether an
ordinal-aware conformal procedure gives tighter valid prediction sets than a
naive nominal one while keeping the model fixed.

dataset (download before running):
  https://github.com/esgario/lara2018
  use classification/dataset/dataset.csv (id, severity, ...) and
  classification/dataset/leaf/<id>.jpg
"""

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

torch.manual_seed(0)
torch.set_num_threads(1)

DATA_ROOT = "lara2018/classification/dataset"
NUM_LEVELS = 5
IMG_SIZE = 96
ALPHA = 0.10   # target conformal coverage = 90%


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_splits():
    df = pd.read_csv(f"{DATA_ROOT}/dataset.csv")
    df["path"] = df["id"].apply(lambda i: f"{DATA_ROOT}/leaf/{i}.jpg")
    import os
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    rng = np.random.default_rng(42)
    train_idx, calib_idx, test_idx = [], [], []
    for sev, group in df.groupby("severity"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_calib = max(8, int(0.15 * n))
        n_test = max(8, int(0.15 * n))
        calib_idx += idx[:n_calib].tolist()
        test_idx += idx[n_calib:n_calib + n_test].tolist()
        train_idx += idx[n_calib + n_test:].tolist()

    return {
        "train": df.loc[train_idx, ["path", "severity"]].values.tolist(),
        "calib": df.loc[calib_idx, ["path", "severity"]].values.tolist(),
        "test": df.loc[test_idx, ["path", "severity"]].values.tolist(),
    }


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])
eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])


class SeverityDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, sev = self.items[idx]
        img = self.tf(Image.open(path).convert("RGB"))
        return img, int(sev)


def make_backbone():
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Identity()
    return m


# ---------------------------------------------------------------------------
# model a: plain softmax, treats severity as 5 unrelated classes
# ---------------------------------------------------------------------------

class SoftmaxHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = make_backbone()
        self.fc = nn.Linear(512, NUM_LEVELS)

    def forward(self, x):
        return self.fc(self.backbone(x))


# ---------------------------------------------------------------------------
# model b: coral ordinal regression (cao, mirjalili, raschka 2020)
# a single shared logit z(x) plus k-1 monotonic bias terms gives
# p(y > k | x) = sigmoid(z(x) + b_k), which is rank-consistent by construction
# rather than merely penalized into being consistent.
# ---------------------------------------------------------------------------

class CoralHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = make_backbone()
        self.fc = nn.Linear(512, 1)
        # init away from zero: with all deltas = 0 the k-1 thresholds all start
        # identical, which makes every p(y>k) identical and collapses all
        # probability mass onto the two endpoint classes (0 and k-1) once you
        # take f(k) - f(k-1) differences. the optimizer escapes this saddle
        # very slowly, so start the thresholds already spread apart.
        self.bias_deltas = nn.Parameter(torch.ones(NUM_LEVELS - 1))

    def forward(self, x):
        z = self.fc(self.backbone(x)).squeeze(-1)
        # p(y>k) must be non-increasing in k (fewer classes exceed a higher
        # threshold). sigmoid is monotonic in its argument, so the bias
        # sequence must be non-increasing too. -cumsum(delta^2) enforces that.
        # (an earlier version of this used +cumsum here, which silently made
        # the model always predict one of the two extreme classes -- caught
        # by noticing training accuracy matched the base rate of class 0.)
        biases = -torch.cumsum(self.bias_deltas ** 2, dim=0)
        return z.unsqueeze(1) + biases.unsqueeze(0)   # (batch, k-1) logits for p(y>k)


def coral_labels(y):
    levels = torch.arange(NUM_LEVELS - 1).unsqueeze(0)
    return (y.unsqueeze(1) > levels).float()


def coral_to_probs(logits):
    # convert p(y>k) logits into per-class probabilities p(k|x) = f(k) - f(k-1)
    p_gt = torch.sigmoid(logits)
    ones = torch.ones(p_gt.size(0), 1)
    zeros = torch.zeros(p_gt.size(0), 1)
    fk = torch.cat([zeros, 1 - p_gt], dim=1)
    fk = torch.cat([fk, ones], dim=1)
    probs = fk[:, 1:] - fk[:, :-1]
    probs = torch.clamp(probs, min=1e-6)
    return probs / probs.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_model(model, train_dl, is_coral, epochs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in train_dl:
            opt.zero_grad()
            out = model(x)
            if is_coral:
                loss = bce(out, coral_labels(y))
                pred = coral_to_probs(out).argmax(1)
            else:
                loss = ce(out, y)
                pred = out.argmax(1)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (pred == y).sum().item()
            tot += x.size(0)
        sched.step()
        print(f"  epoch {ep+1}/{epochs}  loss={loss_sum/tot:.4f}  "
              f"train_acc={correct/tot:.4f}  elapsed={time.time()-t0:.1f}s")
    return model


def get_probs_and_labels(model, dl, is_coral):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in dl:
            out = model(x)
            probs = coral_to_probs(out) if is_coral else torch.softmax(out, dim=1)
            all_probs.append(probs.numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def acc_mae(probs, labels):
    pred = probs.argmax(1)
    return (pred == labels).mean(), np.abs(pred - labels).mean()


# ---------------------------------------------------------------------------
# conformal prediction: naive nominal raps vs ordinal-regression intervals
# ---------------------------------------------------------------------------

K_REG, LAM = 2, 0.05


def raps_score(probs, labels):
    order = np.argsort(-probs, axis=1)
    ranks = np.argsort(order, axis=1)
    cumsum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    scores = np.zeros(len(labels))
    for i, y in enumerate(labels):
        r = ranks[i, y]
        scores[i] = cumsum[i, r] + LAM * max(0, (r + 1) - K_REG)
    return scores


def raps_eval(probs, labels, qhat):
    order = np.argsort(-probs, axis=1)
    cumsum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    reg = LAM * np.maximum(0, np.arange(1, probs.shape[1] + 1) - K_REG)
    covered, sizes = [], []
    for i, y in enumerate(labels):
        size = min(np.searchsorted(cumsum[i] + reg, qhat) + 1, probs.shape[1])
        sizes.append(size)
        covered.append(y in order[i, :size])
    return np.mean(covered), np.mean(sizes)


def ordinal_interval_eval(p_calib, y_calib, p_test, y_test, alpha):
    # treat the ordinal model's expected severity as a real-valued regression
    # estimate, then conformalize the residual the way one would for regression,
    # producing a genuinely contiguous integer interval rather than an arbitrary
    # subset of classes
    levels = np.arange(NUM_LEVELS)
    mu_calib = (p_calib * levels).sum(axis=1)
    mu_test = (p_test * levels).sum(axis=1)

    resid_calib = np.abs(y_calib - mu_calib)
    n = len(resid_calib)
    qhat = np.quantile(resid_calib, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")

    lo = np.clip(np.floor(mu_test - qhat), 0, NUM_LEVELS - 1).astype(int)
    hi = np.clip(np.ceil(mu_test + qhat), 0, NUM_LEVELS - 1).astype(int)
    covered = (y_test >= lo) & (y_test <= hi)
    return covered.mean(), (hi - lo + 1).mean()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    splits = load_splits()
    print("train:", len(splits["train"]), "calib:", len(splits["calib"]),
          "test:", len(splits["test"]))

    train_dl = DataLoader(SeverityDataset(splits["train"], train_tf), batch_size=32, shuffle=True)
    calib_dl = DataLoader(SeverityDataset(splits["calib"], eval_tf), batch_size=32, shuffle=False)
    test_dl = DataLoader(SeverityDataset(splits["test"], eval_tf), batch_size=32, shuffle=False)

    print("training model a (plain softmax)")
    model_a = SoftmaxHead()
    train_model(model_a, train_dl, is_coral=False, epochs=10)

    print("training model b (coral ordinal)")
    model_b = CoralHead()
    # coral's shared single logit is a real capacity bottleneck versus 5
    # independent softmax logits, so it needs more optimization to converge
    train_model(model_b, train_dl, is_coral=True, epochs=25, lr=8e-4)

    p_calib_a, y_calib = get_probs_and_labels(model_a, calib_dl, is_coral=False)
    p_test_a, y_test = get_probs_and_labels(model_a, test_dl, is_coral=False)
    p_calib_b, _ = get_probs_and_labels(model_b, calib_dl, is_coral=True)
    p_test_b, _ = get_probs_and_labels(model_b, test_dl, is_coral=True)

    acc_a, mae_a = acc_mae(p_test_a, y_test)
    acc_b, mae_b = acc_mae(p_test_b, y_test)
    print(f"model a (softmax): accuracy={acc_a:.4f}  mae={mae_a:.4f}")
    print(f"model b (coral):   accuracy={acc_b:.4f}  mae={mae_b:.4f}")

    n = len(y_calib)

    # (a) naive nominal raps on the plain softmax model
    scores_calib_a = raps_score(p_calib_a, y_calib)
    qhat_a = np.quantile(scores_calib_a, np.ceil((n + 1) * (1 - ALPHA)) / n, method="higher")
    cov_a, size_a = raps_eval(p_test_a, y_test, qhat_a)
    print(f"(a) nominal raps on softmax:            coverage={cov_a:.3f}  avg_set_size={size_a:.2f}/{NUM_LEVELS}")

    # (b) ordinal-regression conformal interval on coral
    cov_b, size_b = ordinal_interval_eval(p_calib_b, y_calib, p_test_b, y_test, ALPHA)
    print(f"(b) ordinal interval on coral:           coverage={cov_b:.3f}  avg_width={size_b:.2f}/{NUM_LEVELS}")

    # ablation: naive nominal raps applied to coral's own probabilities, to
    # isolate the effect of the conformal procedure from the effect of the model
    scores_calib_b = raps_score(p_calib_b, y_calib)
    qhat_b = np.quantile(scores_calib_b, np.ceil((n + 1) * (1 - ALPHA)) / n, method="higher")
    cov_b_nom, size_b_nom = raps_eval(p_test_b, y_test, qhat_b)
    print(f"(ablation) nominal raps on coral's probs: coverage={cov_b_nom:.3f}  avg_set_size={size_b_nom:.2f}/{NUM_LEVELS}")


if __name__ == "__main__":
    main()
