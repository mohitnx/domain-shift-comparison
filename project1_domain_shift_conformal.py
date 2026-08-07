"""
domain shift and conformal prediction on tomato leaf disease images

This is a small check of a simple question: does the usual conformal-coverage
guarantee break when the train and test domains shift, and can weighted
conformal prediction help? The setup is plantvillage -> plantdoc, which is a
well-known accuracy collapse in the literature.

datasets (download before running):
  plantvillage: https://github.com/spMohanty/PlantVillage-Dataset
    -> use raw/color/Tomato___* folders
  plantdoc:     https://github.com/pratikkayal/PlantDoc-Dataset
    -> use train/ and test/ folders, tomato classes only

set PV_DIR and PD_DIR below to wherever you cloned these.
"""

import os
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.linear_model import LogisticRegression

torch.manual_seed(0)
random.seed(0)
torch.set_num_threads(1)

PV_DIR = "data/plantvillage_tomato"   # folders named Tomato___<class>
PD_DIR = "data/plantdoc_tomato"       # train/ and test/ subfolders, plantdoc class names
IMG_SIZE = 64
ALPHA = 0.10          # target conformal coverage = 1 - alpha = 90%
K_REG, LAM = 2, 0.05  # raps regularization: only penalize ranks past the top-2

CLASSES = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
    "Septoria_leaf_spot", "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus", "healthy",
]
PV_MAP = {c: f"Tomato___{c}" for c in CLASSES}
PD_MAP = {
    "Bacterial_spot": "Tomato leaf bacterial spot",
    "Early_blight": "Tomato Early blight leaf",
    "Late_blight": "Tomato leaf late blight",
    "Leaf_Mold": "Tomato mold leaf",
    "Septoria_leaf_spot": "Tomato Septoria leaf spot",
    "Tomato_mosaic_virus": "Tomato leaf mosaic virus",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato leaf yellow virus",
    "healthy": "Tomato leaf",
}


# ---------------------------------------------------------------------------
# data splits
# ---------------------------------------------------------------------------

def build_splits(n_train=260, n_calib=60, n_test_src=60, n_domain_pool=100):
    # source domain (plantvillage): split each class into train / calib / in-domain test
    splits = {"train": [], "calib": [], "test_source": [], "test_target": [],
              "domain_pool_source": [], "domain_pool_target": []}

    for cls in CLASSES:
        pv_folder = os.path.join(PV_DIR, PV_MAP[cls])
        files = sorted(os.listdir(pv_folder))
        random.shuffle(files)
        need = n_train + n_calib + n_test_src
        used = files[:need]
        splits["train"] += [(os.path.join(pv_folder, f), cls) for f in used[:n_train]]
        splits["calib"] += [(os.path.join(pv_folder, f), cls) for f in used[n_train:n_train + n_calib]]
        splits["test_source"] += [(os.path.join(pv_folder, f), cls)
                                   for f in used[n_train + n_calib:need]]
        # leftover plantvillage images (not used above) go into the unlabeled
        # source pool used only to fit the domain classifier
        leftover = [f for f in files if f not in used]
        random.shuffle(leftover)
        splits["domain_pool_source"] += [(os.path.join(pv_folder, f), "source")
                                          for f in leftover[:n_domain_pool]]

        # target domain (plantdoc): combine train+test folders, use all of it
        # since plantdoc is small per class; used both for evaluation (with
        # labels) and as the unlabeled target pool for the domain classifier
        for split_name in ["train", "test"]:
            pd_folder = os.path.join(PD_DIR, split_name, PD_MAP[cls])
            if os.path.isdir(pd_folder):
                for f in sorted(os.listdir(pd_folder)):
                    splits["test_target"].append((os.path.join(pd_folder, f), cls))

    splits["domain_pool_target"] = [(p, "target") for p, c in splits["test_target"]]
    return splits


# ---------------------------------------------------------------------------
# datasets / model
# ---------------------------------------------------------------------------

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])


class LeafDataset(Dataset):
    # label mode returns (image, class_index); domain mode returns (image, 0/1 domain flag)
    def __init__(self, items, tf, mode="label"):
        self.items, self.tf, self.mode = items, tf, mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, tag = self.items[idx]
        img = self.tf(Image.open(path).convert("RGB"))
        if self.mode == "label":
            return img, CLASSES.index(tag)
        return img, (0 if tag == "source" else 1)


def train_classifier(train_items, epochs=12):
    dl = DataLoader(LeafDataset(train_items, train_tf), batch_size=64, shuffle=True)
    model = torchvision.models.resnet18(weights=None, num_classes=len(CLASSES))
    # trained from scratch: no internet access to imagenet weights in this
    # environment. on a normal gpu box, start from pretrained weights instead.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in dl:
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
        sched.step()
        print(f"epoch {epoch+1}/{epochs}  loss={loss_sum/tot:.4f}  "
              f"train_acc={correct/tot:.4f}  elapsed={time.time()-t0:.1f}s")
    return model


def get_logits(model, items):
    dl = DataLoader(LeafDataset(items, eval_tf), batch_size=64, shuffle=False)
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            logits.append(model(x).numpy())
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# raps conformal prediction (nominal, order-agnostic)
# ---------------------------------------------------------------------------

def raps_score(probs, labels):
    # cumulative sorted-probability mass up to and including the true class,
    # plus a small penalty for how far down the ranking it sits
    order = np.argsort(-probs, axis=1)
    ranks = np.argsort(order, axis=1)
    cumsum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    scores = np.zeros(len(labels))
    for i, y in enumerate(labels):
        r = ranks[i, y]
        scores[i] = cumsum[i, r] + LAM * max(0, (r + 1) - K_REG)
    return scores


def raps_eval(probs, labels, qhat):
    # build prediction sets by adding classes in probability order until the
    # cumulative (regularized) score would exceed qhat
    order = np.argsort(-probs, axis=1)
    cumsum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    reg = LAM * np.maximum(0, np.arange(1, probs.shape[1] + 1) - K_REG)
    covered, sizes = [], []
    for i, y in enumerate(labels):
        size = min(np.searchsorted(cumsum[i] + reg, qhat) + 1, probs.shape[1])
        sizes.append(size)
        covered.append(y in order[i, :size])
    return np.mean(covered), np.mean(sizes)


# ---------------------------------------------------------------------------
# weighted conformal prediction under covariate shift (tibshirani et al. 2019)
# ---------------------------------------------------------------------------

def fit_domain_classifier(model, domain_pool_items):
    # backbone features (penultimate layer) from the trained classifier, used
    # only to tell source images apart from target images, not disease class
    backbone = nn.Sequential(*list(model.children())[:-1])
    dl = DataLoader(LeafDataset(domain_pool_items, eval_tf, mode="domain"),
                    batch_size=64, shuffle=False)
    feats, doms = [], []
    backbone.eval()
    with torch.no_grad():
        for x, d in dl:
            f = backbone(x).squeeze(-1).squeeze(-1)
            feats.append(f.numpy())
            doms.append(d.numpy())
    feats, doms = np.concatenate(feats), np.concatenate(doms)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(feats, doms)
    prior_ratio = (doms == 0).sum() / (doms == 1).sum()
    return backbone, clf, prior_ratio


def embed(backbone, items):
    dl = DataLoader(LeafDataset(items, eval_tf, mode="domain"), batch_size=64, shuffle=False)
    feats = []
    backbone.eval()
    with torch.no_grad():
        for x, _ in dl:
            feats.append(backbone(x).squeeze(-1).squeeze(-1).numpy())
    return np.concatenate(feats)


def density_ratio(clf, prior_ratio, feats):
    # e(x) = P(domain = target | x) from the domain classifier; converted to
    # a likelihood ratio w(x) ~ dP_target(x)/dP_source(x) via bayes' rule
    e = np.clip(clf.predict_proba(feats)[:, 1], 1e-3, 1 - 1e-3)
    return (e / (1 - e)) * prior_ratio


def weighted_quantile(scores_calib, w_calib, w_test_point, alpha):
    # per-test-point weighted conformal quantile. the test point's own weight
    # is included as a point mass, per the weighted-exchangeability construction
    order = np.argsort(scores_calib)
    s_sorted, w_sorted = scores_calib[order], w_calib[order]
    cum_p = np.cumsum(w_sorted) / (w_sorted.sum() + w_test_point)
    idx = np.searchsorted(cum_p, 1 - alpha)
    return np.inf if idx >= len(s_sorted) else s_sorted[idx]


def weighted_raps_eval(probs, labels, w_test_all, scores_calib, w_calib, alpha):
    order = np.argsort(-probs, axis=1)
    cumsum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    reg = LAM * np.maximum(0, np.arange(1, probs.shape[1] + 1) - K_REG)
    covered, sizes = [], []
    for i, y in enumerate(labels):
        qhat_i = weighted_quantile(scores_calib, w_calib, w_test_all[i], alpha)
        size = min(np.searchsorted(cumsum[i] + reg, qhat_i) + 1, probs.shape[1])
        sizes.append(size)
        covered.append(y in order[i, :size])
    return np.mean(covered), np.mean(sizes)


# ---------------------------------------------------------------------------
# grad-cam, to check whether the model is actually looking at the leaf
# ---------------------------------------------------------------------------

def gradcam(model, img_tensor):
    activations, gradients = {}, {}
    target_layer = model.layer4[-1]
    h1 = target_layer.register_forward_hook(lambda m, i, o: activations.setdefault("v", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.setdefault("v", go[0]))
    out = model(img_tensor.unsqueeze(0))
    cls = out.argmax(1).item()
    model.zero_grad()
    out[0, cls].backward()
    act, grad = activations["v"][0], gradients["v"][0]
    weights = grad.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * act).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear")[0, 0]
    h1.remove()
    h2.remove()
    return cam.detach().numpy(), cls


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    splits = build_splits()
    print("train:", len(splits["train"]), "calib:", len(splits["calib"]),
          "test_source:", len(splits["test_source"]), "test_target:", len(splits["test_target"]))

    model = train_classifier(splits["train"], epochs=12)

    logits_calib, y_calib = get_logits(model, splits["calib"])
    logits_test_src, y_test_src = get_logits(model, splits["test_source"])
    logits_test_tgt, y_test_tgt = get_logits(model, splits["test_target"])

    acc_src = (logits_test_src.argmax(1) == y_test_src).mean()
    acc_tgt = (logits_test_tgt.argmax(1) == y_test_tgt).mean()
    print(f"in-domain accuracy: {acc_src:.4f}   out-of-domain accuracy: {acc_tgt:.4f}")

    p_calib = softmax(logits_calib)
    scores_calib = raps_score(p_calib, y_calib)
    n = len(scores_calib)
    qhat = np.quantile(scores_calib, np.ceil((n + 1) * (1 - ALPHA)) / n, method="higher")

    cov_src, size_src = raps_eval(softmax(logits_test_src), y_test_src, qhat)
    cov_tgt, size_tgt = raps_eval(softmax(logits_test_tgt), y_test_tgt, qhat)
    print(f"standard raps -> in-domain coverage: {cov_src:.3f} (set size {size_src:.2f}), "
          f"out-of-domain coverage: {cov_tgt:.3f} (set size {size_tgt:.2f})")

    domain_pool = splits["domain_pool_source"] + splits["domain_pool_target"]
    backbone, clf, prior_ratio = fit_domain_classifier(model, domain_pool)

    feat_calib = embed(backbone, [(p, "source") for p, c in splits["calib"]])
    feat_test_tgt = embed(backbone, [(p, "target") for p, c in splits["test_target"]])
    w_calib = density_ratio(clf, prior_ratio, feat_calib)
    w_test_tgt = density_ratio(clf, prior_ratio, feat_test_tgt)

    cov_tgt_w, size_tgt_w = weighted_raps_eval(
        softmax(logits_test_tgt), y_test_tgt, w_test_tgt, scores_calib, w_calib, ALPHA)
    print(f"weighted raps -> out-of-domain coverage: {cov_tgt_w:.3f} (set size {size_tgt_w:.2f})")

    ess = w_calib.sum() ** 2 / (w_calib ** 2).sum()
    print(f"effective sample size of reweighted calibration set: {ess:.1f} / {len(w_calib)}")


if __name__ == "__main__":
    main()
