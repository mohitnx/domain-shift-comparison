# Uncertainty Quantification for Plant Disease Diagnosis

Two small experiments using the same reliability/UQ setup as the earlier
medical-imaging work, applied here to plant-disease images. Same core idea,
different data and labels. A couple of different failure modes, but the main
question is still the same: do the uncertainty estimates hold up in practice?

Both scripts are single-file, run end-to-end on CPU, and train in under an
hour on small (~1,700-2,400 image) subsets. They are meant to isolate a
clear method question, not chase state-of-the-art accuracy.

## Project 1 — `project1_domain_shift_conformal.py`

**Question:** PlantVillage (lab photos, uniform background) to PlantDoc (real
field photos) is a documented accuracy collapse. Does the conformal prediction
*coverage guarantee* collapse too, separately from accuracy, and can it be
repaired?

**Method:** ResNet18 trained on PlantVillage, then checked in-domain and on
PlantDoc. RAPS calibration on PlantVillage, plus weighted conformal prediction
(Tibshirani et al., 2019) using a logistic-regression domain classifier on
backbone embeddings to estimate the source/target density ratio. Grad-CAM is
mostly a sanity check on where the model is looking.

**Results:**

| | in-domain (PlantVillage) | out-of-domain (PlantDoc) |
|---|---|---|
| accuracy | 94.1% | 17.2% |
| RAPS coverage (target 90%) | 99.6% | **40.1%** |
| RAPS avg. set size | 2.82 / 8 | 2.75 / 8 |
| weighted-conformal coverage | — | **96.9%** |
| weighted-conformal avg. set size | — | 7.70 / 8 |

Effective sample size of the reweighted calibration set: **6.6 / 480** (1.4%).

**Reading:** the coverage collapse (99.6% → 40.1%) is a sharper failure than
the accuracy drop — it means a "90% confidence" prediction set from a
lab-trained model is actively misleading in the field, not just less accurate.
Weighted conformal prediction restores the guarantee, but the effective sample
size shows why that costs a lot: only ~6-7 of 480 calibration points really
look like the field distribution, so the corrected quantile has to hedge hard
(avg. set size 7.70/8) to stay valid. The fix is provably correct, but not free.

## Project 2 — `project2_ordinal_severity_conformal.py`

**Question:** severity labels are ordered (0 = healthy, 4 = most severe).
Does an ordinal-regression model (CORAL) actually out-predict a plain softmax
classifier on a real severity-grading dataset, and does an ordinal-aware
conformal procedure give tighter valid prediction sets than a naive one,
holding the model fixed? These are different questions, and conflating them is
a common mistake.

**Method:** coffee-leaf biotic-stress dataset (Esgario et al., 2020), 5-level
ordinal severity. Model A is plain softmax ResNet18. Model B is CORAL — a
single shared logit plus structurally monotonic bias thresholds, with rank
consistency built in. Conformal comparison: naive nominal RAPS vs. an
ordinal interval built from CORAL's predicted expected severity, conformalized
like a real-valued regression residual.

**Results:**

| model | test accuracy | MAE (severity levels) |
|---|---|---|
| A — softmax | **81.6%** | **0.204** |
| B — CORAL | 76.8% | 0.244 |

| conformal procedure | coverage | avg. set size / interval width |
|---|---|---|
| (a) nominal RAPS on softmax (Model A) | 98.0% | **2.82 / 5** |
| (b) ordinal interval on CORAL (Model B) | 99.6% | 3.15 / 5 |
| (ablation) nominal RAPS on CORAL's own probs | 99.6% | 3.47 / 5 |

**Reading:** plain softmax beat CORAL on both accuracy and MAE, even after
giving CORAL 2.5x the training epochs — not the result the "ordinal method
should obviously win" intuition predicts. With ~1,200 training images,
funneling everything through CORAL's single shared logit is a real capacity
bottleneck the structural guarantee does not make up for. But holding the model
fixed, (b) vs. the ablation isolates the actual effect of exploiting order at
the conformal-set stage: ~9% tighter valid intervals (3.15 vs 3.47), independent
of whether the underlying model is ordinal-aware. The two questions have
different answers, and only checking the second would have overstated ordinal
regression's benefit.

## A debugging note (kept deliberately, in Project 2)

The first CORAL version had the monotonicity direction backwards
(non-decreasing biases instead of non-increasing), which silently collapsed
the model into one of the two extreme severity classes. It looked like a
training failure (~17% accuracy) rather than a wiring bug. That was easy to
spot because the number matched one class's base rate a bit too closely, and a
zero-init saddle point made it worse. Both issues are pretty specific to
ordinal regression's structure, and that is part of the answer to whether it
really helps, I think.

## Relation to the broader goal (reliability/explainability in high-stakes ML)

These are not standalone agriculture projects; they are the same UQ
question set (does the model know what it doesn't know, and does that
knowledge survive the conditions it will actually be deployed under) run on a
domain where the ground truth and the failure modes are easy to verify and
fairly quick to iterate on. The mapping to healthcare is pretty direct:

- **Domain shift** (Project 1) is the same problem as a model trained at one
  hospital failing at another, or a dermatology model trained on one skin tone
  distribution failing on another — different acquisition conditions creating
  a covariate shift that breaks both accuracy and stated confidence. Weighted
  conformal prediction and its efficiency cost (measured via effective sample
  size) apply unchanged.
- **Ordinal severity grading** (Project 2) is structurally identical to
  diabetic retinopathy grading, cancer staging, or Gleason scoring — clinical
  problems that are already posed as ordinal, where the same question applies:
  does an ordinal model's architecture actually earn its keep over a simpler
  baseline, and does the *uncertainty quantification procedure* respect order
  even when the underlying model doesn't need to.

Both projects use the same basic tools as the earlier medical-imaging work
(RAPS, calibration, Grad-CAM) rather than ag-specific methods, so the overall
story is fairly coherant: validity and honesty of uncertainty estimates under
real deployment conditions, just applied to a different domain.

## Setup

```
pip install -r requirements.txt
bash download_data.sh          # clones and arranges both datasets locally
python3 project1_domain_shift_conformal.py
python3 project2_ordinal_severity_conformal.py
```

## Reproducing

Both scripts expect the relevant dataset downloaded locally (paths and
download links are in the docstring at the top of each file). Everything here
ran from scratch on CPU (no pretrained ImageNet weights — no internet access
to the PyTorch weight servers in the sandbox this was built in). On a GPU
environment with pretrained weights and higher resolution, absolute numbers
would likely improve; the qualitative findings (coverage collapse under shift,
partial and expensive repair via weighting; model-vs-procedure distinction for
ordinal structure) are structural properties of the problem, not artifacts of
under-training, more or less.

- PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset
- PlantDoc: https://github.com/pratikkayal/PlantDoc-Dataset
- Coffee leaf severity: https://github.com/esgario/lara2018
- Tibshirani et al. (2019), *Conformal Prediction Under Covariate Shift*, NeurIPS
- Angelopoulos et al. (2020), *Uncertainty Sets for Image Classifiers Using
  Conformal Prediction*
- Cao, Mirjalili & Raschka (2020), *Rank Consistent Ordinal Regression for
  Neural Networks with Application to Age Estimation* (CORAL)
- Esgario, Krohling & Ventura (2020), *Deep Learning for Classification and
  Severity Estimation of Coffee Leaf Biotic Stress*, Computers and Electronics
  in Agriculture
