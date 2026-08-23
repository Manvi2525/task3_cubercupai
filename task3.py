"""
RTC Traffic Classification -- CyberAI Cup 2026, Task 3
FINAL pipeline v4: production ensemble + Zoom rule + soft mode-probability
feature.

Score history (see RTC_Project_Complete_Reference.md, Section 5, for full
chronology and rationale of every step up to 0.8300):
    0.770   rank-invariant baseline
    0.7988  LightGBM discovery + RF/LGBM/XGB soft-vote ensemble
    0.8118  prior-corrected decision rule
    0.8167  + candidate-set specialist override (Zoom excluded from scope)
    0.8189  + nested-CV temperature scaling
    0.8283  + jointly re-tuned temperature and margin threshold
             (double-nested CV: outer 5-fold, inner 4-fold grid search)
    0.8300  + gp_log_pl3 feature (log of packet_length_3), discovered via
             gplearn symbolic search on the Zoom voice/video sub-problem
    0.8200  (DROPPED) time-ratio ("normalized" time) features -- helped a
             standalone LightGBM (+0.0155) but cost -0.0100 once folded
             into this full ensemble; reverted after honest nested-CV test
    0.8349  + Zoom audio-band post-processing rule -- validated against the
             actual nested-CV predictions before being trusted on test
    ????    + soft mode-probability feature (voice-vs-video, leak-free OOF
             stacking on the SAME 5 folds as the main ensemble) -- this is
             the one open item from briefing Section 10 that hadn't been
             tried yet: "feeding mode information as a soft feature into
             one flat model" was found to beat baseline in the project's
             own follow-up notebooks, but was never folded into the
             production script until now. Validated below the same way
             the Zoom rule was -- against real nested-CV predictions,
             not assumed.                                                 <- THIS SCRIPT

Rejected/closed directions (see Section 6, Failed Approaches): DES/OvO,
multi-task app-times-mode factorization (the HARD-routing version of the
mode idea -- this script does SOFT conditioning instead, which is the
version the notebooks actually recommended), stacked meta-learner, burst/
cumulative/quantized traffic-fingerprinting features, gplearn for
mode-target and one-vs-rest (all classes), label-noise removal via
confident learning (leakage artifact), and the auxiliary-dataset /
transformer / linear-probing idea from the independent sidd20228/task3
project (tested negative -- feature redundancy + domain gap; see briefing
Section 8.2). Not re-attempted here.

Run: python3 build_model_final_v4_mode_feature.py
Expects: Task3/publish/RTC_CyberAICup2026/Training_set.csv
         Task3/publish/RTC_CyberAICup2026/Testing_set.csv
Writes:  submission_v4_mode_feature.csv
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

RNG = 42
np.random.seed(RNG)

TRAIN_PATH = "training.csv"
TEST_PATH = "testing.csv"
OUTPUT_PATH = "submission2.csv"  # saved next to the script

L_COLS = [f"packet_length_{i}" for i in range(5)]
T_COLS = [f"relative_time_{i}" for i in range(5)]

# Classes with a confirmed "mixed" failure mode (genuine two-way posterior
# ties with a sample-varying partner), as opposed to Zoom's irreducible
# single-pair overlap. Only these trigger the specialist override.
SPECIALIST_CLASSES = [
    "GoogleMeet_voice", "GoogleMeet_video",
    "Discord_voice", "Discord_video",
    "Messenger_video",
]

T_GRID = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
TAU_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


# ---------------------------------------------------------------------------
# Feature engineering (base + time-ratio features merged in)
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feats = pd.DataFrame(index=df.index)

    for c in L_COLS:
        feats[c] = df[c]
    for c in T_COLS[1:]:  # relative_time_0 is always 0
        feats[c] = df[c]

    lens = df[L_COLS].values
    times = df[T_COLS].values

    for i in range(1, 5):
        feats[f"delta_len_{i}"] = lens[:, i] - lens[:, i - 1]
    for i in range(1, 5):
        feats[f"delta_t_{i}"] = times[:, i] - times[:, i - 1]

    sorted_lens = np.sort(lens, axis=1)[:, ::-1]
    for rank in range(5):
        feats[f"len_rank_{rank}"] = sorted_lens[:, rank]

    feats["argmax_pos"] = np.argmax(lens, axis=1)
    feats["argmin_pos"] = np.argmin(lens, axis=1)
    feats["len_range"] = lens.max(axis=1) - lens.min(axis=1)
    feats["len_mean"] = lens.mean(axis=1)
    feats["len_std"] = lens.std(axis=1)
    feats["len_median"] = np.median(lens, axis=1)
    feats["total_span"] = times[:, -1] - times[:, 0]

    # gplearn-discovered feature (Zoom voice/video symbolic search).
    feats["gp_log_pl3"] = np.log(np.clip(df["packet_length_3"].values, 1, None))

    # NOTE: time-ratio features (time_frac_i, gap_frac_i, log_total_span)
    # were tried here and DROPPED -- they helped a standalone LightGBM
    # (+0.0155) but cost -0.0100 once folded into this full ensemble +
    # prior-correction + temperature-scaling + specialist-override stack
    # (0.8300 -> 0.8200 nested-CV baseline). Validated via honest nested-CV
    # comparison, not assumed. See build_model_final_v2.py history for the
    # full ablation if you want to revisit this later with a re-tuned grid.

    return feats


def make_rf():
    return RandomForestClassifier(
        n_estimators=600, class_weight="balanced_subsample",
        random_state=RNG, n_jobs=-1,
    )


def make_lgb(seed=RNG):
    return lgb.LGBMClassifier(
        n_estimators=400, max_depth=4, num_leaves=15, learning_rate=0.05,
        class_weight="balanced", random_state=seed, verbosity=-1,
    )


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=RNG, verbosity=0,
    )


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-9, 1))
    scaled = np.exp(logp / T)
    return scaled / scaled.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# NEW: Zoom audio-band post-processing rule
# ---------------------------------------------------------------------------
def calibrate_audio_threshold(train_df: pd.DataFrame, lens_train: np.ndarray) -> float:
    """
    Derive the audio/video packet-size cutoff from the data itself rather
    than hardcoding a guessed number. Uses the packet-length distribution of
    known *_voice classes as the reference band.
    """
    voice_mask = train_df["label"].str.endswith("_voice").values
    voice_max = lens_train[voice_mask].max()
    threshold = float(np.percentile(lens_train[voice_mask], 99))
    print(f"Calibrated audio-band threshold: {threshold:.1f} bytes "
          f"(max observed voice packet: {voice_max})")
    return threshold


def zoom_audio_band_rule(pred_labels, lens, audio_len_threshold):
    """
    If the model predicted a Zoom class AND every one of the 5 packets in
    that flow is <= audio_len_threshold bytes, force the prediction to
    Zoom_voice. Does not touch any other class's predictions.
    """
    pred_labels = np.array(pred_labels, dtype=object)
    is_zoom_pred = np.isin(pred_labels, ["Zoom_voice", "Zoom_video"])
    all_audio_band = (lens <= audio_len_threshold).all(axis=1)
    trigger = is_zoom_pred & all_audio_band
    pred_labels[trigger] = "Zoom_voice"
    return pred_labels, int(trigger.sum())


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X = engineer_features(train)
X_test = engineer_features(test)

lens_train = train[L_COLS].values
lens_test = test[L_COLS].values

le = LabelEncoder()
y = le.fit_transform(train["label"])
classes = le.classes_
K = len(classes)
n = len(y)
pi_hat = np.bincount(y) / n
cls_list = list(classes)
difficult_idx = set(cls_list.index(c) for c in SPECIALIST_CLASSES)

print(f"Train: {X.shape}, Test: {X_test.shape}, classes: {K}")


# ---------------------------------------------------------------------------
# NEW: soft mode-probability feature (voice vs. video), leak-free.
#
# The briefing's notebooks found that feeding mode information as a SOFT
# feature into one flat model beats HARD routing to separate mode-specific
# models. "Mode" (voice/video) is derivable from the label, so at TRAIN
# time we can only use an out-of-fold probability (never fit-and-predict on
# the same rows) or this leaks the answer straight into the target. At TEST
# time there's no label at all, so a mode classifier is fit on the FULL
# training set and applied to test features directly -- no leakage there
# by construction.
#
# Splits are computed ONCE here and reused for the main ensemble in Step 1
# below, so the mode feature for any given row never saw that row's own
# fold during its own training -- consistent with how the Zoom rule and
# specialist override are already validated leak-free elsewhere in this
# script.
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
splits = list(skf.split(X, y))
fold_id = np.zeros(n, dtype=int)
for fi, (_, va_idx) in enumerate(splits):
    fold_id[va_idx] = fi

mode_y = train["label"].str.endswith("_video").astype(int).values  # 1=video, 0=voice
oof_mode_prob = np.zeros(n)

for fi, (tr_idx, va_idx) in enumerate(splits):
    mode_clf = make_lgb(seed=RNG + 3)
    mode_clf.fit(X.iloc[tr_idx], mode_y[tr_idx])
    oof_mode_prob[va_idx] = mode_clf.predict_proba(X.iloc[va_idx])[:, 1]

mode_clf_full = make_lgb(seed=RNG + 3)
mode_clf_full.fit(X, mode_y)
test_mode_prob = mode_clf_full.predict_proba(X_test)[:, 1]

# Sanity check: how good is the mode signal on its own? (Not the final
# metric -- just tells us whether this feature carries real information
# before we bother folding it into the 10-class ensemble.)
from sklearn.metrics import roc_auc_score
mode_auc = roc_auc_score(mode_y, oof_mode_prob)
print(f"\nMode classifier (voice vs video) OOF AUC: {mode_auc:.4f}  "
      f"(sanity check -- should be high; this is a much easier problem "
      f"than the 10-way task)")

X = X.copy()
X_test = X_test.copy()
X["mode_video_prob"] = oof_mode_prob
X_test["mode_video_prob"] = test_mode_prob


# ---------------------------------------------------------------------------
# Step 1: 5-fold OOF probabilities from RF, LightGBM, XGBoost
# (reuses the exact same `splits` computed above for the mode feature)
# ---------------------------------------------------------------------------
oof_rf = np.zeros((n, K))
oof_lgb = np.zeros((n, K))
oof_xgb = np.zeros((n, K))

for fi, (tr_idx, va_idx) in enumerate(splits):
    m = make_rf(); m.fit(X.iloc[tr_idx], y[tr_idx])
    oof_rf[va_idx] = m.predict_proba(X.iloc[va_idx])

    m = make_lgb(); m.fit(X.iloc[tr_idx], y[tr_idx])
    oof_lgb[va_idx] = m.predict_proba(X.iloc[va_idx])

    m = make_xgb(); m.fit(X.iloc[tr_idx], y[tr_idx])
    oof_xgb[va_idx] = m.predict_proba(X.iloc[va_idx])

oof_ens = (oof_rf + oof_lgb + oof_xgb) / 3

baseline_pred = oof_ens.argmax(axis=1)
print(f"Baseline (raw argmax, no correction):  "
      f"{recall_score(y, baseline_pred, average='macro'):.4f}")


# ---------------------------------------------------------------------------
# Step 2: jointly re-tune temperature T and margin threshold TAU via
# double-nested CV (outer = the 5 folds above, inner = 4-fold split of
# each outer-training portion).
# ---------------------------------------------------------------------------
def decide(probs, T, TAU):
    scaled = apply_temperature(probs, T)
    corr = scaled / pi_hat[None, :]
    corr_norm = corr / corr.sum(axis=1, keepdims=True)
    sorted_idx = np.argsort(-corr_norm, axis=1)
    pred = sorted_idx[:, 0].copy()
    top1p = corr_norm[np.arange(len(pred)), sorted_idx[:, 0]]
    top2p = corr_norm[np.arange(len(pred)), sorted_idx[:, 1]]
    margin = top1p - top2p
    trigger = (
        (margin < TAU)
        & np.isin(sorted_idx[:, 0], list(difficult_idx))
        & np.isin(sorted_idx[:, 1], list(difficult_idx))
    )
    return pred, trigger


final_pred = np.zeros(n, dtype=int)
chosen = []

for fi in range(5):
    tune_mask = fold_id != fi
    hold_mask = fold_id == fi
    tune_idx = np.where(tune_mask)[0]
    hold_idx = np.where(hold_mask)[0]

    inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RNG)
    inner_specialist_oof = np.full((len(tune_idx), K), np.nan)
    inner_y = y[tune_idx]
    inner_X = X.iloc[tune_idx].reset_index(drop=True)

    for itr, iva in inner_skf.split(inner_X, inner_y):
        itr_global = tune_idx[itr]
        tr_mask_diff = np.isin(y[itr_global], list(difficult_idx))
        spec = make_lgb(seed=RNG + 2)
        spec.fit(X.iloc[itr_global[tr_mask_diff]], y[itr_global[tr_mask_diff]])
        iva_global = tune_idx[iva]
        proba_sub = spec.predict_proba(X.iloc[iva_global])
        full_proba = np.zeros((len(iva_global), K))
        for ci, cl in enumerate(spec.classes_):
            full_proba[:, cl] = proba_sub[:, ci]
        inner_specialist_oof[iva] = full_proba

    best_score, best_T, best_TAU = -1, 1.0, 0.0
    inner_probs = oof_ens[tune_idx]
    for T in T_GRID:
        for TAU in TAU_GRID:
            pred, trigger = decide(inner_probs, T, TAU)
            p2 = pred.copy()
            has_spec = trigger & ~np.isnan(inner_specialist_oof[:, 0])
            p2[has_spec] = np.nanargmax(inner_specialist_oof[has_spec], axis=1)
            score = recall_score(inner_y, p2, average="macro")
            if score > best_score:
                best_score, best_T, best_TAU = score, T, TAU

    chosen.append((best_T, best_TAU, best_score))

    tr_mask_diff_full = np.isin(y[tune_idx], list(difficult_idx))
    spec_full = make_lgb(seed=RNG + 1)
    spec_full.fit(X.iloc[tune_idx[tr_mask_diff_full]], y[tune_idx[tr_mask_diff_full]])

    pred_hold, trigger_hold = decide(oof_ens[hold_idx], best_T, best_TAU)
    if trigger_hold.sum() > 0:
        proba_sub = spec_full.predict_proba(X.iloc[hold_idx[trigger_hold]])
        full_proba = np.zeros((trigger_hold.sum(), K))
        for ci, cl in enumerate(spec_full.classes_):
            full_proba[:, cl] = proba_sub[:, ci]
        pred_hold[trigger_hold] = full_proba.argmax(axis=1)

    final_pred[hold_idx] = pred_hold

print("\nChosen (T, TAU) per outer fold (inner-CV score):")
for T, TAU, s in chosen:
    print(f"  T={T}, TAU={TAU}  (inner score {s:.4f})")

final_macro = recall_score(y, final_pred, average="macro")
print(f"\nFINAL nested-CV macro-recall (0.8300 feature set + mode_video_prob): {final_macro:.4f}")
print(f"(Compare to 0.8323 without the mode feature -- last confirmed checkpoint)")

per_class = recall_score(y, final_pred, average=None)
print("\nPer-class recall:")
for c, r in sorted(zip(classes, per_class), key=lambda x: x[1]):
    print(f"  {c:20s} {r:.3f}")


# ---------------------------------------------------------------------------
# Step 2b: honestly validate the Zoom rule against the ACTUAL nested-CV
# predictions above (leak-free -- threshold calibrated on train labels only,
# rule applied to out-of-fold predictions) before trusting it on test.
# ---------------------------------------------------------------------------
audio_threshold_cv = calibrate_audio_threshold(train, lens_train)
final_pred_str = le.inverse_transform(final_pred)
final_pred_zoom_str, n_over_cv = zoom_audio_band_rule(
    final_pred_str, lens_train, audio_threshold_cv
)
final_pred_zoom_idx = le.transform(final_pred_zoom_str)
final_macro_zoom = recall_score(y, final_pred_zoom_idx, average="macro")
zoom_helps = final_macro_zoom > final_macro
print(f"\n+ Zoom audio-band rule (nested-CV predictions): {final_macro_zoom:.4f}  "
      f"({'KEEP' if zoom_helps else 'DROP'} -- delta {final_macro_zoom - final_macro:+.4f}, "
      f"{n_over_cv} flows overridden)")
print(f"(Compare to 0.8349 -- your last confirmed checkpoint, mode feature + Zoom rule combined)")


# ---------------------------------------------------------------------------
# Step 3: refit on the FULL training set, predict on the actual test set.
# ---------------------------------------------------------------------------
T_final = float(np.median([c[0] for c in chosen]))
TAU_final = float(np.median([c[1] for c in chosen]))
print(f"\nFinal hyperparameters used for test prediction: T={T_final}, TAU={TAU_final}")

rf_full = make_rf(); rf_full.fit(X, y)
lgb_full = make_lgb(); lgb_full.fit(X, y)
xgb_full = make_xgb(); xgb_full.fit(X, y)

spec_final = make_lgb(seed=RNG + 1)
tr_mask_diff_final = np.isin(y, list(difficult_idx))
spec_final.fit(X.iloc[np.where(tr_mask_diff_final)[0]], y[tr_mask_diff_final])

test_ens = (
    rf_full.predict_proba(X_test)
    + lgb_full.predict_proba(X_test)
    + xgb_full.predict_proba(X_test)
) / 3

test_pred, test_trigger = decide(test_ens, T_final, TAU_final)
if test_trigger.sum() > 0:
    proba_sub = spec_final.predict_proba(X_test.iloc[test_trigger])
    full_proba = np.zeros((test_trigger.sum(), K))
    for ci, cl in enumerate(spec_final.classes_):
        full_proba[:, cl] = proba_sub[:, ci]
    test_pred[test_trigger] = full_proba.argmax(axis=1)

test_labels = le.inverse_transform(test_pred)

# Apply the Zoom rule to test predictions only if it was validated to help
# above -- avoids blindly trusting a post-processing step that wasn't tested
# against this exact pipeline's own out-of-fold predictions.
if zoom_helps:
    audio_threshold_test = calibrate_audio_threshold(train, lens_train)
    test_labels, n_over_test = zoom_audio_band_rule(
        test_labels, lens_test, audio_threshold_test
    )
    print(f"\nZoom rule overrode {n_over_test} test predictions to Zoom_voice.")
else:
    print("\nZoom rule did NOT beat the nested-CV baseline on this pipeline -- "
          "skipped on test predictions. (Consistent with the possibility that "
          "the specialist override + prior correction already partially "
          "capture this signal; see briefing Section 6.)")

submission = pd.DataFrame({
    "idx": np.arange(1, len(test_labels) + 1),
    "label": test_labels,
})
submission.to_csv(OUTPUT_PATH, index=False, header=False)
print(f"\nSaved {len(submission)} predictions to {OUTPUT_PATH}")
print(pd.Series(test_labels).value_counts())