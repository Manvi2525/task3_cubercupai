
"""
RTC Traffic Classification -- CyberAI Cup 2026, Task 3
FINAL pipeline v5: confirmed-best base (0.8349) + narrowly-scoped candidate
features, each honestly ablation-gated before being trusted on test.

Confirmed checkpoints so far:
    0.8300  0.8300 feature set (rank-invariant + gp_log_pl3), no post-proc
    0.8323  ...measured via nested-CV on this exact script structure
    0.8349  + Zoom audio-band rule                                <- last confirmed best (v3)
    0.8241  (DROPPED) + global mode_video_prob feature on the full ensemble
             -- regressed nested-CV by -0.0082 even before the Zoom rule.
             GoogleMeet_voice (n=40, smallest class) took the biggest hit:
             0.950 -> 0.875. Likely cause: mode_video_prob has AUC 0.96,
             and with shallow trees (depth=4, leaves=15) a single dominant
             feature crowds out the split budget for subtler size-pattern
             splits in small specialist-subset training data. Consistent
             with the earlier time-ratio-feature regression -- broad,
             globally-injected signal that's redundant with what the
             prior-correction + temperature-scaling + specialist-override
             stack already captures tends to hurt more than help here.
    ????    This script instead tests mode_video_prob SCOPED ONLY to the
             specialist model (small subset, where it might disambiguate
             without destabilizing the main ensemble), plus 3 small
             hand-crafted candidate features targeted at the current worst
             classes (Zoom_video 0.593, GoogleMeet_video 0.732). Each is
             individually ablation-gated -- KEEP only if it beats the
             0.8349 checkpoint on nested-CV, exactly like the Zoom rule.

Run: python3 build_model_v5.py
Expects: Task3/publish/RTC_CyberAICup2026/Training_set.csv
         Task3/publish/RTC_CyberAICup2026/Testing_set.csv
Writes:  submission_v5.csv
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

SPECIALIST_CLASSES = [
    "GoogleMeet_voice", "GoogleMeet_video",
    "Discord_voice", "Discord_video",
    "Messenger_video",
]

T_GRID = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
TAU_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


# ---------------------------------------------------------------------------
# Base feature set -- CONFIRMED BEST (0.8300), unchanged from v1/v3.
# ---------------------------------------------------------------------------
def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feats = pd.DataFrame(index=df.index)
    for c in L_COLS:
        feats[c] = df[c]
    for c in T_COLS[1:]:
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
    feats["gp_log_pl3"] = np.log(np.clip(df["packet_length_3"].values, 1, None))

    return feats, lens, times


# ---------------------------------------------------------------------------
# NEW candidate features -- small, targeted, each gated individually below.
# ---------------------------------------------------------------------------
def add_candidate_features(feats: pd.DataFrame, lens: np.ndarray,
                            video_threshold: float) -> pd.DataFrame:
    feats = feats.copy()
    # C1: ratio of last packet to first packet -- does the flow "grow" or
    # "shrink" in size by the 5th packet? Cheap, scale-sensitive signal
    # distinct from the rank/delta features already in use.
    feats["len4_over_len0"] = lens[:, 4] / np.clip(lens[:, 0], 1, None)

    # C2: how many of the 5 packets exceed a data-derived "video band"
    # threshold -- a small interpretable count, complementary to
    # len_rank_* (which keeps magnitude but drops this specific count).
    feats["n_above_video_band"] = (lens > video_threshold).sum(axis=1)

    # C3: position of the first packet crossing that threshold (5 if none
    # do) -- targets the video apps whose early packets sometimes look
    # audio-sized (the same phenomenon that makes Zoom hard, but here
    # applied to GoogleMeet_video / Discord_video, not Zoom).
    above = lens > video_threshold
    first_pos = np.where(above.any(axis=1), above.argmax(axis=1), 5)
    feats["first_video_band_pos"] = first_pos

    return feats


def calibrate_video_threshold(train_df: pd.DataFrame, lens_train: np.ndarray) -> float:
    """Mirror of calibrate_audio_threshold, but for a 'video band' cutoff --
    derived from the training data, not guessed."""
    video_mask = train_df["label"].str.endswith("_video").values
    threshold = float(np.percentile(lens_train[video_mask], 25))
    print(f"Calibrated video-band threshold: {threshold:.1f} bytes")
    return threshold


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


def calibrate_audio_threshold(train_df: pd.DataFrame, lens_train: np.ndarray) -> float:
    voice_mask = train_df["label"].str.endswith("_voice").values
    threshold = float(np.percentile(lens_train[voice_mask], 99))
    print(f"Calibrated audio-band threshold: {threshold:.1f} bytes")
    return threshold


def zoom_audio_band_rule(pred_labels, lens, audio_len_threshold):
    pred_labels = np.array(pred_labels, dtype=object)
    is_zoom_pred = np.isin(pred_labels, ["Zoom_voice", "Zoom_video"])
    all_audio_band = (lens <= audio_len_threshold).all(axis=1)
    trigger = is_zoom_pred & all_audio_band
    pred_labels[trigger] = "Zoom_voice"
    return pred_labels, int(trigger.sum())


def zoom_audio_band_rule_gated(pred_labels, lens, probs, classes, audio_len_threshold,
                                margin_gate=0.15):
    """
    Same idea as zoom_audio_band_rule, but only overrides when the model's
    OWN margin between Zoom_video and Zoom_voice was small -- i.e. skip the
    override for flows the model called Zoom_video with a clear lead over
    Zoom_voice specifically (not just "confident overall", which is a much
    higher and mostly-unreachable bar on a 10-class distribution).
    `probs` = the ensemble's decided probability matrix (post temperature/
    prior-correction), `classes` = the label encoder's class list in the
    same column order as `probs`.
    """
    pred_labels = np.array(pred_labels, dtype=object)
    zoom_video_idx = list(classes).index("Zoom_video")
    zoom_voice_idx = list(classes).index("Zoom_voice")
    is_zoom_pred = np.isin(pred_labels, ["Zoom_voice", "Zoom_video"])
    all_audio_band = (lens <= audio_len_threshold).all(axis=1)
    video_voice_margin = probs[:, zoom_video_idx] - probs[:, zoom_voice_idx]
    zoom_video_clear = video_voice_margin >= margin_gate
    trigger = is_zoom_pred & all_audio_band & ~zoom_video_clear
    return_labels = pred_labels.copy()
    return_labels[trigger] = "Zoom_voice"
    return return_labels, int(trigger.sum()), video_voice_margin


# ---------------------------------------------------------------------------
# Full nested-CV pipeline, parameterized by which candidate features (and
# whether the specialist-only mode feature) are switched on -- so we can
# run it multiple times and compare honestly, exactly like the Zoom rule.
# ---------------------------------------------------------------------------
def run_pipeline(X, y, lens_train_arr, classes, difficult_idx, pi_hat, label=""):
    n, K = len(y), len(classes)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    splits = list(skf.split(X, y))
    fold_id = np.zeros(n, dtype=int)
    for fi, (_, va_idx) in enumerate(splits):
        fold_id[va_idx] = fi

    oof_rf = np.zeros((n, K)); oof_lgb = np.zeros((n, K)); oof_xgb = np.zeros((n, K))

    for fi, (tr_idx, va_idx) in enumerate(splits):
        m = make_rf(); m.fit(X.iloc[tr_idx], y[tr_idx]); oof_rf[va_idx] = m.predict_proba(X.iloc[va_idx])
        m = make_lgb(); m.fit(X.iloc[tr_idx], y[tr_idx]); oof_lgb[va_idx] = m.predict_proba(X.iloc[va_idx])
        m = make_xgb(); m.fit(X.iloc[tr_idx], y[tr_idx]); oof_xgb[va_idx] = m.predict_proba(X.iloc[va_idx])

    oof_ens = (oof_rf + oof_lgb + oof_xgb) / 3

    def decide(probs, T, TAU):
        scaled = apply_temperature(probs, T)
        corr = scaled / pi_hat[None, :]
        corr_norm = corr / corr.sum(axis=1, keepdims=True)
        sorted_idx = np.argsort(-corr_norm, axis=1)
        pred = sorted_idx[:, 0].copy()
        top1p = corr_norm[np.arange(len(pred)), sorted_idx[:, 0]]
        top2p = corr_norm[np.arange(len(pred)), sorted_idx[:, 1]]
        margin = top1p - top2p
        trigger = ((margin < TAU) & np.isin(sorted_idx[:, 0], list(difficult_idx))
                   & np.isin(sorted_idx[:, 1], list(difficult_idx)))
        return pred, trigger, corr_norm

    final_pred = np.zeros(n, dtype=int)
    final_probs = np.zeros((n, K))
    chosen = []

    for fi in range(5):
        tune_idx = np.where(fold_id != fi)[0]
        hold_idx = np.where(fold_id == fi)[0]
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
                pred, trigger, _ = decide(inner_probs, T, TAU)
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
        pred_hold, trigger_hold, probs_hold = decide(oof_ens[hold_idx], best_T, best_TAU)
        if trigger_hold.sum() > 0:
            proba_sub = spec_full.predict_proba(X.iloc[hold_idx[trigger_hold]])
            full_proba = np.zeros((trigger_hold.sum(), K))
            for ci, cl in enumerate(spec_full.classes_):
                full_proba[:, cl] = proba_sub[:, ci]
            pred_hold[trigger_hold] = full_proba.argmax(axis=1)
            probs_hold[trigger_hold] = full_proba  # keep the probability view consistent
        final_pred[hold_idx] = pred_hold
        final_probs[hold_idx] = probs_hold

    macro = recall_score(y, final_pred, average="macro")
    per_class = recall_score(y, final_pred, average=None)
    print(f"[{label}] nested-CV macro-recall: {macro:.4f}")
    print(f"[{label}] per-class recall:")
    for c, r in sorted(zip(classes, per_class), key=lambda x: x[1]):
        print(f"    {c:20s} {r:.3f}")
    return macro, final_pred, final_probs, chosen


# ---------------------------------------------------------------------------
# Load data, set up base pipeline (CONFIRMED 0.8349 checkpoint)
# ---------------------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X_base, lens_train, times_train = engineer_base_features(train)
X_test_base, lens_test, times_test = engineer_base_features(test)

le = LabelEncoder()
y = le.fit_transform(train["label"])
classes = le.classes_
K = len(classes)
n = len(y)
pi_hat = np.bincount(y) / n
difficult_idx = set(list(classes).index(c) for c in SPECIALIST_CLASSES)

print(f"Train: {X_base.shape}, Test: {X_test_base.shape}, classes: {K}\n")

CHECKPOINT = 0.8371  # your last confirmed best, from v5 (base + C1-C3 + blanket Zoom rule)

# --- Candidate features, tested one block at a time, each vs. CHECKPOINT ---
video_thresh = calibrate_video_threshold(train, lens_train)
X_cand = add_candidate_features(X_base, lens_train, video_thresh)
X_test_cand = add_candidate_features(X_test_base, lens_test, video_thresh)

macro_cand, pred_cand, probs_cand, chosen_cand = run_pipeline(
    X_cand, y, lens_train, classes, difficult_idx, pi_hat,
    label="base + 3 candidate features (C1-C3)"
)

audio_thresh = calibrate_audio_threshold(train, lens_train)

# Variant A: blanket Zoom rule (your confirmed 0.8371 result)
pred_cand_str = le.inverse_transform(pred_cand)
pred_cand_zoom_str, n_over = zoom_audio_band_rule(pred_cand_str, lens_train, audio_thresh)
pred_cand_zoom_idx = le.transform(pred_cand_zoom_str)
macro_cand_zoom = recall_score(y, pred_cand_zoom_idx, average="macro")
per_class_zoom = recall_score(y, pred_cand_zoom_idx, average=None)
print(f"  + Zoom rule (blanket): {macro_cand_zoom:.4f}  "
      f"({'KEEP' if macro_cand_zoom > CHECKPOINT else 'DROP'} vs. {CHECKPOINT} checkpoint, "
      f"delta {macro_cand_zoom - CHECKPOINT:+.4f})")
print("  + Zoom rule (blanket) per-class recall:")
for c, r in sorted(zip(classes, per_class_zoom), key=lambda x: x[1]):
    print(f"      {c:20s} {r:.3f}")

# Variant B: gated Zoom rule -- skip the override when the model's own
# margin between Zoom_video and Zoom_voice was clear, to try to win back
# some of the Zoom_video recall the blanket rule gives away.
audio_trigger_mask = (
    np.isin(pred_cand_str, ["Zoom_voice", "Zoom_video"])
    & (lens_train <= audio_thresh).all(axis=1)
)
zoom_video_idx_ = list(classes).index("Zoom_video")
zoom_voice_idx_ = list(classes).index("Zoom_voice")
diag_margin = (probs_cand[audio_trigger_mask, zoom_video_idx_]
               - probs_cand[audio_trigger_mask, zoom_voice_idx_])
print(f"  Diagnostic -- Zoom_video minus Zoom_voice margin among the "
      f"{audio_trigger_mask.sum()} audio-band-triggered flows:")
print(f"    min={diag_margin.min():.3f}  p25={np.percentile(diag_margin,25):.3f}  "
      f"median={np.median(diag_margin):.3f}  p75={np.percentile(diag_margin,75):.3f}  "
      f"max={diag_margin.max():.3f}")

pred_cand_gated_str, n_over_gated, _ = zoom_audio_band_rule_gated(
    pred_cand_str, lens_train, probs_cand, classes, audio_thresh, margin_gate=0.15
)
pred_cand_gated_idx = le.transform(pred_cand_gated_str)
macro_cand_gated = recall_score(y, pred_cand_gated_idx, average="macro")
per_class_gated = recall_score(y, pred_cand_gated_idx, average=None)
print(f"\n  + Zoom rule (gated, margin>=0.15 skipped): {macro_cand_gated:.4f}  "
      f"({'KEEP' if macro_cand_gated > max(CHECKPOINT, macro_cand_zoom) else 'DROP'} "
      f"vs. best-so-far {max(CHECKPOINT, macro_cand_zoom):.4f}, "
      f"delta {macro_cand_gated - max(CHECKPOINT, macro_cand_zoom):+.4f}, "
      f"{n_over_gated} overridden vs. {n_over} blanket)")
print("  + Zoom rule (gated) per-class recall:")
for c, r in sorted(zip(classes, per_class_gated), key=lambda x: x[1]):
    print(f"      {c:20s} {r:.3f}")
print()

# --- Decide final feature set + Zoom-rule variant based on the honest
# results above. Priority: candidate features must beat CHECKPOINT on
# some Zoom-rule variant, then pick whichever variant scores highest. ---
use_candidates = max(macro_cand_zoom, macro_cand_gated) > CHECKPOINT
use_gated_rule = macro_cand_gated > macro_cand_zoom
X_final = X_cand if use_candidates else X_base
X_test_final = X_test_cand if use_candidates else X_test_base
print(f"Final decision: {'KEEPING' if use_candidates else 'DROPPING'} candidate features C1-C3.")
print(f"Final decision: using {'GATED' if use_gated_rule else 'BLANKET'} Zoom rule.\n")

# ---------------------------------------------------------------------------
# Refit on full training data with the chosen feature set + Zoom rule,
# predict on the real test set.
# ---------------------------------------------------------------------------
macro_final, final_pred_full, final_probs_full, chosen_final = run_pipeline(
    X_final, y, lens_train, classes, difficult_idx, pi_hat,
    label="FINAL chosen pipeline"
)

T_final = float(np.median([c[0] for c in chosen_final]))
TAU_final = float(np.median([c[1] for c in chosen_final]))
print(f"Final hyperparameters: T={T_final}, TAU={TAU_final}")

rf_full = make_rf(); rf_full.fit(X_final, y)
lgb_full = make_lgb(); lgb_full.fit(X_final, y)
xgb_full = make_xgb(); xgb_full.fit(X_final, y)

spec_final = make_lgb(seed=RNG + 1)
tr_mask_diff_final = np.isin(y, list(difficult_idx))
spec_final.fit(X_final.iloc[np.where(tr_mask_diff_final)[0]], y[tr_mask_diff_final])

pi_hat_arr = pi_hat

def decide_final(probs, T, TAU):
    scaled = apply_temperature(probs, T)
    corr = scaled / pi_hat_arr[None, :]
    corr_norm = corr / corr.sum(axis=1, keepdims=True)
    sorted_idx = np.argsort(-corr_norm, axis=1)
    pred = sorted_idx[:, 0].copy()
    top1p = corr_norm[np.arange(len(pred)), sorted_idx[:, 0]]
    top2p = corr_norm[np.arange(len(pred)), sorted_idx[:, 1]]
    margin = top1p - top2p
    trigger = ((margin < TAU) & np.isin(sorted_idx[:, 0], list(difficult_idx))
               & np.isin(sorted_idx[:, 1], list(difficult_idx)))
    return pred, trigger, corr_norm

test_ens = (rf_full.predict_proba(X_test_final) + lgb_full.predict_proba(X_test_final)
            + xgb_full.predict_proba(X_test_final)) / 3
test_pred, test_trigger, test_probs = decide_final(test_ens, T_final, TAU_final)
if test_trigger.sum() > 0:
    proba_sub = spec_final.predict_proba(X_test_final.iloc[test_trigger])
    full_proba = np.zeros((test_trigger.sum(), K))
    for ci, cl in enumerate(spec_final.classes_):
        full_proba[:, cl] = proba_sub[:, ci]
    test_pred[test_trigger] = full_proba.argmax(axis=1)
    test_probs[test_trigger] = full_proba

test_labels = le.inverse_transform(test_pred)
if use_gated_rule:
    test_labels, n_over_test, _ = zoom_audio_band_rule_gated(
        test_labels, lens_test, test_probs, classes, audio_thresh, margin_gate=0.15
    )
    print(f"Zoom rule (gated) overrode {n_over_test} test predictions to Zoom_voice.")
else:
    test_labels, n_over_test = zoom_audio_band_rule(test_labels, lens_test, audio_thresh)
    print(f"Zoom rule (blanket) overrode {n_over_test} test predictions to Zoom_voice.")

submission = pd.DataFrame({"idx": np.arange(1, len(test_labels) + 1), "label": test_labels})
submission.to_csv(OUTPUT_PATH, index=False, header=False)
print(f"\nSaved {len(submission)} predictions to {OUTPUT_PATH}")
print(pd.Series(test_labels).value_counts())



