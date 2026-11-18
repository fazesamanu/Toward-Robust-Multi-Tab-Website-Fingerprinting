import torch
import random
import numpy as np
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import roc_auc_score, precision_recall_curve
except (ImportError, OSError):
    def _rankdata_avg(a):
        a = np.asarray(a, dtype=np.float64)
        sorter = np.argsort(a, kind="mergesort")
        inv = np.empty(sorter.size, dtype=np.intp)
        inv[sorter] = np.arange(sorter.size, dtype=np.intp)
        arr = a[sorter]
        obs = np.r_[True, arr[1:] != arr[:-1]]
        dense = obs.cumsum()[inv]
        count = np.r_[np.nonzero(obs)[0], len(obs)]
        ranks = 0.5 * (count[dense] + count[dense - 1] + 1)
        return ranks

    def _binary_roc_auc(y_t, y_s):
        y_t = (np.asarray(y_t) > 0.5).astype(np.int64)
        n_pos = int(np.sum(y_t == 1))
        n_neg = int(np.sum(y_t == 0))
        if n_pos == 0 or n_neg == 0:
            return 0.5
        ranks = _rankdata_avg(y_s)
        sum_pos = float(np.sum(ranks[y_t == 1]))
        return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    def roc_auc_score(y_true, y_score, average="macro"):
        yt = np.asarray(y_true)
        ys = np.asarray(y_score)
        if yt.ndim == 1:
            return _binary_roc_auc(yt, ys)
        aucs = [_binary_roc_auc(yt[:, c], ys[:, c]) for c in range(yt.shape[1])]
        return float(np.mean(aucs)) if average == "macro" else np.array(aucs, dtype=np.float64)

    def accuracy_score(y_true, y_pred):
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        if yt.ndim == 2:
            return float(np.mean(np.all(yt == yp, axis=1)))
        return float(np.mean(yt == yp))

    def _prf_per_class(y_true, y_pred):
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        if yt.ndim == 1:
            labels = np.unique(np.concatenate([yt, yp]))
            tp = np.array([np.sum((yt == c) & (yp == c)) for c in labels], dtype=np.float64)
            fp = np.array([np.sum((yt != c) & (yp == c)) for c in labels], dtype=np.float64)
            fn = np.array([np.sum((yt == c) & (yp != c)) for c in labels], dtype=np.float64)
        else:
            tp = np.sum((yt == 1) & (yp == 1), axis=0).astype(np.float64)
            fp = np.sum((yt == 0) & (yp == 1), axis=0).astype(np.float64)
            fn = np.sum((yt == 1) & (yp == 0), axis=0).astype(np.float64)
        return tp, fp, fn

    def precision_score(y_true, y_pred, average="macro", zero_division=0):
        tp, fp, _ = _prf_per_class(y_true, y_pred)
        denom = tp + fp
        p = np.where(denom > 0, tp / np.where(denom > 0, denom, 1.0), float(zero_division))
        return p if average is None else float(np.mean(p))

    def recall_score(y_true, y_pred, average="macro", zero_division=0):
        tp, _, fn = _prf_per_class(y_true, y_pred)
        denom = tp + fn
        r = np.where(denom > 0, tp / np.where(denom > 0, denom, 1.0), float(zero_division))
        return r if average is None else float(np.mean(r))

    def f1_score(y_true, y_pred, average="macro", zero_division=0):
        p = precision_score(y_true, y_pred, average=None, zero_division=zero_division)
        r = recall_score(y_true, y_pred, average=None, zero_division=zero_division)
        denom = p + r
        f = np.where(denom > 0, 2.0 * p * r / np.where(denom > 0, denom, 1.0), float(zero_division))
        return f if average is None else float(np.mean(f))

    def precision_recall_curve(y_true, probas_pred):
        yt = (np.asarray(y_true) > 0.5).astype(np.int64)
        ys = np.asarray(probas_pred, dtype=np.float64)
        order = np.argsort(ys)[::-1]
        ys_s = ys[order]
        yt_s = yt[order]
        tps = np.cumsum(yt_s)
        fps = np.cumsum(1 - yt_s)
        total_pos = max(1, int(tps[-1]))
        prec = tps / (tps + fps)
        rec = tps / total_pos
        return np.r_[prec[::-1], 1.0], np.r_[rec[::-1], 0.0], ys_s[::-1]


def precision_at_k(y_true, y_pred, k):
    """
    Calculate P@k (Section V-A, Equation 10):
    P@k = (1 / k) * sum_{l in r_k(y_hat)} y_l
    """
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    rows = np.arange(y_true.shape[0])[:, None]
    true_positives = np.sum(y_true[rows, top_k_preds], axis=1)
    return float(np.mean(true_positives / k))


def average_precision_at_k(y_true, y_pred, k):
    """
    Calculate MAP@k (Section V-A, Equation 11):
    MAP@k = (1 / k) * sum_{i=1}^k P@i
    """
    res = 0.0
    for i in range(k):
        res += precision_at_k(y_true, y_pred, i + 1)
    return float(res / k)


def safe_macro_auc(y_true, y_pred):
    """
    Compute macro-averaged ROC-AUC across websites, skipping constant columns if any.
    """
    valid_cols = [c for c in range(y_true.shape[1]) if len(np.unique(y_true[:, c])) > 1]
    if not valid_cols:
        return 0.5
    return float(roc_auc_score(y_true[:, valid_cols], y_pred[:, valid_cols], average="macro"))


def compute_open_world_auc(y_true, y_pred, num_monitored=100):
    """
    Compute AUC_M (monitored websites) and AUC_N (non-monitored website category)
    for the Open-World scenario (Section V-C, Figure 8).
    """
    y_true_m = y_true[:, :num_monitored]
    y_pred_m = y_pred[:, :num_monitored]
    auc_m = safe_macro_auc(y_true_m, y_pred_m)

    if y_true.shape[1] > num_monitored:
        y_true_n = y_true[:, num_monitored:]
        y_pred_n = y_pred[:, num_monitored:]
        auc_n = safe_macro_auc(y_true_n, y_pred_n)
    else:
        # Derive non-monitored confidence from complement of monitored probabilities
        has_non_monitored = (np.sum(y_true_m, axis=1) < np.max(np.sum(y_true_m, axis=1))).astype(np.float32)
        if len(np.unique(has_non_monitored)) > 1:
            non_mon_score = 1.0 - np.mean(np.sort(y_pred_m, axis=1)[:, -2:], axis=1)
            auc_n = float(roc_auc_score(has_non_monitored, non_mon_score))
        else:
            auc_n = auc_m
    return round(auc_m, 4), round(auc_n, 4)


def compute_pr_curve_macro(y_true, y_pred, num_thresholds=50):
    """
    Compute average Precision and Recall across all monitored websites at varying
    thresholds in [0, 1] (Section V-B, Figure 7).
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    precisions = []
    recalls = []
    for tau in thresholds:
        pred_bin = (y_pred >= tau).astype(np.float32)
        tp = np.sum((pred_bin == 1) & (y_true == 1), axis=0)
        fp = np.sum((pred_bin == 1) & (y_true == 0), axis=0)
        fn = np.sum((pred_bin == 0) & (y_true == 1), axis=0)
        prec_c = np.where(tp + fp > 0, tp / (tp + fp + 1e-12), 1.0)
        rec_c = np.where(tp + fn > 0, tp / (tp + fn + 1e-12), 0.0)
        precisions.append(float(np.mean(prec_c)))
        recalls.append(float(np.mean(rec_c)))
    # Ensure endpoints (recall=0, precision=1) and (recall=1, precision=base)
    recalls = np.array([0.0] + recalls[::-1] + [1.0])
    precisions = np.array([1.0] + precisions[::-1] + [float(np.mean(y_true))])
    return recalls, precisions


def measurement(y_true, y_pred, eval_metrics, num_tabs=1):
    """
    Calculate evaluation metrics for the given true and predicted labels.
    Supports Accuracy, Precision, Recall, F1-score, AUC, P@k, AP@k, MAP@k.
    """
    results = {}
    for eval_metric in eval_metrics:
        if eval_metric == "Accuracy":
            results[eval_metric] = round(float(accuracy_score(y_true, y_pred)), 4)
        elif eval_metric == "Precision":
            y_bin = (y_pred >= 0.5).astype(int) if y_true.ndim == 2 else y_pred
            results[eval_metric] = round(float(precision_score(y_true, y_bin, average="macro", zero_division=0)), 4)
        elif eval_metric == "Recall":
            y_bin = (y_pred >= 0.5).astype(int) if y_true.ndim == 2 else y_pred
            results[eval_metric] = round(float(recall_score(y_true, y_bin, average="macro", zero_division=0)), 4)
        elif eval_metric == "F1-score":
            y_bin = (y_pred >= 0.5).astype(int) if y_true.ndim == 2 else y_pred
            results[eval_metric] = round(float(f1_score(y_true, y_bin, average="macro", zero_division=0)), 4)
        elif eval_metric == "P@min":
            results[eval_metric] = round(float(np.min(precision_score(y_true, y_pred, average=None, zero_division=0))), 4)
        elif eval_metric == "r-Precision":
            results[eval_metric] = round(float(cal_r_precision(y_true, y_pred)), 4)
        elif eval_metric == "AUC":
            results[eval_metric] = round(safe_macro_auc(y_true, y_pred), 4)
        elif eval_metric.startswith("P@"):
            k = int(eval_metric.split("@")[1])
            results[eval_metric] = round(precision_at_k(y_true, y_pred, k), 4)
        elif eval_metric.startswith("AP@") or eval_metric.startswith("MAP@"):
            k = int(eval_metric.split("@")[1])
            results[eval_metric] = round(average_precision_at_k(y_true, y_pred, k), 4)
        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")
    return results


def cal_r_precision(y_true, y_pred, base_r=20):
    open_class = int(y_true.max())
    web2tp = {web: 0 for web in range(open_class + 1)}
    web2fp = {web: 0 for web in range(open_class + 1)}
    web2wp = {web: 0 for web in range(open_class + 1)}

    for index in range(len(y_true)):
        cur_true = int(y_true[index])
        cur_pred = int(y_pred[index])
        if cur_true == cur_pred:
            web2tp[cur_pred] += 1
        else:
            if cur_true == open_class:
                web2fp[cur_pred] += 1
            else:
                web2wp[cur_pred] += 1

    res = 0.0
    for web in range(open_class):
        denominator = web2tp[web] + base_r * web2fp[web] + web2wp[web]
        if denominator == 0:
            continue
        res += web2tp[web] / denominator
    res /= max(1, open_class)
    return res