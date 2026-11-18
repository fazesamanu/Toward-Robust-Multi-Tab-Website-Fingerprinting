import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def length_align(X, seq_len):
    """
    Align the length of the sequences to the specified sequence length.
    """
    if seq_len < X.shape[-1]:
        X = X[..., :seq_len]
    if seq_len > X.shape[-1]:
        padding_num = seq_len - X.shape[-1]
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, padding_num)]
        X = np.pad(X, pad_width=pad_width, mode="constant", constant_values=0)
    return X


def load_data(data_path, feature_type, seq_len, num_tab=1):
    """
    Load and process data from a specified path.
    """
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]
    return transform_features(X, y, feature_type, seq_len, num_tab)


def transform_features(X, y, feature_type, seq_len, num_tab=1):
    """
    Transform raw trace array X and labels y into model-ready tensors.
    """
    if feature_type == "DIR":
        X = np.sign(X)
        X = length_align(X, seq_len)
        X = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
    elif feature_type == "DT":
        X = length_align(X, seq_len)
        X = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
    elif feature_type == "DT2":
        X_dir = np.sign(X)
        X_time = np.abs(X)
        X_time = np.diff(X_time, prepend=0)
        X_time[X_time < 0] = 0
        X_dir = length_align(X_dir, seq_len)[:, np.newaxis]
        X_time = length_align(X_time, seq_len)[:, np.newaxis]
        X = np.concatenate([X_dir, X_time], axis=1)
        X = torch.tensor(X, dtype=torch.float32)
    elif feature_type == "TAM":
        X = length_align(X, seq_len)
        if X.ndim == 3:
            X = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
        else:
            X = torch.tensor(X, dtype=torch.float32)
    elif feature_type in ["TAF", "MTAF"]:
        X = length_align(X, seq_len)
        X = torch.tensor(X, dtype=torch.float32)
    elif feature_type == "Origin":
        X = length_align(X, seq_len)
        return X, y
    else:
        raise ValueError(f"Feature type {feature_type} is not matched.")

    if num_tab == 1 and y.ndim == 1:
        y = torch.tensor(y, dtype=torch.int64)
    else:
        y = torch.tensor(y, dtype=torch.float32)

    return X, y


def load_iter(X, y, batch_size, is_train=True, num_workers=0, weight_sample=False):
    """
    Load data into a DataLoader for batch processing.
    """
    if weight_sample:
        class_sample_count = np.unique(y.numpy(), return_counts=True)[1]
        weight = 1.0 / class_sample_count
        samples_weight = weight[y.numpy()]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
        )
    dataset = torch.utils.data.TensorDataset(X, y)
    drop_last = is_train and (len(dataset) > batch_size)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, drop_last=drop_last, num_workers=num_workers
    )


def fast_count_burst(arr):
    if len(arr) == 0:
        return np.array([], dtype=np.float32)
    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0]
    segment_starts = np.insert(change_indices + 1, 0, 0)
    segment_ends = np.append(change_indices, len(arr) - 1)
    segment_lengths = segment_ends - segment_starts + 1
    segment_signs = np.sign(arr[segment_starts])
    adjusted_lengths = segment_lengths * segment_signs
    return adjusted_lengths


def agg_interval(packets):
    features = []
    features.append([np.sum(packets > 0), np.sum(packets < 0)])

    dirs = np.sign(packets)
    bursts = fast_count_burst(dirs)
    features.append([np.sum(bursts > 0), np.sum(bursts < 0)])

    pos_bursts = bursts[bursts > 0]
    neg_bursts = np.abs(bursts[bursts < 0])
    vals = [
        np.mean(pos_bursts) if len(pos_bursts) > 0 else 0.0,
        np.mean(neg_bursts) if len(neg_bursts) > 0 else 0.0,
    ]
    features.append(vals)
    return np.array(features, dtype=np.float32)


def agg_interval2(packets):
    """
    Extract 8 multi-level traffic aggregation features for a time window (Section IV-B, Fig. 4):
      0, 1: Outgoing & Incoming packet count
      2, 3: Outgoing & Incoming inter-packet interval statistic
      4, 5: Outgoing & Incoming burst count
      6, 7: Outgoing & Incoming average burst size
    """
    pos_mask = packets > 0
    neg_mask = packets < 0
    pos_packets = packets[pos_mask]
    neg_packets = np.abs(packets[neg_mask])

    pos_cnt = float(pos_packets.size)
    neg_cnt = float(neg_packets.size)

    pos_int = float(np.sum(np.diff(pos_packets))) if pos_packets.size > 1 else 0.0
    neg_int = float(np.sum(np.diff(neg_packets))) if neg_packets.size > 1 else 0.0

    dirs = np.sign(packets)
    bursts = fast_count_burst(dirs)
    pos_bursts = bursts[bursts > 0]
    neg_bursts = np.abs(bursts[bursts < 0])

    pos_b_cnt = float(pos_bursts.size)
    neg_b_cnt = float(neg_bursts.size)
    pos_b_size = float(np.mean(pos_bursts)) if pos_bursts.size > 0 else 0.0
    neg_b_size = float(np.mean(neg_bursts)) if neg_bursts.size > 0 else 0.0

    return np.array(
        [pos_cnt, neg_cnt, pos_int, neg_int, pos_b_cnt, neg_b_cnt, pos_b_size, neg_b_size],
        dtype=np.float32,
    )


def process_MTAF(index, sequence, interval, max_len):
    packets = np.trim_zeros(sequence, "fb")
    TAF = np.zeros((8, max_len), dtype=np.float32)
    if packets.size == 0:
        return index, TAF

    abs_packets = np.abs(packets)
    # Ensure monotonically non-decreasing timestamps for searchsorted
    if np.any(np.diff(abs_packets) < 0):
        order = np.argsort(abs_packets, kind="mergesort")
        packets = packets[order]
        abs_packets = abs_packets[order]

    st_time = abs_packets[0]
    edges = np.empty(max_len + 1, dtype=np.int64)
    edges[0] = 0
    if max_len > 1:
        boundaries = st_time + np.arange(1, max_len, dtype=np.float64) * interval
        edges[1:-1] = np.searchsorted(abs_packets, boundaries)
    edges[-1] = abs_packets.shape[0]

    non_empty = np.nonzero(edges[1:] > edges[:-1])[0]
    for interval_idx in non_empty:
        cur_packets = packets[edges[interval_idx] : edges[interval_idx + 1]]
        TAF[:, interval_idx] = agg_interval2(cur_packets)

    return index, TAF


def extract_MTAF(sequences, interval=20, max_len=8000, num_workers=0, feature_mask=None, show_progress=False):
    """
    Extract Multi-Level Traffic Aggregation Features (MTAF) (Section IV-B, Table I).
    Parameters:
      sequences (ndarray): Input sequences of signed timestamps (in seconds).
      interval (int): Aggregation window interval t in ms (default 20 ms).
      max_len (int): Number of time windows / input dimension d (default 8000).
      num_workers (int): Process pool workers (0 = fast in-process loop).
      feature_mask (list/ndarray): Optional mask over the 8 channels for Table IX ablations.
    """
    seqs_ms = np.array(sequences, dtype=np.float64, copy=True) * 1000.0
    num_sequences = seqs_ms.shape[0]
    TAF = np.zeros((num_sequences, 8, max_len), dtype=np.float32)

    if num_workers > 1 and num_sequences > 32:
        with ProcessPoolExecutor(max_workers=min(num_workers, num_sequences)) as executor:
            futures = [
                executor.submit(process_MTAF, idx, seqs_ms[idx], interval, max_len)
                for idx in range(num_sequences)
            ]
            iterator = tqdm(as_completed(futures), total=num_sequences) if show_progress else as_completed(futures)
            for future in iterator:
                idx, result = future.result()
                TAF[idx] = result
    else:
        iterator = tqdm(range(num_sequences)) if show_progress else range(num_sequences)
        for idx in iterator:
            _, result = process_MTAF(idx, seqs_ms[idx], interval, max_len)
            TAF[idx] = result

    if feature_mask is not None:
        mask = np.asarray(feature_mask, dtype=np.float32).reshape(1, 8, 1)
        TAF = TAF * mask

    return TAF


def process_TAF(index, sequence, interval, max_len):
    packets = np.trim_zeros(sequence, "fb")
    TAF = np.zeros((3, 2, max_len), dtype=np.float32)
    if packets.size == 0:
        return index, TAF

    abs_packets = np.abs(packets)
    if np.any(np.diff(abs_packets) < 0):
        order = np.argsort(abs_packets, kind="mergesort")
        packets = packets[order]
        abs_packets = abs_packets[order]

    st_time = abs_packets[0]
    edges = np.empty(max_len + 1, dtype=np.int64)
    edges[0] = 0
    if max_len > 1:
        boundaries = st_time + np.arange(1, max_len, dtype=np.float64) * interval
        edges[1:-1] = np.searchsorted(abs_packets, boundaries)
    edges[-1] = abs_packets.shape[0]

    non_empty = np.nonzero(edges[1:] > edges[:-1])[0]
    for interval_idx in non_empty:
        cur_packets = packets[edges[interval_idx] : edges[interval_idx + 1]]
        TAF[:, :, interval_idx] = agg_interval(cur_packets)

    return index, TAF


def extract_TAF(sequences, interval=40, max_len=2000, num_workers=0):
    seqs_ms = np.array(sequences, dtype=np.float64, copy=True) * 1000.0
    num_sequences = seqs_ms.shape[0]
    TAF = np.zeros((num_sequences, 3, 2, max_len), dtype=np.float32)
    for idx in range(num_sequences):
        _, result = process_TAF(idx, seqs_ms[idx], interval, max_len)
        TAF[idx] = result
    return TAF


def process_TAM(index, sequence, maximum_load_time, max_matrix_len):
    feature = np.zeros((2, max_matrix_len), dtype=np.float32)
    packets = np.trim_zeros(sequence, "fb")
    if packets.size == 0:
        return index, feature

    pos_packs = packets[packets > 0]
    neg_packs = np.abs(packets[packets < 0])

    if pos_packs.size > 0:
        idx_pos = np.where(
            pos_packs >= maximum_load_time,
            max_matrix_len - 1,
            (pos_packs * (max_matrix_len - 1) / maximum_load_time).astype(np.int64),
        )
        idx_pos = np.clip(idx_pos, 0, max_matrix_len - 1)
        np.add.at(feature[0], idx_pos, 1.0)

    if neg_packs.size > 0:
        idx_neg = np.where(
            neg_packs >= maximum_load_time,
            max_matrix_len - 1,
            (neg_packs * (max_matrix_len - 1) / maximum_load_time).astype(np.int64),
        )
        idx_neg = np.clip(idx_neg, 0, max_matrix_len - 1)
        np.add.at(feature[1], idx_neg, 1.0)

    return index, feature


def extract_TAM(sequences, maximum_load_time=80, max_matrix_len=1800, num_workers=0):
    """
    Extract the Traffic Aggregation Matrix (TAM) used by RF / MultiTabRF.
    """
    num_sequences = sequences.shape[0]
    TAM = np.zeros((num_sequences, 2, max_matrix_len), dtype=np.float32)
    for idx in range(num_sequences):
        _, result = process_TAM(idx, sequences[idx], maximum_load_time, max_matrix_len)
        TAM[idx] = result
    return TAM
