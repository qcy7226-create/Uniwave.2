#!/usr/bin/env python3
"""
UniWave-2 Preprocessing: FASTA -> Waveform HDF5
Usage:
    python preprocess.py --fasta_files class1.fasta class2.fasta ... --output data.h5
"""
import os
import argparse
import h5py
import numpy as np
import random
from sklearn.model_selection import train_test_split
from scipy.signal import lfilter
import pywt

# -------------------- Configuration Constants --------------------
# Ghose-Crippen logP hydrophobicity values (corrected per reviewer)
BASE_MAPPING = {
    'A': -1.07,
    'T': -0.36,
    'C': -0.76,
    'G': -1.36
}

WAVELET = 'sym6'
WAVELET_LEVEL = 3
THRESHOLD_PERCENTILE = 20
K_OVERSAMPLE = 6
DOWNSAMPLE_FACTOR = 3

# -------------------- Core Functions --------------------
def read_multiple_fasta(file_path, strict_length=1000):
    """Read FASTA and extract sequences of exact length (random crop if longer)."""
    sequences = []
    valid_bases = {'A', 'T', 'C', 'G'}
    with open(file_path, 'r', encoding='utf-8') as f:
        current_seq = []
        for line in f:
            line = line.strip().upper()
            if line.startswith('>'):
                if current_seq:
                    filtered = [c for c in current_seq if c in valid_bases]
                    full_seq = ''.join(filtered)
                    if len(full_seq) == strict_length:
                        sequences.append(full_seq)
                    elif len(full_seq) > strict_length:
                        start = random.randint(0, len(full_seq) - strict_length)
                        sequences.append(full_seq[start:start+strict_length])
                    current_seq = []
                continue
            current_seq.extend(line)
        if current_seq:
            filtered = [c for c in current_seq if c in valid_bases]
            full_seq = ''.join(filtered)
            if len(full_seq) == strict_length:
                sequences.append(full_seq)
            elif len(full_seq) > strict_length:
                start = (len(full_seq) - strict_length) // 2
                sequences.append(full_seq[start:start+strict_length])
    print(f"File {file_path}: {len(sequences)} sequences of length {strict_length}")
    assert all(len(s) == strict_length for s in sequences), "Length mismatch!"
    return sequences

def global_deduplicate(all_samples):
    """Remove identical sequences across all classes."""
    global_seen = set()
    deduped = []
    for class_samples in all_samples:
        unique = []
        for s in class_samples:
            if s not in global_seen:
                global_seen.add(s)
                unique.append(s)
        deduped.append(unique)
    return deduped

def encode_sequence(sequence, mapping):
    return np.array([mapping[base] for base in sequence], dtype=np.float32)

def interpolate_signal(signal, target_length=6000):
    """Frequency-domain zero-padding interpolation."""
    freq = np.fft.rfft(signal)
    new_freq = np.zeros(target_length // 2 + 1, dtype=np.complex64)
    new_freq[:len(freq)] = freq
    return np.fft.irfft(new_freq, n=target_length)

def adaptive_signal_enhance(signal):
    """Local statistics-based enhancement."""
    window_size = 60
    kernel = np.ones(window_size) / window_size
    means = np.convolve(signal, kernel, mode='same')
    squares = np.convolve(signal**2, kernel, mode='same')
    stds = np.sqrt(squares - means**2 + 1e-6)
    enhancement = np.where(
        stds > 0.1,
        (signal - means) / stds * 0.2 + signal,
        signal * 1.1
    )
    return enhancement

def wavelet_convolutional_compression(signal):
    """Wavelet decomposition, thresholding, reconstruction, and 3x downsampling."""
    assert len(signal) == 6000, "Signal must be 6000 points"
    coeffs = pywt.wavedec(signal, WAVELET, level=WAVELET_LEVEL, mode='periodization')
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        c = coeffs[i]
        energy_threshold = np.percentile(np.abs(c), THRESHOLD_PERCENTILE)
        mask = (np.abs(c) > energy_threshold).astype(float)
        kernel = np.exp(-np.linspace(-3, 3, 5)**2)
        kernel /= kernel.sum()
        filtered = np.convolve(c * mask, kernel, mode='same')
        new_coeffs.append(filtered)
    reconstructed = pywt.waverec(new_coeffs, WAVELET, mode='periodization')
    downsampled = reconstructed[::3]
    return downsampled[:2000].astype(np.float32)

def post_filter(signal_2k):
    """Bidirectional mean filtering."""
    kernel = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    filtered = lfilter(kernel, 1, signal_2k[::-1])[::-1]
    return 0.7 * filtered + 0.3 * signal_2k

def process_pipeline(seq_list, mapping):
    """Full waveform encoding pipeline for a list of sequences."""
    processed = []
    for seq in seq_list:
        encoded = encode_sequence(seq, mapping)
        interpolated = interpolate_signal(encoded)
        enhanced = adaptive_signal_enhance(interpolated)
        downsampled = wavelet_convolutional_compression(enhanced)
        final = post_filter(downsampled)
        processed.append(final)
    return np.array(processed, dtype=np.float32)

# -------------------- Main Preprocessing Function --------------------
def multi_dimension_wave(fasta_paths, mapping, output_file,
                         test_size=0.1, val_size=0.2, seed=726,
                         max_len=1000):
    """
    Main entry: read FASTA files, deduplicate, split, encode, and save HDF5.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Read all sequences
    all_samples = []
    for path in fasta_paths:
        seqs = read_multiple_fasta(path, strict_length=max_len)
        all_samples.append(seqs)

    print("\n=== Before deduplication ===")
    for i, s in enumerate(all_samples):
        print(f"Class {i}: {len(s)}")
    original_total = sum(len(c) for c in all_samples)
    print(f"Total: {original_total}")

    # Deduplicate
    all_samples = global_deduplicate(all_samples)
    deduped_total = sum(len(c) for c in all_samples)
    print("\n=== After deduplication ===")
    for i, s in enumerate(all_samples):
        print(f"Class {i}: {len(s)}")
    print(f"Removed: {original_total - deduped_total}")

    # Balance classes (downsample to min count)
    min_count = min(len(s) for s in all_samples)
    balanced = [random.sample(s, min_count) for s in all_samples]

    # Split into train/val/test (stratified)
    train_list, val_list, test_list = [], [], []
    for class_samples in balanced:
        # First split: 10% test
        train_val, test = train_test_split(class_samples, test_size=test_size, random_state=seed)
        # From remaining, split val (proportion adjusted to achieve final 20% of total)
        # Since test=0.1, remaining=0.9, we want val=0.2 of total -> val/(remaining)=0.2/0.9 ≈ 0.2222
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=seed)
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    # Encode each split
    train_encoded = [process_pipeline(s, mapping) for s in train_list]
    val_encoded   = [process_pipeline(s, mapping) for s in val_list]
    test_encoded  = [process_pipeline(s, mapping) for s in test_list]

    # Check data leakage (optional, but we include a simple hash check)
    def check_leak(source, target):
        source_hashes = set(arr.tobytes() for arr in source)
        return sum(1 for arr in target if arr.tobytes() in source_hashes)

    for i in range(len(train_encoded)):
        assert check_leak(train_encoded[i], test_encoded[i]) == 0, f"Leak in class {i} test"
        assert check_leak(train_encoded[i], val_encoded[i]) == 0, f"Leak in class {i} val"
        assert check_leak(val_encoded[i], test_encoded[i]) == 0, f"Leak in class {i} test-val"

    # Standardize using training mean/std
    all_train = np.concatenate(train_encoded)
    global_mean = np.mean(all_train)
    global_std = np.std(all_train)
    train_encoded = [(cls - global_mean) / global_std for cls in train_encoded]
    val_encoded   = [(cls - global_mean) / global_std for cls in val_encoded]
    test_encoded  = [(cls - global_mean) / global_std for cls in test_encoded]

    # Save statistics
    np.savez(os.path.splitext(output_file)[0] + "_stats.npz", mean=global_mean, std=global_std)

    # Determine final sequence length after downsampling
    final_len = max_len * K_OVERSAMPLE // DOWNSAMPLE_FACTOR

    # Write HDF5
    with h5py.File(output_file, 'w') as h5f:
        for group_name, data in zip(["train", "val", "test"],
                                    [train_encoded, val_encoded, test_encoded]):
            group = h5f.create_group(group_name)
            for class_idx in range(len(data)):
                cls_group = group.create_group(f"class_{class_idx}")
                cls_group.create_dataset(
                    "data",
                    data=data[class_idx],
                    compression="gzip",
                    chunks=(10, final_len),
                    dtype=np.float32
                )
        # Metadata
        h5f.attrs.update({
            "creation_date": np.datetime64('now').astype('S'),
            "split_ratio": f"train:{1-test_size-val_size:.2f}, val:{val_size:.2f}, test:{test_size:.2f}",
            "class_labels": [os.path.basename(p) for p in fasta_paths],
            "wavelet_params": f"{WAVELET}/{WAVELET_LEVEL}levels/{THRESHOLD_PERCENTILE}%",
            "hydrophobicity": str(mapping)
        })

    # Statistics
    print("\n=== Final data statistics ===")
    print(f"Training set: {sum(len(c) for c in train_encoded)}")
    print(f"Validation set: {sum(len(c) for c in val_encoded)}")
    print(f"Test set: {sum(len(c) for c in test_encoded)}")
    for i in range(len(train_encoded)):
        print(f"Class {i}: train {len(train_encoded[i])}, val {len(val_encoded[i])}, test {len(test_encoded[i])}")

    print(f"\nHDF5 saved to {output_file}")

# -------------------- Command-line Interface --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniWave-2 Preprocessing")
    parser.add_argument("--fasta_files", nargs="+", required=True,
                        help="List of FASTA files (one per class)")
    parser.add_argument("--output", required=True,
                        help="Output HDF5 file path")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Proportion for test set (default 0.1)")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Proportion for validation set (default 0.2)")
    parser.add_argument("--seed", type=int, default=726,
                        help="Random seed")
    parser.add_argument("--max_len", type=int, default=1000,
                        help="Original sequence length (default 1000)")
    args = parser.parse_args()

    multi_dimension_wave(
        fasta_paths=args.fasta_files,
        mapping=BASE_MAPPING,
        output_file=args.output,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        max_len=args.max_len
    )