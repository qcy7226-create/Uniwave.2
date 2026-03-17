# Encoding after downsampling

import h5py
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from scipy.signal import lfilter
import pywt


def read_multiple_fasta(file_path, strict_length=1000):
    """Strictly filter bp sequences"""
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
                        sequences.append(full_seq[start:start + strict_length])
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
                sequences.append(full_seq[start:start + strict_length])

    print(f"File {file_path} effective 1000bp sequences: {len(sequences)}")
    assert all(len(seq) == strict_length for seq in sequences), "There are sequences with abnormal length"
    return sequences


def global_deduplicate(all_samples):
    """Global deduplication across categories"""
    global_seen = set()
    deduped_samples = []

    for class_samples in all_samples:
        unique = []
        for s in class_samples:
            if s not in global_seen:
                global_seen.add(s)
                unique.append(s)
        deduped_samples.append(unique)
    return deduped_samples


def encode_sequence(sequence, mapping):
    """Enhanced encoding function"""
    return np.array([mapping[base] for base in sequence], dtype=np.float32)


def interpolate_signal(signal, target_length=6000):
    """
    Frequency domain interpolation: increase signal density by zero padding in FFT
    Parameters:
        signal: original time-domain signal (length N)
        target_length: target length after interpolation (M > N)
    Returns:
        Interpolated time-domain signal (length M)
    """
    # 1. Compute FFT to obtain frequency domain representation
    freq_domain = np.fft.rfft(signal)  # Use real FFT for efficiency

    # 2. Zero pad in the high-frequency region of the frequency domain
    current_length = len(freq_domain)
    new_freq = np.zeros(target_length // 2 + 1, dtype=np.complex64)
    new_freq[:current_length] = freq_domain

    # 3. Inverse FFT back to time domain
    interpolated = np.fft.irfft(new_freq, n=target_length)

    return interpolated


# New function 1: Adaptive signal enhancement (after interpolation, before downsampling)
def adaptive_signal_enhance(signal):
    """Signal enhancement based on local statistics (keeping 6000-point dimension)"""
    # Optimized sliding window calculation (vectorized implementation)
    window_size = 60
    kernel = np.ones(window_size) / window_size

    # Compute mean and standard deviation
    means = np.convolve(signal, kernel, mode='same')
    squares = np.convolve(signal ** 2, kernel, mode='same')
    stds = np.sqrt(squares - means ** 2 + 1e-6)

    # Dynamic gain control (avoid loops)
    enhancement = np.where(
        stds > 0.1,
        (signal - means) / stds * 0.2 + signal,  # High fluctuation region
        signal * 1.1  # Flat region
    )
    return enhancement


# New wavelet compression parameter configuration
WAVELET = 'sym6'              # Wavelet basis type
WAVELET_LEVEL = 3            # Decomposition level
THRESHOLD_PERCENTILE = 20    # High-frequency coefficient retention threshold

def wavelet_convolutional_compression(signal):
    """Core function for wavelet-convolution joint compression downsampling"""
    # Ensure input is the interpolated signal
    assert len(signal) == 6000, "Input signal length must be 6000 points"
    # Wavelet decomposition
    coeffs = pywt.wavedec(signal, WAVELET, level=WAVELET_LEVEL, mode='periodization')
    # Initialize new coefficient list (keep approximation coefficients)
    new_coeffs = [coeffs[0]]
    # Process high-frequency detail coefficients
    for i in range(1, len(coeffs)):
        c = coeffs[i]
        # Dynamically compute energy threshold
        energy_threshold = np.percentile(np.abs(c), THRESHOLD_PERCENTILE)
        # Generate energy mask
        energy_mask = (np.abs(c) > energy_threshold).astype(float)
        # Apply convolutional filtering (non-learning version, using predefined Gaussian kernel)
        conv_kernel = np.exp(-np.linspace(-3, 3, 5) ** 2)  # Simple Gaussian kernel
        conv_kernel /= conv_kernel.sum()
        filtered = np.convolve(c * energy_mask, conv_kernel, mode='same')
        new_coeffs.append(filtered)
    # Wavelet reconstruction
    reconstructed = pywt.waverec(new_coeffs, WAVELET, mode='periodization')
    # Exact downsampling to 2000 points
    downsampled = reconstructed[::3]
    return downsampled[:2000].astype(np.float32)  # Ensure data precision

# New function 2: Post-processing filtering (after downsampling)
def post_filter(signal_2k):
    """Lightweight post-filtering (keeping 2000-point dimension)"""
    # Optimized filter implementation (zero-phase filtering)
    kernel = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    filtered = lfilter(kernel, 1, signal_2k[::-1])[::-1]  # Bidirectional filtering
    return 0.7*filtered + 0.3*signal_2k

def multi_dimension_wave(fasta_paths, mapping, output_file="ecoli_hqd_1000(9-).h5", test_size=0.1, val_size=0.2, seed=726):
    """Improved preprocessing pipeline"""
    # New interpolation parameters
    K_OVERSAMPLE = 6
    DOWNSAMPLE_FACTOR = 3  # Downsampling factor
    MAX_LENGTH = 1000 * K_OVERSAMPLE // DOWNSAMPLE_FACTOR

    # Data collection and deduplication
    all_samples = []
    for path in fasta_paths:
        seqs = read_multiple_fasta(path)
        all_samples.append(seqs)

    print("\n=== Statistics before deduplication ===")
    for i, samples in enumerate(all_samples):
        print(f"Class {i} original sample count: {len(samples)}")
    original_total = sum(len(c) for c in all_samples)
    print(f"Total sample count: {original_total}")

    # Global deduplication
    all_samples = global_deduplicate(all_samples)

    print("\n=== Statistics after deduplication ===")
    for i, samples in enumerate(all_samples):
        print(f"Class {i} deduplicated sample count: {len(samples)}")
    deduped_total = sum(len(c) for c in all_samples)
    print(f"Removed duplicate samples: {original_total - deduped_total}")
    print(f"Effective total samples: {deduped_total}")

    # Balanced sampling
    min_count = min(len(s) for s in all_samples)
    balanced = [random.sample(s, min_count) for s in all_samples]

    # New test set splitting
    processed_train, processed_val, processed_test = [], [], []

    for class_idx, samples in enumerate(balanced):
        # First split test set (10%)
        remain, test = train_test_split(samples, test_size=test_size, random_state=seed)

        # Split remaining data into training and validation sets (adjust val_size relative to remaining)
        train_val_test_size = val_size / (1 - test_size)  # Convert to relative proportion
        train, val = train_test_split(remain, test_size=train_val_test_size, random_state=seed)

        # ================= Encoding + interpolation processing =================
        def process_pipeline(seq_list):
            """Encapsulated processing pipeline (with WCJC downsampling)"""
            processed = []
            for seq in seq_list:
                # Original encoding and interpolation remain unchanged
                encoded = encode_sequence(seq, mapping)
                interpolated = interpolate_signal(encoded)

                # New step 1: Signal enhancement after interpolation
                enhanced = adaptive_signal_enhance(interpolated)

                # New WCJC downsampling to 2000 points
                downsampled = wavelet_convolutional_compression(enhanced)

                # New step 2: Filtering after downsampling
                final_signal = post_filter(downsampled)
                processed.append(final_signal)
            return np.array(processed)

        train_encoded = process_pipeline(train)
        val_encoded = process_pipeline(val)
        test_encoded = process_pipeline(test)

        processed_train.append(train_encoded)
        processed_val.append(val_encoded)
        processed_test.append(test_encoded)

    # Cross-dataset deduplication check
    def check_leak(source, target):
        source_hashes = set(arr.tobytes() for arr in source)
        return sum(1 for arr in target if arr.tobytes() in source_hashes)

    # # Check test set and validation set leakage
    # for cls_idx in range(len(processed_train)):
    #     # 1. Check if test set contains training/validation data
    #     train_test_leak = check_leak(processed_train[cls_idx], processed_test[cls_idx])
    #     val_test_leak = check_leak(processed_val[cls_idx], processed_test[cls_idx])
    #     assert train_test_leak == 0 and val_test_leak == 0, f"Class {cls_idx} test set data leakage!"
    #
    #     # 2. Check if validation set contains training data (newly added)
    #     train_val_leak = check_leak(processed_train[cls_idx], processed_val[cls_idx])
    #     assert train_val_leak == 0, f"Class {cls_idx} validation set and training set have duplicate samples!"

    # === New standardization code ===
    # Merge training data to compute statistics
    all_train = np.concatenate(processed_train)
    global_mean = np.mean(all_train)
    global_std = np.std(all_train)

    # Apply standardization to all data
    processed_train = [(cls - global_mean) / global_std for cls in processed_train]
    processed_val = [(cls - global_mean) / global_std for cls in processed_val]
    processed_test = [(cls - global_mean) / global_std for cls in processed_test]

    # Save statistics
    np.savez("dataset_stats.npz", mean=global_mean, std=global_std)

    # Store in HDF5 (newly added test set)
    with h5py.File(output_file, "w") as h5f:
        # Create dataset groups
        for group_name, data in zip(["train", "val", "test"],
                                    [processed_train, processed_val, processed_test]):
            group = h5f.create_group(group_name)
            for class_idx in range(len(data)):
                cls_group = group.create_group(f"class_{class_idx}")
                cls_group.create_dataset(
                    "data",
                    data=data[class_idx],
                    compression="gzip",
                    chunks=(10, MAX_LENGTH),
                    dtype=np.float32
                )

        # Metadata
        h5f.attrs.update({
            "creation_date": np.datetime64('now').astype('S'),
            "author": "Your Name",
            "description": "Three-part data with independent test set",
            "dimensions": str(MAX_LENGTH),
            "split_ratio": f"train:{(1 - test_size - val_size):.1f}, val:{val_size:.1f}, test:{test_size:.1f}",
            "class_labels": [os.path.basename(p) for p in fasta_paths],
            "wavelet_params": f"{WAVELET}/{WAVELET_LEVEL} levels / {THRESHOLD_PERCENTILE}% energy threshold"
        })

    # Data statistics output
    print("\n=== Final data statistics ===")
    total_train = sum(len(c) for c in processed_train)
    total_val = sum(len(c) for c in processed_val)
    total_test = sum(len(c) for c in processed_test)
    print(f"Training set: {total_train} | Validation set: {total_val} | Test set: {total_test}")

    for class_idx in range(len(processed_train)):
        print(f"\nClass {class_idx}:")
        print(f"  Training samples: {len(processed_train[class_idx])}")
        print(f"  Validation samples: {len(processed_val[class_idx])}")
        print(f"  Test samples: {len(processed_test[class_idx])}")


if __name__ == '__main__':
    # Configuration parameters (modify to actual paths)
    fasta_paths = [
        r"C:\Users\qyj\Desktop\文件\数据\CISAID数据\Alpha.fasta",
        r"C:\Users\qyj\Desktop\文件\数据\CISAID数据\Beta.fasta",
        r"C:\Users\qyj\Desktop\文件\数据\CISAID数据\Gamma.fasta",
        r"C:\Users\qyj\Desktop\文件\数据\CISAID数据\Delta.fasta",
        r"C:\Users\qyj\Desktop\文件\数据\CISAID数据\BA.1.fasta",
        r"C:\Users\qyj\Desktop\文件\数据\CISAID数据\BA.2.fasta",
        r"C:\Users\qyj\Desktop\文件\数据\CISAID数据\BA.4.fasta",
        r"C:\Users\qyj\Desktop\文件\数据\CISAID数据\BA.5.fasta",
        r"C:\Users\qyj\Desktop\文件\数据\CISAID数据\XBB.1.5.fasta"
    ]
    base_mapping = {'A': -0.12, 'T': -0.31, 'C': -0.37, 'G': 0.13}
    # Execute processing pipeline
    multi_dimension_wave(fasta_paths, base_mapping)
    # Final leakage check
    print("\n=== Final leakage check ===")
    with h5py.File("ecoli_hqd_1000(9-).h5", 'r') as h5:
        # Collect all training data hashes
        train_hashes = set()
        for cls in h5['train']:
            data = h5[f'train/{cls}/data'][:]
            for arr in data:
                train_hashes.add(arr.tobytes())

        # Check test set
        test_leak_count = 0
        for cls in h5['test']:
            data = h5[f'test/{cls}/data'][:]
            for arr in data:
                if arr.tobytes() in train_hashes:
                    test_leak_count += 1
        print(f"Number of duplicate samples between test set and training set: {test_leak_count} (should be 0)")