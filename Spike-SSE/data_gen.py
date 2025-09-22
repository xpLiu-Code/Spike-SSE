import numpy as np
import itertools
import h5py
from tqdm import tqdm
import os


def generate_doa_dataset():
    num_antennas = 10
    num_snapshots = 10
    wavelength = 1.0
    d = wavelength / 2
    angles = np.linspace(-60, 60, 121)
    snrs = np.linspace(-10, 20, 31)

    num_classes_single = len(angles)
    num_classes_double = len(list(itertools.combinations(range(len(angles)), 2)))
    num_classes_triple = len(list(itertools.combinations(range(len(angles)), 3)))

    total_samples = (num_classes_single + num_classes_double + num_classes_triple) * len(snrs)

    print(f"Number of antennas: {num_antennas}, Snapshots: {num_snapshots}")
    print(f"Angle range: -60° to 60°, Points: {len(angles)}")
    print(f"SNR range: -10dB to 20dB, Points: {len(snrs)}")
    print(f"Single target classes: {num_classes_single}")
    print(f"Double target classes: {num_classes_double}")
    print(f"Triple target classes: {num_classes_triple}")
    print(f"Total samples: {total_samples}")

    if not os.path.exists('Data'):
        os.makedirs('Data')

    data = np.zeros((total_samples, num_antennas, num_snapshots), dtype=np.complex64)
    labels = np.zeros((total_samples, 3), dtype=np.float32)
    snr_values = np.zeros(total_samples, dtype=np.float32)
    num_targets = np.zeros(total_samples, dtype=np.int8)

    def steering_vector(theta):
        theta_rad = np.deg2rad(theta)
        return np.exp(-1j * 2 * np.pi * d * np.arange(num_antennas) * np.sin(theta_rad))

    def generate_signal(thetas, snr_db):
        num_sources = len(thetas)
        A = np.array([steering_vector(theta) for theta in thetas]).T

        source_signal = (np.sign(np.random.randn(num_sources, num_snapshots)) +
                         1j * np.sign(np.random.randn(num_sources, num_snapshots))) / np.sqrt(2)

        X = A @ source_signal

        signal_power = np.mean(np.abs(X) ** 2)

        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power / 2) * (np.random.randn(num_antennas, num_snapshots) +
                                            1j * np.random.randn(num_antennas, num_snapshots))
        return X + noise

    sample_idx = 0

    print("Generating single target data...")
    for angle_idx, angle in enumerate(tqdm(angles)):
        for snr in snrs:
            signal = generate_signal([angle], snr)
            data[sample_idx] = signal
            labels[sample_idx] = [angle, 0, 0]
            snr_values[sample_idx] = snr
            num_targets[sample_idx] = 1
            sample_idx += 1

    print("Generating double target data...")
    for angle_idx1, angle1 in enumerate(tqdm(angles)):
        for angle_idx2, angle2 in enumerate(angles[angle_idx1 + 1:], start=angle_idx1 + 1):
            for snr in snrs:
                signal = generate_signal([angle1, angle2], snr)
                data[sample_idx] = signal
                labels[sample_idx] = [angle1, angle2, 0]
                snr_values[sample_idx] = snr
                num_targets[sample_idx] = 2
                sample_idx += 1

    print("Generating triple target data...")
    for angle_idx1, angle1 in enumerate(tqdm(angles)):
        for angle_idx2, angle2 in enumerate(angles[angle_idx1 + 1:], start=angle_idx1 + 1):
            for angle_idx3, angle3 in enumerate(angles[angle_idx2 + 1:], start=angle_idx2 + 1):
                for snr in snrs:
                    signal = generate_signal([angle1, angle2, angle3], snr)
                    data[sample_idx] = signal
                    labels[sample_idx] = [angle1, angle2, angle3]
                    snr_values[sample_idx] = snr
                    num_targets[sample_idx] = 3
                    sample_idx += 1

    print("Saving dataset...")
    with h5py.File('Data/doa_dataset.h5', 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('labels', data=labels)
        f.create_dataset('snr', data=snr_values)
        f.create_dataset('num_targets', data=num_targets)
        f.create_dataset('angles', data=angles)
        f.create_dataset('snrs', data=snrs)

    print("Dataset generation completed! Saved to Data/doa_dataset.h5")


if __name__ == "__main__":
    generate_doa_dataset()