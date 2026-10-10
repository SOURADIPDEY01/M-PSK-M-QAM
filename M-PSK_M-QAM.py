import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# NORMALIZED M-PSK AND SQUARE M-QAM WITH GRAY MAPPING
# Minimum-Euclidean-distance detection
# ============================================================

RNG = np.random.default_rng(7)
SNR_DB_LIST = [0, 5, 10, 20]
NUM_SYMBOLS = 100_000


def int_to_bits(values, width):
    """Convert non-negative integers to MSB-first bit rows."""
    values = np.asarray(values, dtype=np.int64)
    shifts = np.arange(width - 1, -1, -1)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8)


def bits_to_int(bits):
    """Convert MSB-first bit rows to integers."""
    bits = np.asarray(bits, dtype=np.uint8)
    weights = 1 << np.arange(bits.shape[1] - 1, -1, -1)
    return bits @ weights


def gray_to_binary(gray):
    """Convert Gray-coded integer values to binary integers."""
    gray = np.asarray(gray, dtype=np.int64)
    binary = gray.copy()
    shift = 1
    while shift < 64:
        binary ^= binary >> shift
        shift <<= 1
    return binary


def binary_to_gray(binary):
    binary = np.asarray(binary, dtype=np.int64)
    return binary ^ (binary >> 1)


# --------------------------- M-PSK ----------------------------

def psk_mapper(bit_rows, M):
    """
    Gray-coded M-PSK mapper.
    Consecutive constellation points around the circle have Gray labels.
    The constellation is normalized to average symbol energy Es = 1.
    """
    k = int(np.log2(M))
    bit_rows = np.asarray(bit_rows, dtype=np.uint8)
    gray_labels = bits_to_int(bit_rows)
    binary_indices = gray_to_binary(gray_labels)
    phase = 2 * np.pi * binary_indices / M
    # Unit-circle PSK already has average energy 1.
    return np.exp(1j * phase)


def psk_constellation(M):
    k = int(np.log2(M))
    labels = np.arange(M, dtype=np.int64)
    gray_labels = binary_to_gray(labels)
    bits = int_to_bits(gray_labels, k)
    return psk_mapper(bits, M), bits


# -------------------------- Square QAM ------------------------

def qam_mapper(bit_rows, M):
    """
    Gray-coded square M-QAM mapper.
    The first half of the bits select the I axis and the second half
    select the Q axis. Gray labels are mapped to equally spaced levels.
    Average symbol energy is normalized to Es = 1.
    """
    side = int(np.sqrt(M))
    if side * side != M or (side & (side - 1)) != 0:
        raise ValueError("Square QAM order M must be a power-of-two square, e.g. 16 or 64.")

    k = int(np.log2(M))
    if k % 2:
        raise ValueError("Square QAM requires an even number of bits per symbol.")

    axis_bits = k // 2
    bit_rows = np.asarray(bit_rows, dtype=np.uint8)
    gray_i = bits_to_int(bit_rows[:, :axis_bits])
    gray_q = bits_to_int(bit_rows[:, axis_bits:])

    # Gray-coded axis label -> binary level index.
    idx_i = gray_to_binary(gray_i)
    idx_q = gray_to_binary(gray_q)

    # Equally spaced odd integer levels: -(side-1), ..., +(side-1)
    levels_i = 2 * idx_i - (side - 1)
    levels_q = 2 * idx_q - (side - 1)

    # E[|s|^2] for unnormalized square QAM is 2/3*(M-1).
    scale = np.sqrt(2 * (M - 1) / 3)
    return (levels_i + 1j * levels_q) / scale


def qam_constellation(M):
    k = int(np.log2(M))
    side = int(np.sqrt(M))
    if side * side != M or k % 2:
        raise ValueError("Square QAM requires M to be a power-of-two square.")
    axis_bits = k // 2
    # Build a separable Gray label: Gray-coded I-axis bits followed by
    # Gray-coded Q-axis bits. Adjacent horizontal/vertical points differ
    # in exactly one bit.
    points = []
    label_bits = []
    for idx_i in range(side):
        for idx_q in range(side):
            gray_i = int(binary_to_gray(np.array([idx_i]))[0])
            gray_q = int(binary_to_gray(np.array([idx_q]))[0])
            bits_i = int_to_bits(np.array([gray_i]), axis_bits)[0]
            bits_q = int_to_bits(np.array([gray_q]), axis_bits)[0]
            bits = np.concatenate((bits_i, bits_q))
            point = qam_mapper(bits.reshape(1, -1), M)[0]
            points.append(point)
            label_bits.append(bits)
    return np.asarray(points), np.asarray(label_bits, dtype=np.uint8)


# -------------------------- Demappers -------------------------

def minimum_distance_detect(received, constellation):
    """Choose the constellation point with minimum Euclidean distance."""
    # Chunk to keep memory use reasonable for large sample counts.
    received = np.asarray(received)
    detected = np.empty(received.size, dtype=np.int64)
    chunk_size = 5000
    for start in range(0, received.size, chunk_size):
        r = received[start:start + chunk_size]
        distances = np.abs(r[:, None] - constellation[None, :]) ** 2
        detected[start:start + chunk_size] = np.argmin(distances, axis=1)
    return detected


def psk_demapper(received, M):
    constellation, label_bits = psk_constellation(M)
    indices = minimum_distance_detect(received, constellation)
    return label_bits[indices], indices


def qam_demapper(received, M):
    constellation, label_bits = qam_constellation(M)
    indices = minimum_distance_detect(received, constellation)
    return label_bits[indices], indices


# ----------------------- AWGN channel -------------------------

def add_awgn(symbols, snr_db, bits_per_symbol):
    """
    SNR here means Eb/N0 in dB.
    With normalized Es=1, Eb=Es/log2(M). Complex AWGN has
    variance N0/2 in each real dimension.
    """
    eb_n0 = 10 ** (snr_db / 10)
    k = bits_per_symbol
    n0 = 1 / (k * eb_n0)
    noise = np.sqrt(n0 / 2) * (
        RNG.standard_normal(symbols.size) +
        1j * RNG.standard_normal(symbols.size)
    )
    return symbols + noise


def modulation_info(name, M):
    if name == "PSK":
        return psk_constellation(M)
    return qam_constellation(M)


def map_bits(bit_rows, name, M):
    if name == "PSK":
        return psk_mapper(bit_rows, M)
    return qam_mapper(bit_rows, M)


def demap_symbols(received, name, M):
    if name == "PSK":
        return psk_demapper(received, M)
    return qam_demapper(received, M)


# ----------------------- Visualization ------------------------

def plot_constellations(results, constellation_data):
    """Plot ideal and noisy received constellations at each SNR."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    for row, (name, M) in enumerate(results):
        ideal, _ = constellation_data[(name, M)]
        axes[row, 0].scatter(ideal.real, ideal.imag, s=22)
        axes[row, 0].set_title(f"Ideal {M}-{name} constellation")
        axes[row, 0].set_xlabel("In-phase")
        axes[row, 0].set_ylabel("Quadrature")
        axes[row, 0].grid(True)
        axes[row, 0].axis("equal")

        # Show the received cloud at 10 dB.
        k = int(np.log2(M))
        sample_bits = RNG.integers(0, 2, size=(3000, k), dtype=np.uint8)
        tx = map_bits(sample_bits, name, M)
        rx = add_awgn(tx, 10, k)
        axes[row, 1].scatter(rx.real, rx.imag, s=5, alpha=0.35, label="Received")
        axes[row, 1].scatter(ideal.real, ideal.imag, marker="x", s=35, label="Ideal points")
        axes[row, 1].set_title(f"{M}-{name}: received cloud at Eb/N0 = 10 dB")
        axes[row, 1].set_xlabel("In-phase")
        axes[row, 1].set_ylabel("Quadrature")
        axes[row, 1].grid(True)
        axes[row, 1].axis("equal")
        axes[row, 1].legend()
    plt.tight_layout()
    plt.show()


def plot_decision_regions(results, constellation_data):
    """Plot minimum-distance decision regions in the I-Q plane."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for ax, (name, M) in zip(axes, results):
        constellation, label_bits = constellation_data[(name, M)]
        lim = 1.65 if name == "PSK" else 1.65
        grid = np.linspace(-lim, lim, 350)
        xx, yy = np.meshgrid(grid, grid)
        points = (xx + 1j * yy).ravel()
        regions = minimum_distance_detect(points, constellation).reshape(xx.shape)

        ax.contourf(xx, yy, regions, levels=np.arange(M + 1) - 0.5, alpha=0.22)
        ax.scatter(constellation.real, constellation.imag, c=np.arange(M),
                   cmap="tab20", edgecolors="black", s=28)
        ax.set_title(f"Minimum-distance decision regions: {M}-{name}")
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.grid(True, alpha=0.35)
        ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


def plot_ber_ser(results, snr_values, ber_results, ser_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, M in results:
        axes[0].semilogy(snr_values, ber_results[(name, M)], marker="o",
                         label=f"{M}-{name}")
        axes[1].semilogy(snr_values, ser_results[(name, M)], marker="s",
                         label=f"{M}-{name}")
    axes[0].set_title("BER versus Eb/N0")
    axes[0].set_xlabel("Eb/N0 (dB)")
    axes[0].set_ylabel("Bit Error Rate (BER)")
    axes[1].set_title("SER versus Eb/N0")
    axes[1].set_xlabel("Eb/N0 (dB)")
    axes[1].set_ylabel("Symbol Error Rate (SER)")
    for ax in axes:
        ax.grid(True, which="both")
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ber_vs_efficiency(results, ber_results, snr_values):
    """Plot BER against spectral efficiency for each tested Eb/N0."""
    efficiencies = [int(np.log2(M)) for _, M in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, snr_db in enumerate(snr_values):
        bers = [max(ber_results[(name, M)][i], 1 / (NUM_SYMBOLS * int(np.log2(M))))
                for name, M in results]
        ax.semilogy(efficiencies, bers, marker="o", label=f"{snr_db} dB")
    ax.set_title("BER versus spectral efficiency")
    ax.set_xlabel("Spectral efficiency (bits/symbol, ideal)")
    ax.set_ylabel("BER (zero-error estimates shown at simulation resolution)")
    ax.set_xticks(sorted(set(efficiencies)))
    ax.grid(True, which="both")
    ax.legend(title="Eb/N0")
    plt.tight_layout()
    plt.show()


# ------------------------------ Main --------------------------

def main():
    schemes = [("PSK", 4), ("PSK", 8), ("QAM", 16), ("QAM", 64)]
    constellation_data = {}
    print("=" * 76)
    print("NORMALIZED GRAY-CODED M-PSK AND SQUARE M-QAM")
    print("=" * 76)
    print("Average symbol energy verification (all constellations should be ~1):")

    for name, M in schemes:
        constellation, bits = modulation_info(name, M)
        constellation_data[(name, M)] = (constellation, bits)
        avg_energy = np.mean(np.abs(constellation) ** 2)
        min_distance = np.min([
            abs(constellation[i] - constellation[j])
            for i in range(M) for j in range(i + 1, M)
        ])
        print(f"{M:2d}-{name:3s}: average Es = {avg_energy:.6f}, "
              f"minimum distance = {min_distance:.6f}, "
              f"spectral efficiency = {int(np.log2(M))} bits/symbol")

    ber_results = {}
    ser_results = {}

    print("\nMonte Carlo simulation (SNR is Eb/N0 in dB)")
    print("-" * 76)
    print(f"{'Modulation':<12} {'Eb/N0':>7} {'BER':>14} {'SER':>14}")
    print("-" * 76)

    for name, M in schemes:
        k = int(np.log2(M))
        ideal_constellation, label_bits = constellation_data[(name, M)]
        ber_results[(name, M)] = []
        ser_results[(name, M)] = []

        tx_bits = RNG.integers(0, 2, size=(NUM_SYMBOLS, k), dtype=np.uint8)
        tx_symbols = map_bits(tx_bits, name, M)

        for snr_db in SNR_DB_LIST:
            rx_symbols = add_awgn(tx_symbols, snr_db, k)
            rx_bits, detected_indices = demap_symbols(rx_symbols, name, M)

            bit_errors = np.count_nonzero(rx_bits != tx_bits)
            symbol_errors = np.count_nonzero(np.any(rx_bits != tx_bits, axis=1))
            ber = bit_errors / (NUM_SYMBOLS * k)
            ser = symbol_errors / NUM_SYMBOLS

            ber_results[(name, M)].append(ber)
            ser_results[(name, M)].append(ser)
            print(f"{M}-{name:<8} {snr_db:7.1f} {ber:14.6e} {ser:14.6e}")

    print("\nNote: BER/SER values are Monte Carlo estimates and may vary with the random seed.")
    print("A measured BER of 0 means no bit errors occurred in this finite run, not zero true BER.")

    plot_constellations(schemes, constellation_data)
    plot_decision_regions(schemes, constellation_data)
    plot_ber_ser(schemes, SNR_DB_LIST, ber_results, ser_results)
    plot_ber_vs_efficiency(schemes, ber_results, SNR_DB_LIST)


if __name__ == "__main__":
    main()
