# Critical Pointers for Simulating Realistic Reverb, Noise, and Audio Mixing

A comprehensive guide for generating high-quality synthetic audio datasets for speech separation and enhancement tasks.

---

## Table of Contents

1. [Reverb Simulation - Acoustic Realism](#1-reverb-simulation---acoustic-realism)
2. [Noise Addition - Signal Quality](#2-noise-addition---signal-quality)
3. [Audio Mixing - Level Management](#3-audio-mixing---level-management)
4. [Multichannel Considerations](#4-multichannel-considerations)
5. [Validation & Quality Checks](#5-validation--quality-checks)
6. [Dataset Balance](#6-dataset-balance)
7. [Debugging Tips](#7-debugging-tips)
8. [Pre-Flight Checklist](#8-pre-flight-checklist)

---

## 1. Reverb Simulation - Acoustic Realism

### 1.1 Separate RIRs for Each Source

**The Golden Rule:** Each audio source must have its own Room Impulse Response (RIR) based on its spatial position.

#### ❌ Wrong Approach
```python
# Acoustically INCORRECT - mixing before reverb
mix = s1 + s2
reverb_mix = convolve(mix, RIR)  
```

#### ✅ Correct Approach
```python
# Acoustically CORRECT - reverb before mixing
s1_reverb = convolve(s1, RIR_from_position_1)
s2_reverb = convolve(s2, RIR_from_position_2)
mix = s1_reverb + s2_reverb
```

**Why it matters:** In real acoustic environments, each speaker's sound travels a unique path to each microphone, resulting in different arrival times, reflections, and frequency responses.

---

### 1.2 Room Geometry Constraints

Proper spatial placement is critical for realistic simulations.

#### Minimum Safety Distances

| Element | Minimum Distance | Reason |
|---------|-----------------|--------|
| Source → Wall | ≥ 0.3m | Avoid boundary effects |
| Source → Mic Array | ≥ 0.5m | Avoid near-field artifacts |
| Source ↔ Source | ≥ 0.5m | Realistic speaker separation |
| Mic Array → Wall | ≥ 0.2m | Prevent array from hitting boundaries |

#### Implementation Example
```python
# Add safety margins when placing sources
margin = 0.3
valid_x_range = [margin, room_sz[0] - margin]
valid_y_range = [margin, room_sz[1] - margin]
valid_z_range = [margin, room_sz[2] - margin]

# Example: Random source placement with constraints
src_x = random.uniform(valid_x_range[0], valid_x_range[1])
src_y = random.uniform(valid_y_range[0], valid_y_range[1])
src_z = random.uniform(1.0, valid_z_range[1])  # Typical speaker height
```

#### Room Size Recommendations

| Room Type | Dimensions (L×W×H) | Use Case |
|-----------|-------------------|----------|
| Small Office | 4–6m × 3–5m × 2.5–3m | Close-talk scenarios |
| Meeting Room | 6–8m × 5–7m × 2.7–3.5m | Conference systems |
| Large Hall | 8–12m × 7–10m × 3–4m | Far-field speech |

---

### 1.3 T60 (Reverberation Time) Selection

T60 is the time it takes for sound to decay by 60 dB. Choose values appropriate to your target environment.

#### T60 Reference Guide

| Environment | T60 Range | Characteristics | Recommended for |
|-------------|-----------|----------------|-----------------|
| Anechoic Chamber | < 0.1s | No reflections | Testing only (unrealistic) |
| Recording Studio | 0.1–0.3s | Very dry, controlled | Clean speech tasks |
| Living Room | 0.3–0.5s | Moderate reverb | Residential scenarios |
| Office/Classroom | 0.4–0.7s | Noticeable echo | Typical indoor use |
| Conference Hall | 0.7–1.2s | Strong reverb | Large spaces |
| Cathedral | 2–10s | Extreme reverb | Not recommended for speech |

#### Implementation
```python
# Balanced approach for speech separation
T60 = random.uniform(0.20, 0.85)  # 200–850 ms

# Avoid extremes:
# - T60 < 0.15s: Sounds artificial, lacking spatial context
# - T60 > 1.5s: Speech becomes unintelligible, overlaps excessively
```

**Critical Warning:** T60 > 1.5s makes speech separation nearly impossible due to excessive temporal smearing.

---

### 1.4 RIR Length (Tmax) Configuration

The maximum simulation time determines how much of the reverb tail is captured.

#### Rule of Thumb
```
Tmax ≥ 3 × T60
```

#### Examples
```python
# For T60 = 0.4s
Tmax = 1.6  # Captures 4× T60 (full decay + margin)

# For T60 = 0.8s
Tmax = 2.5  # Still captures adequate tail
```

#### Trade-offs

| Tmax Setting | Pros | Cons |
|--------------|------|------|
| Too short (< 2×T60) | Faster computation | Cuts off reverb tail, unrealistic |
| Optimal (3-4×T60) | Full decay captured | Balanced performance |
| Too long (> 5×T60) | Unnecessary | Wasted computation, larger files |

---

## 2. Noise Addition - Signal Quality

### 2.1 SNR (Signal-to-Noise Ratio) Range Selection

SNR determines how much noise is present relative to the speech signal.

#### SNR Reference Table

| SNR (dB) | Environment | Perceptual Quality | Use Case |
|----------|-------------|-------------------|----------|
| 0–5 dB | Very noisy | Difficult to understand | Cafeteria, busy street |
| 5–10 dB | Noisy | Intelligible but effortful | Factory floor, traffic |
| 10–15 dB | Moderate | Clear with background | Open office, restaurant |
| 15–20 dB | Quiet | Comfortable listening | Quiet room, library |
| 20–25 dB | Very quiet | Nearly clean | Studio, quiet office |
| > 25 dB | Pristine | Essentially noiseless | Anechoic, lab conditions |

#### Implementation
```python
# Realistic range for robust model training
SNR = random.uniform(0, 25)  # dB

# For challenging scenarios (stress testing)
SNR = random.uniform(-5, 15)  # Includes negative SNRs
```

---

### 2.2 Correct Noise Addition Order

**Critical:** Always add noise AFTER applying reverb, not before.

#### ❌ Wrong Order
```python
s_noisy = s_clean + noise
s_reverb = convolve(s_noisy, RIR)  # Incorrect: reverb affects noise unrealistically
```

#### ✅ Correct Order
```python
s_reverb = convolve(s_clean, RIR)
s_noisy = s_reverb + noise * scale  # Correct: simulates environmental noise
```

**Why?** In real environments:
1. Speech is reverberated by the room
2. Microphones capture reverberated speech + ambient noise
3. Noise is typically diffuse and doesn't undergo the same RIR as point sources

---

### 2.3 Noise Scaling Formula

Properly scale noise to achieve target SNR.

#### Mathematical Foundation
```
SNR (dB) = 10 × log₁₀(P_signal / P_noise)

Where:
P_signal = mean(signal²)  # Signal power
P_noise = mean(noise²)    # Noise power
```

#### Implementation
```python
# Calculate powers
noise_power = np.mean(noise_segment**2)
signal_power = np.mean(multichannel_signal**2, axis=1, keepdims=True)

# Calculate scaling factor
target_SNR = 15  # dB
scale = np.sqrt(signal_power / (noise_power * 10**(target_SNR/10) + 1e-9))

# Apply scaled noise
noisy_signal = multichannel_signal + scale * noise_segment[np.newaxis, :]
```

**Important Notes:**
- `keepdims=True`: Essential for proper broadcasting with multichannel audio
- `+ 1e-9`: Prevents division by zero for silent segments
- `[np.newaxis, :]`: Broadcasts noise across all channels

---

### 2.4 Noise Types and Diversity

Include multiple noise categories for robust model training.

#### Recommended Noise Categories

| Category | Examples | Characteristics | Percentage in Dataset |
|----------|----------|----------------|----------------------|
| **Stationary** | White noise, pink noise, HVAC hum, fan | Constant spectral characteristics | 20–30% |
| **Non-stationary** | Babble, cafeteria, traffic, music | Time-varying spectrum | 40–50% |
| **Impulsive** | Door slams, keyboard clicks, dishes | Short duration, high energy | 10–20% |
| **Speech-like** | Background conversation, TV | Confuser for separation models | 20–30% |

#### Noise Collection Tips
```python
# Load diverse noise sources
noise_types = {
    'stationary': ['hvac.wav', 'fan_noise.wav', 'white_noise.wav'],
    'non_stationary': ['cafeteria.wav', 'traffic.wav', 'babble.wav'],
    'impulsive': ['keyboard.wav', 'door_slam.wav'],
    'speech_like': ['background_tv.wav', 'distant_speech.wav']
}

# Randomly select from different categories
category = random.choice(list(noise_types.keys()))
noise_file = random.choice(noise_types[category])
```

---

## 3. Audio Mixing - Level Management

### 3.1 Sample Rate Consistency

**Rule #1:** ALL audio signals must have identical sample rates before mixing.

#### Common Sample Rates

| Sample Rate | Use Case | Quality |
|-------------|----------|---------|
| 8 kHz | Telephony | Telephone quality |
| 16 kHz | Speech processing | Standard for speech tasks |
| 22.05 kHz | Low-quality music | Acceptable for voice |
| 44.1 kHz | CD quality | High fidelity audio |
| 48 kHz | Professional audio | Broadcast standard |

#### Implementation
```python
import librosa

# Resample all sources to target rate
target_fs = 16000

s1 = librosa.resample(s1_original, orig_sr=32000, target_sr=target_fs)
s2 = librosa.resample(s2_original, orig_sr=32000, target_sr=target_fs)
noise = librosa.resample(noise_original, orig_sr=44100, target_sr=target_fs)

# Verify before mixing
assert s1.shape[0] > 0 and s2.shape[0] > 0
```

**Warning:** Mixing signals at different sample rates produces garbage output!

---

### 3.2 Length Matching After Reverb

Convolution extends signal length. Always match lengths before mixing.

#### Why Length Changes
```
Original signal length: N samples
RIR length: M samples
After convolution: N + M - 1 samples
```

#### Implementation
```python
# After applying reverb to both speakers
s1_reverb = apply_reverb(s1, RIR1)  # Shape: (8, N+M-1)
s2_reverb = apply_reverb(s2, RIR2)  # Shape: (8, N+M-1) - likely different!

# Find minimum length
min_len = min(s1_reverb.shape[1], s2_reverb.shape[1])

# Truncate to match
s1_reverb = s1_reverb[:, :min_len]
s2_reverb = s2_reverb[:, :min_len]

# Now safe to mix
mix = s1_reverb + s2_reverb
```

---

### 3.3 Normalization Strategy

Prevent clipping and maintain consistent loudness across dataset.

#### dBFS (Decibels relative to Full Scale)

| Target dBFS | Peak Level | Use Case | Clipping Risk |
|-------------|-----------|----------|---------------|
| -30 to -25 | 0.032–0.056 | Very safe, max headroom | None |
| -25 to -20 | 0.056–0.100 | Professional standard | Very low |
| -20 to -15 | 0.100–0.178 | Louder, less headroom | Low |
| -15 to -10 | 0.178–0.316 | Aggressive levels | Moderate |
| > -10 | > 0.316 | Danger zone | High |

#### Implementation
```python
def normalize_to_dbfs(signal, target_dbfs=-25):
    """
    Normalize audio to target dBFS level
    
    Args:
        signal: Audio array (any shape)
        target_dbfs: Target level in dB (default: -25)
    
    Returns:
        Normalized audio
    """
    rms = np.sqrt(np.mean(signal**2, axis=-1, keepdims=True))
    
    # Avoid division by zero
    rms = np.maximum(rms, 1e-9)
    
    # Calculate scaling factor
    target_linear = 10**(target_dbfs / 20)
    scale = target_linear / rms
    
    return signal * scale

# Usage
normalized_mix = normalize_to_dbfs(mixed_signal, target_dbfs=-25)
```

---

### 3.4 Clipping Prevention

Always check and prevent digital clipping.

#### Detection and Prevention
```python
def prevent_clipping(signal, safety_margin=0.95):
    """
    Scale down signal if it exceeds safe peak level
    
    Args:
        signal: Audio array
        safety_margin: Maximum allowed peak (0-1), default 0.95
    
    Returns:
        Safe signal with no clipping
    """
    peak = np.max(np.abs(signal))
    
    if peak > safety_margin:
        scale = safety_margin / peak
        signal = signal * scale
        print(f"Warning: Scaled down by {scale:.3f} to prevent clipping")
    
    return signal

# Usage in pipeline
mixed_signal = s1_reverb + s2_reverb + noise
mixed_signal = prevent_clipping(mixed_signal, safety_margin=0.95)
```

#### Visual Clipping Check
```python
import matplotlib.pyplot as plt

def check_clipping_visual(signal, fs, duration=1.0):
    """Plot waveform to visually inspect clipping"""
    samples = int(duration * fs)
    t = np.arange(samples) / fs
    
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal[:samples])
    plt.axhline(y=1.0, color='r', linestyle='--', label='Clipping threshold')
    plt.axhline(y=-1.0, color='r', linestyle='--')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.title('Waveform - Check for Clipping')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

## 4. Multichannel Considerations

### 4.1 Channel Order Consistency

**Critical:** Maintain consistent array shape convention throughout your pipeline.

#### Standard Conventions

| Convention | Shape | Usage | Libraries |
|------------|-------|-------|-----------|
| Channels-first | `(n_channels, n_samples)` | PyTorch default | PyTorch, custom processing |
| Channels-last | `(n_samples, n_channels)` | File I/O standard | soundfile, librosa, scipy |

#### Implementation
```python
# Internal processing: (n_channels, n_samples)
multichannel_signal.shape  # (8, 160000)

# When saving to file: transpose to (n_samples, n_channels)
import soundfile as sf
sf.write('output.wav', multichannel_signal.T, samplerate=16000)

# When loading: transpose back
loaded_signal, fs = sf.read('output.wav')  # (160000, 8)
loaded_signal = loaded_signal.T  # (8, 160000) for processing
```

#### Verification Function
```python
def verify_multichannel_shape(signal, expected_channels=8, convention='channels_first'):
    """Verify and report multichannel array shape"""
    
    if convention == 'channels_first':
        assert signal.shape[0] == expected_channels, \
            f"Expected {expected_channels} channels, got {signal.shape[0]}"
        print(f"✓ Shape: ({signal.shape[0]} channels, {signal.shape[1]} samples)")
    
    elif convention == 'channels_last':
        assert signal.shape[1] == expected_channels, \
            f"Expected {expected_channels} channels, got {signal.shape[1]}"
        print(f"✓ Shape: ({signal.shape[0]} samples, {signal.shape[1]} channels)")
    
    return True
```

---

### 4.2 Per-Channel Noise (Advanced)

For maximum realism, add independent noise to each microphone channel.

#### Simple Approach (Correlated Noise)
```python
# Same noise to all channels - computationally efficient
noisy = signal + noise[np.newaxis, :]  # Broadcast across channels
```

#### Advanced Approach (Uncorrelated Noise)
```python
# Independent noise per channel - more realistic
noisy = np.zeros_like(signal)

for ch in range(n_channels):
    # Different noise segment for each channel
    start = random.randint(0, len(noise) - signal.shape[1])
    noise_ch = noise[start:start + signal.shape[1]]
    
    # Calculate SNR per channel
    signal_power_ch = np.mean(signal[ch]**2)
    noise_power_ch = np.mean(noise_ch**2)
    scale_ch = np.sqrt(signal_power_ch / (noise_power_ch * 10**(SNR/10)))
    
    noisy[ch] = signal[ch] + scale_ch * noise_ch
```

**Trade-off:** 
- Correlated noise: Faster, good enough for most tasks
- Uncorrelated noise: More realistic, better for advanced beamforming evaluation

---

### 4.3 Microphone Array Geometry

Common array configurations and their applications.

#### Array Types

| Configuration | Description | Use Case | Code Example |
|--------------|-------------|----------|--------------|
| **Linear** | Mics in a line | Direction finding | `x = np.linspace(-0.1, 0.1, 8)` |
| **Circular** | Mics in a circle | 360° coverage | `angles = np.linspace(0, 2π, 8)` |
| **Planar** | Mics in 2D grid | 2D localization | Grid of (x,y) positions |
| **Spherical** | Mics on sphere | 3D spatial audio | Spherical coordinates |

#### Circular Array Implementation (Most Common)
```python
def create_circular_array(n_mics=8, radius=0.08):
    """
    Create circular microphone array
    
    Args:
        n_mics: Number of microphones
        radius: Array radius in meters (typically 0.05-0.15m)
    
    Returns:
        Array of shape (n_mics, 3) with [x, y, z] positions
    """
    angles = np.linspace(0, 2*np.pi, n_mics, endpoint=False)
    
    positions = np.stack([
        radius * np.cos(angles),  # X coordinates
        radius * np.sin(angles),  # Y coordinates
        np.zeros(n_mics)          # Z coordinates (flat array)
    ], axis=1)
    
    return positions

# Example: 8-mic array with 8cm radius
mic_array = create_circular_array(n_mics=8, radius=0.08)
```

---

## 5. Validation & Quality Checks

### 5.1 Numerical Validity Checks

Always verify signal integrity before saving.

#### Essential Checks
```python
def validate_signal(signal, name="signal"):
    """
    Comprehensive signal validation
    
    Args:
        signal: Audio array to validate
        name: Signal identifier for error messages
    
    Returns:
        True if valid, raises AssertionError otherwise
    """
    # 1. Check for NaN (Not a Number)
    assert not np.isnan(signal).any(), f"{name} contains NaN values!"
    
    # 2. Check for Inf (Infinity)
    assert not np.isinf(signal).any(), f"{name} contains Inf values!"
    
    # 3. Check shape is reasonable
    assert signal.size > 0, f"{name} is empty!"
    
    # 4. Check dynamic range
    peak = np.max(np.abs(signal))
    assert peak > 1e-6, f"{name} is too quiet (peak={peak:.2e})"
    assert peak <= 1.0, f"{name} is clipping (peak={peak:.3f})"
    
    # 5. Check for DC offset
    dc_offset = np.mean(signal)
    if abs(dc_offset) > 0.01:
        print(f"Warning: {name} has DC offset of {dc_offset:.4f}")
    
    print(f"✓ {name} validation passed")
    return True

# Usage
validate_signal(mixed_signal, name="8-channel mix")
validate_signal(s1_reference, name="Speaker 1")
```

---

### 5.2 Audio Quality Metrics

Compute key metrics to ensure consistent quality.

#### Metric Computation
```python
def compute_audio_metrics(signal, fs=16000, name="signal"):
    """
    Compute standard audio quality metrics
    
    Args:
        signal: Audio array (mono or multichannel)
        fs: Sample rate
        name: Signal identifier
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Ensure mono for some calculations
    if signal.ndim > 1:
        signal_mono = np.mean(signal, axis=0)
    else:
        signal_mono = signal
    
    # Peak amplitude
    metrics['peak'] = np.max(np.abs(signal))
    metrics['peak_db'] = 20 * np.log10(metrics['peak'] + 1e-9)
    
    # RMS (Root Mean Square)
    metrics['rms'] = np.sqrt(np.mean(signal**2))
    metrics['rms_db'] = 20 * np.log10(metrics['rms'] + 1e-9)
    
    # Crest factor (peak-to-RMS ratio)
    metrics['crest_factor'] = metrics['peak'] / (metrics['rms'] + 1e-9)
    metrics['crest_factor_db'] = metrics['peak_db'] - metrics['rms_db']
    
    # Duration
    metrics['duration_s'] = len(signal_mono) / fs
    
    # Dynamic range
    metrics['dynamic_range_db'] = metrics['peak_db'] - 20*np.log10(np.min(np.abs(signal_mono[signal_mono != 0])) + 1e-9)
    
    # Print report
    print(f"\n=== {name} Metrics ===")
    print(f"Duration: {metrics['duration_s']:.2f} s")
    print(f"Peak: {metrics['peak']:.4f} ({metrics['peak_db']:.2f} dBFS)")
    print(f"RMS: {metrics['rms']:.4f} ({metrics['rms_db']:.2f} dBFS)")
    print(f"Crest Factor: {metrics['crest_factor']:.2f} ({metrics['crest_factor_db']:.2f} dB)")
    
    return metrics

# Usage
metrics = compute_audio_metrics(mixed_signal, fs=16000, name="Mixed Signal")
```

---

### 5.3 Common Failure Modes and Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| **Silent output** | RIRs are all zeros | Check room configuration, ensure T60 > 0 |
| **Distorted audio** | Clipping from excessive gain | Apply normalization, reduce levels |
| **Robotic sound** | Sample rate mismatch | Verify all signals at same fs before mixing |
| **Too much echo** | T60 too high or Tmax too long | Reduce T60 to < 0.85s, check Tmax = 3×T60 |
| **No spatial separation** | All sources at same position | Verify different source positions |
| **Noise dominates** | SNR too low | Check SNR calculation, increase target SNR |
| **Channels uncorrelated** | Wrong convolution | Ensure same signal convolved with all mic RIRs |

---

### 5.4 Listen to Your Data!

**Critical:** Always listen to generated samples before committing to full dataset generation.

```python
def save_debug_samples(mix, s1, s2, idx, output_dir="debug_samples"):
    """Save a mix and its references for manual inspection"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    sf.write(f"{output_dir}/sample_{idx:03d}_mix_ch0.wav", mix[0], fs)
    sf.write(f"{output_dir}/sample_{idx:03d}_mix_ch4.wav", mix[4], fs)  # Opposite mic
    sf.write(f"{output_dir}/sample_{idx:03d}_s1.wav", s1, fs)
    sf.write(f"{output_dir}/sample_{idx:03d}_s2.wav", s2, fs)
    
    print(f"Debug samples saved to {output_dir}/")

# Generate and save first few samples
for i in range(5):
    mix, s1, s2, room, t60, snr = create_one_mixture(files[i], files[i+1])
    save_debug_samples(mix, s1, s2, i)
    print(f"Sample {i}: Room={room}, T60={t60:.2f}s, SNR={snr:.1f}dB")

print("\n⚠️ LISTEN TO debug_samples/ BEFORE PROCEEDING!")
```

---

## 6. Dataset Balance

### 6.1 Parameter Variation Strategy

Balanced variation helps models generalize to unseen conditions.

#### Recommended Distributions

```python
import numpy as np

# Room sizes: Uniform distribution across ranges
def sample_room_size():
    return [
        random.uniform(5.0, 12.0),   # Length: small office to large hall
        random.uniform(4.5, 10.0),   # Width
        random.uniform(2.7, 4.0)     # Height: standard ceiling to high ceiling
    ]

# T60: Slightly biased toward moderate values
def sample_t60():
    # More samples in 0.3-0.6s range (typical rooms)
    if random.random() < 0.6:
        return random.uniform(0.3, 0.6)  # 60% of samples
    else:
        return random.uniform(0.2, 0.85)  # 40% of samples

# SNR: Uniform across challenging to clean
def sample_snr():
    return random.uniform(0, 25)  # Equal probability across range

# Source distance: Logarithmic distribution (more near-field samples)
def sample_source_distance(room_sz):
    max_dist = min(room_sz[0], room_sz[1]) / 2 - 0.5
    # Log-uniform: more samples at smaller distances
    log_dist = random.uniform(np.log(0.5), np.log(max_dist))
    return np.exp(log_dist)
```

---

### 6.2 Dataset Split Strategy

Proper train/validation/test splits with considerations for generalization.

#### Recommended Splits

| Split | Percentage | Number of Samples (if total=23k) | Purpose |
|-------|-----------|----------------------------------|---------|
| Train | 80-85% | 18,400–19,550 | Model training |
| Validation | 10-15% | 2,300–3,450 | Hyperparameter tuning |
| Test | 5-10% | 1,150–2,300 | Final evaluation |

#### Implementation
```python
def generate_dataset_with_splits(total_samples=23000, 
                                 train_ratio=0.85, 
                                 val_ratio=0.10):
    """
    Generate dataset with proper splits
    
    Args:
        total_samples: Total number of mixtures to generate
        train_ratio: Fraction for training (default: 0.85)
        val_ratio: Fraction for validation (default: 0.10)
    """
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    n_test = total_samples - n_train - n_val
    
    print(f"Dataset split:")
    print(f"  Train: {n_train} ({train_ratio*100:.1f}%)")
    print(f"  Val:   {n_val} ({val_ratio*100:.1f}%)")
    print(f"  Test:  {n_test} ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    for i in range(total_samples):
        # Determine split
        if i < n_train:
            split = "tr"
        elif i < n_train + n_val:
            split = "cv"
        else:
            split = "tt"
        
        # Generate mixture...
        # (rest of generation code)
```

---

### 6.3 Stratified Sampling (Advanced)

Ensure balanced representation of different conditions.

```python
def stratified_parameter_sampling():
    """
    Sample parameters with stratification to ensure balanced coverage
    """
    # Define bins for each parameter
    t60_bins = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.85)]
    snr_bins = [(0, 8), (8, 17), (17, 25)]
    room_size_bins = ['small', 'medium', 'large']
    
    # Rotate through bins to ensure balance
    t60_bin = random.choice(t60_bins)
    snr_bin = random.choice(snr_bins)
    
    # Sample within selected bins
    T60 = random.uniform(t60_bin[0], t60_bin[1])
    SNR = random.uniform(snr_bin[0], snr_bin[1])
    
    # Room size
    room_category = random.choice(room_size_bins)
    if room_category == 'small':
        room_sz = [random.uniform(5, 7), random.uniform(4.5, 6), random.uniform(2.7, 3.2)]
    elif room_category == 'medium':
        room_sz = [random.uniform(7, 9), random.uniform(6, 8), random.uniform(3.0, 3.5)]
    else:  # large
        room_sz = [random.uniform(9, 12), random.uniform(8, 10), random.uniform(3.5, 4.0)]
    
    return T60, SNR, room_sz
```

---

## 7. Debugging Tips

### 7.1 Single Sample Testing

**Golden Rule:** Test with ONE sample before generating thousands.

```python
def test_single_mixture(speech_file1, speech_file2):
    """
    Generate and thoroughly test a single mixture
    """
    print("=" * 60)
    print("SINGLE MIXTURE TEST")
    print("=" * 60)
    
    # Generate
    mix, s1, s2, room, t60, snr = create_one_mixture(speech_file1, speech_file2)
    
    # Validate
    print("\n1. Shape Validation:")
    print(f"   Mix shape: {mix.shape}")
    print(f"   S1 shape: {s1.shape}")
    print(f"   S2 shape: {s2.shape}")
    
    # Check for issues
    print("\n2. Numerical Checks:")
    validate_signal(mix, "Mix")
    validate_signal(s1, "S1")
    validate_signal(s2, "S2")
    
    # Metrics
    print("\n3. Quality Metrics:")
    compute_audio_metrics(mix[0], name="Mix (Channel 0)")
    compute_audio_metrics(s1, name="S1 Reference")
    
    # Parameters
    print("\n4. Generation Parameters:")
    print(f"   Room: {[f'{x:.2f}' for x in room]} meters")
    print(f"   T60: {t60:.3f} seconds")
    print(f"   SNR: {snr:.2f} dB")
    
    # Save for listening
    print("\n5. Saving debug files...")
    sf.write("debug_mix_ch0.wav", mix[0], 16000)
    sf.write("debug_mix_ch4.wav", mix[4], 16000)
    sf.write("debug_s1.wav", s1, 16000)
    sf.write("debug_s2.wav", s2, 16000)
    
    print("\n✓ Test complete! Listen to debug_*.wav files")
    print("=" * 60)

# Run test
clean_files = list(Path(wsj_clean_dir).rglob("*.wav"))
test_single_mixture(str(clean_files[0]), str(clean_files[1]))
```

---

### 7.2 Progressive Testing

Test increasingly complex scenarios before full generation.

```python
# Phase 1: Test reverb only (no noise)
def test_reverb_only():
    # Temporarily set SNR very high
    original_snr_range = (0, 25)
    test_snr_range = (100, 100)  # Essentially no noise
    # Generate samples...

# Phase 2: Test noise only (no reverb)  
def test_noise_only():
    # Temporarily set T60 very low
    original_t60_range = (0.2, 0.85)
    test_t60_range = (0.05, 0.05)  # Essentially anechoic
    # Generate samples...

# Phase 3: Test combined
def test_combined():
    # Use normal parameters
    # Generate samples...
```

---

### 7.3 Visualization Tools

```python
import matplotlib.pyplot as plt

def visualize_multichannel(signal, fs=16000, duration=2.0):
    """
    Plot multichannel signal for visual inspection
    
    Args:
        signal: Multichannel array (n_channels, n_samples)
        fs: Sample rate
        duration: Duration to plot in seconds
    """
    n_channels = signal.shape[0]
    samples = int(duration * fs)
    t = np.arange(samples) / fs
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels))
    
    for ch in range(n_channels):
        axes[ch].plot(t, signal[ch, :samples])
        axes[ch].set_ylabel(f'Ch {ch}')
        axes[ch].grid(True)
        axes[ch].set_ylim([-1, 1])
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('8-Channel Mixture')
    plt.tight_layout()
    plt.savefig('multichannel_waveform.png', dpi=150)
    plt.close()
    print("Saved multichannel_waveform.png")

def plot_spectrogram(signal, fs=16000, title="Spectrogram"):
    """Plot spectrogram for frequency analysis"""
    from scipy import signal as sp_signal
    
    f, t, Sxx = sp_signal.spectrogram(signal, fs, nperseg=512)
    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.colorbar(label='Power (dB)')
    plt.ylim([0, 8000])  # Focus on speech range
    plt.savefig('spectrogram.png', dpi=150)
    plt.close()
    print("Saved spectrogram.png")

# Usage
visualize_multichannel(mix_8ch, fs=16000, duration=3.0)
plot_spectrogram(mix_8ch[0], fs=16000, title="Mix Channel 0")
```

---

### 7.4 Error Logging

Implement comprehensive logging for debugging failures.

```python
import logging
from datetime import datetime

# Setup logging
log_filename = f"dataset_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def create_mixture_with_logging(path1, path2, idx):
    """Wrapper with comprehensive error logging"""
    try:
        mix, s1, s2, room, t60, snr = create_one_mixture(path1, path2)
        
        # Log success
        logging.info(f"Sample {idx}: SUCCESS - T60={t60:.3f}s, SNR={snr:.1f}dB, "
                    f"Room={[f'{x:.1f}' for x in room]}")
        
        return mix, s1, s2, room, t60, snr
        
    except Exception as e:
        # Log detailed error
        logging.error(f"Sample {idx}: FAILED")
        logging.error(f"  Files: {path1}, {path2}")
        logging.error(f"  Error: {type(e).__name__}: {str(e)}")
        logging.error(f"  Traceback: {traceback.format_exc()}")
        
        raise  # Re-raise to handle upstream

# Usage in generation loop
for i in range(total_samples):
    try:
        mix, s1, s2, room, t60, snr = create_mixture_with_logging(
            str(spk1), str(spk2), i
        )
        # Save files...
    except Exception:
        logging.warning(f"Skipping sample {i} due to error")
        continue
```

---

## 8. Pre-Flight Checklist

Before generating your full dataset, verify all items:

### ✅ Data Preparation
- [ ] All speech files are accessible and readable
- [ ] All noise files are loaded and resampled
- [ ] File lists are shuffled (for speaker randomness)
- [ ] Output directories are created and writable

### ✅ Parameter Configuration
- [ ] Target sample rate is set (8kHz or 16kHz recommended)
- [ ] Room dimensions allow safe placement (min 5×4.5×2.7m)
- [ ] T60 range is realistic (0.2–0.85s for speech)
- [ ] SNR range matches target scenarios (0–25dB typical)
- [ ] Number of microphones matches your model (8 recommended)

### ✅ Acoustic Realism
- [ ] Separate RIRs computed for each speaker
- [ ] Speaker positions differ spatially (≥60° angular separation)
- [ ] Microphone array geometry is correct
- [ ] Reverb applied before noise addition
- [ ] RIR length (Tmax) ≥ 3×T60

### ✅ Audio Quality
- [ ] All signals at same sample rate before mixing
- [ ] Normalization applied (target: -25 to -20 dBFS)
- [ ] Clipping prevention implemented
- [ ] Length matching after convolution
- [ ] NaN/Inf checks in place

### ✅ Testing
- [ ] Generated and listened to 1 test sample
- [ ] Generated and inspected 5–10 samples
- [ ] Visualized waveforms (no clipping visible)
- [ ] Checked spectrograms (realistic frequency content)
- [ ] Verified all three outputs (mix, s1, s2) sound correct

### ✅ Performance
- [ ] Estimated generation time for full dataset
- [ ] Sufficient disk space available (estimate: 2GB per 1000 samples)
- [ ] Error handling and logging implemented
- [ ] Progress monitoring set up (print every 500 samples)

### ✅ Dataset Balance
- [ ] Train/val/test splits defined
- [ ] Parameter distributions planned
- [ ] Stratified sampling considered (if needed)

---

## Example: Complete Pipeline

Here's a complete example incorporating all best practices:

```python
import numpy as np
import soundfile as sf
import librosa
import random
import logging
from pathlib import Path

# Configuration
CONFIG = {
    'fs': 16000,
    'n_mics': 8,
    'array_radius': 0.08,
    'total_samples': 23000,
    'train_ratio': 0.85,
    'val_ratio': 0.10,
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_and_generate_dataset():
    """Complete dataset generation with all safety checks"""
    
    # 1. Pre-flight checks
    logger.info("=" * 60)
    logger.info("PRE-FLIGHT CHECKS")
    logger.info("=" * 60)
    
    # Check input files
    clean_files = list(Path(wsj_clean_dir).rglob("*.wav"))
    logger.info(f"✓ Found {len(clean_files)} clean speech files")
    assert len(clean_files) >= 100, "Need at least 100 speech files!"
    
    # Load noise
    logger.info("Loading noise files...")
    noises = load_noise_files(noise_dir, CONFIG['fs'])
    logger.info(f"✓ Loaded {len(noises)} noise files")
    assert len(noises) >= 5, "Need at least 5 noise files!"
    
    # 2. Test single sample
    logger.info("\n" + "=" * 60)
    logger.info("TESTING SINGLE SAMPLE")
    logger.info("=" * 60)
    test_single_mixture(str(clean_files[0]), str(clean_files[1]))
    
    response = input("\nDoes the test sample sound good? (y/n): ")
    if response.lower() != 'y':
        logger.error("Fix issues before proceeding!")
        return
    
    # 3. Generate dataset
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING FULL DATASET")
    logger.info("=" * 60)
    
    n_train = int(CONFIG['total_samples'] * CONFIG['train_ratio'])
    n_val = int(CONFIG['total_samples'] * CONFIG['val_ratio'])
    
    success_count = 0
    failure_count = 0
    
    for i in range(CONFIG['total_samples']):
        try:
            # Random speaker pair
            spk1, spk2 = random.sample(clean_files, 2)
            
            # Generate
            mix, s1, s2, room, t60, snr = create_one_mixture(str(spk1), str(spk2))
            
            # Validate
            validate_signal(mix, f"Mix {i}")
            validate_signal(s1, f"S1 {i}")
            validate_signal(s2, f"S2 {i}")
            
            # Determine split
            split = "tr" if i < n_train else ("cv" if i < n_train + n_val else "tt")
            uid = f"{split}{i:06d}"
            
            # Save
            sf.write(f"{output_dir}/mix/{uid}_mix.wav", mix.T, CONFIG['fs'])
            sf.write(f"{output_dir}/s1/{uid}_s1.wav", s1, CONFIG['fs'])
            sf.write(f"{output_dir}/s2/{uid}_s2.wav", s2, CONFIG['fs'])
            
            success_count += 1
            
            # Progress
            if (i + 1) % 500 == 0:
                logger.info(f"Progress: {i+1}/{CONFIG['total_samples']} "
                           f"(Success: {success_count}, Failed: {failure_count})")
            
        except Exception as e:
            failure_count += 1
            logger.error(f"Failed on sample {i}: {e}")
            continue
    
    # 4. Final report
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total successful: {success_count}/{CONFIG['total_samples']}")
    logger.info(f"Total failed: {failure_count}")
    logger.info(f"Success rate: {100*success_count/CONFIG['total_samples']:.1f}%")
    logger.info(f"Dataset location: {output_dir}")

# Run
if __name__ == "__main__":
    validate_and_generate_dataset()
```

---

## Quick Reference Card

### Critical Rules Summary

1. **Reverb**: Separate RIR per speaker → convolve → then mix
2. **Noise**: Add AFTER reverb, not before
3. **Sample Rate**: ALL signals must match before mixing
4. **Normalization**: Target -25 to -20 dBFS
5. **Room Size**: Min 5×4.5×2.7m with 0.3m margins
6. **T60**: Keep 0.2–0.85s for speech intelligibility
7. **SNR**: 0–25 dB for realistic scenarios
8. **Testing**: ALWAYS test 1 sample before full generation

### Common Mistakes to Avoid

| ❌ Don't | ✅ Do |
|---------|-------|
| Mix then reverb | Reverb each source separately then mix |
| Add noise before reverb | Add noise after reverb |
| Mix different sample rates | Resample all to same fs first |
| Place sources at same position | Ensure spatial separation ≥ 0.5m |
| Skip validation checks | Validate every generated sample |
| Generate all before listening | Test first few samples manually |

---

## Additional Resources

### Recommended Papers
- **Room Acoustics**: Allen & Berkley, "Image method for efficiently simulating small-room acoustics" (1979)
- **Speech Enhancement**: Loizou, "Speech Enhancement: Theory and Practice" (2007)
- **Microphone Arrays**: Benesty et al., "Microphone Array Signal Processing" (2008)

### Useful Libraries
- **gpuRIR**: Fast GPU-based room impulse response simulator
- **pyroomacoustics**: Python room acoustics simulation
- **librosa**: Audio processing and resampling
- **soundfile**: High-quality audio I/O

### Validation Tools
- **Audacity**: Visual inspection of waveforms
- **Praat**: Speech analysis and spectrogram viewing
- **PESQ/STOI**: Objective speech quality metrics

---

## Conclusion

Generating realistic synthetic audio data requires attention to acoustic principles, careful parameter selection, and thorough validation. Follow this guide to create high-quality datasets that will help your models generalize to real-world scenarios.

**Remember**: The quality of your training data directly impacts model performance. It's worth spending time to get it right!

---
## Training
Default epochs: Changed from 100 → 10 epochs
Save interval: Changed from 5 → 2 epochs (so you'll have checkpoints at epochs 2, 4, 6, 8, 10)
Quick Start Command:

```
python nbc_train.py \
    --dataset_root WSJ0_8ch_noisy_reverb \
    --batch_size 4 \
    --epochs 10
```
---

---
**Document Version**: 1.0  
**Last Updated**: November 2025  
**Author**: Audio Processing Best Practices Guide  
**License**: Open for educational and research purposes