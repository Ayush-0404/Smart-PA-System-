# Smart PA System - Complete Technical Explanation

**Project:** Smart PA (Public Address) System with Audio Clarity Enhancement  
**Author:** Group 5 - Ayush Kumar (22125) & Dheeraj (22161)

---

## 📋 Executive Summary - 

This project implements an intelligent public address system that **automatically optimizes speech intelligibility** in noisy environments. It combines offline audio processing (clarity enhancement) with real-time adaptive gain control (dynamic leveling) to ensure announcements remain clear and audible regardless of ambient noise conditions.

### What Problem Does It Solve?
In real-world PA systems (airports, train stations, shopping malls), announcements often become unintelligible due to:
- Background noise masking speech frequencies
- Inconsistent playback levels
- Poor frequency balance in recordings
- Harsh sibilance and muddy low-mids

This system addresses all these issues automatically.

---

## 🎯 High-Level System Overview

### Signal Flow
```
Input Audio File (WAV)
    ↓
[Load & Normalize]
    ↓
[OFFLINE PROCESSING: AudioClarity Pipeline]
    ├─ Spectral Noise Gate
    ├─ Vocal-Focused EQ
    ├─ De-esser
    ├─ Multiband Compressor
    └─ Harmonic Exciter
    ↓
Pre-Processed Announcement
    ↓
[REAL-TIME PLAYBACK: SmartPAEngine]
    ├─ Mic Input → RMS Analysis
    ├─ Dynamic Gain Calculation
    ├─ Optional Adaptive Clarity Boost
    └─ Hard Limiter
    ↓
PA Output + Saved Files
    ├─ final_output.wav
    └─ audio_comparison.png
```

---

## 🔧 Component Breakdown

### 1. **AudioClarity Class** - Offline Processing Pipeline

This class contains five professional audio processing stages that transform a raw announcement into broadcast-quality speech.

#### 1.1 Spectral Noise Gate
**Purpose:** Remove background noise without artifacts

**How it works:**
- Uses Short-Time Fourier Transform (STFT) to convert audio to frequency domain
- Analyzes first 0.5 seconds to estimate noise floor (median across frequency bins)
- Creates a frequency-dependent gate that attenuates noise bins by -20dB (not complete muting)
- Applies frequency smoothing to avoid "musical" artifacts
- Reconstructs time-domain signal with inverse STFT

**Key Parameters:**
```python
nperseg = 2048          # FFT window size (46ms at 44.1kHz)
noverlap = nperseg // 2 # 50% overlap for smooth reconstruction
threshold_db = -35      # Gate threshold relative to noise floor
freq_smoothing = 5      # Smoothing across frequency bins
```

**Why it matters:** Traditional noise gates can create "pumping" artifacts. Spectral gating processes each frequency independently, preserving speech while reducing steady-state noise (HVAC, traffic, etc.).

**Mathematical Foundation:**
$$\text{Gate}[f, t] = \begin{cases} 
1.0 & \text{if } |X[f,t]| > \text{threshold}[f] \\
0.1 & \text{otherwise}
\end{cases}$$

Where $X[f,t]$ is the STFT magnitude at frequency $f$ and time $t$.

---

#### 1.2 Vocal-Focused EQ
**Purpose:** Enhance frequencies critical for speech intelligibility

**Processing Chain:**

1. **High-Pass Filter (80 Hz, 4th order Butterworth)**
   - Removes rumble, wind noise, mic handling
   - Cleans up low-frequency mud

2. **Presence Boost (2-4 kHz, +6dB)**
   - Critical range for consonants (S, T, K, P)
   - Enhances clarity and articulation
   - Uses bandpass filter + additive mixing

3. **Warmth Boost (150-300 Hz, +3dB)**
   - Adds body to voice without muddiness
   - Scaled by 0.5 to avoid excessive low-mid buildup

4. **Mud Cut (200-500 Hz, -3dB)**
   - Reduces "boxy" resonances
   - Clears space for upper mids

5. **Air Shelf (>6 kHz, +4dB)**
   - Adds "sparkle" and open-ness
   - Helps cut through noise
   - Scaled by 0.4 for subtlety

**Implementation:**
All filters use second-order sections (SOS) for numerical stability:
```python
sos_hp = signal.butter(4, 80, 'hp', fs=sr, output='sos')
audio = signal.sosfilt(sos_hp, audio)
```

**Why these frequencies?**
- **80 Hz cutoff:** Below fundamental frequency of human speech
- **2-4 kHz:** Formants and consonants live here (most intelligibility)
- **6+ kHz:** Air and clarity without harshness

---

#### 1.3 De-esser
**Purpose:** Control harsh sibilance (S, T, CH, SH sounds)

**How it works:**
1. High-pass filter at 7 kHz isolates sibilant frequencies
2. Hilbert transform extracts envelope (energy over time)
3. Smooth envelope with moving average (10ms window)
4. When envelope exceeds threshold, apply gain reduction
5. Subtract original sibilants, add reduced version back

**Key Parameters:**
```python
freq = 7000          # Sibilance frequency threshold
threshold = -20 dB   # Level at which de-essing starts
ratio = 3:1          # Compression ratio for excess sibilance
```

**Compression Formula:**
```
gain_reduction = 1 / (1 + ((excess - 1) / ratio))
```

**Why it matters:** PA systems with horn drivers can make sibilance painfully harsh. De-essing makes speech comfortable without losing clarity.

---

#### 1.4 Multiband Compressor
**Purpose:** Control dynamics separately across frequency ranges

**Band Split:**
- **Low:** < 250 Hz (bass/fundamentals)
- **Mid:** 250 Hz - 4 kHz (speech core)
- **High:** > 4 kHz (air/sibilance)

**Per-Band Compression:**
Each band has its own envelope follower with attack/release times:
```python
Low:   threshold=-18dB, ratio=2.5:1  (gentle, preserve warmth)
Mid:   threshold=-12dB, ratio=3.5:1  (aggressive, even speech)
High:  threshold=-10dB, ratio=2.0:1  (moderate, control peaks)
```

**Envelope Follower (Attack/Release):**
```python
if new_level > current_envelope:
    envelope = α_attack × new + (1 - α_attack) × current
else:
    envelope = α_release × new + (1 - α_release) × current
```

Where:
```
alpha_attack = 1 - exp(-1 / (t_attack * sample_rate))
```

**Recombination:**
```python
output = low_compressed + (mid_compressed × 1.2) + high_compressed
```
*Note: Mid boosted by 1.2× because speech energy concentrates here*

**Why it matters:** Single-band compression can make bass "pump" when speech gets loud. Multiband keeps each range consistent independently.

---

#### 1.5 Harmonic Exciter
**Purpose:** Add subtle harmonic content for "presence"

**How it works:**
1. Apply soft clipping via `tanh(audio × 1.5)` to generate harmonics
2. High-pass filter the distorted signal (>2 kHz)
3. Mix 15% of excited highs with 85% original

**Why use tanh?**
- Creates smooth, musical odd harmonics (3rd, 5th, 7th)
- Avoids harsh clipping artifacts
- Similar to analog saturation

**Mix Formula:**
```
output = (1 - alpha) * original + alpha * (original + 0.3 * excited_highs)
```
Where alpha = 0.15 (amount parameter)

**Why it matters:** Adds "air" and "sheen" that helps speech cut through without sounding processed.

---

### 2. **DynamicLeveler Class** - Real-Time Gain Control

**Purpose:** Automatically adjust playback volume based on ambient noise

**How it works:**

1. **RMS Calculation** (every audio frame):
```
RMS = sqrt( (1/N) * sum_{n=1}^N x[n]^2 )
```

2. **Smoothing** (prevents jumpy gain changes):
```
smoothed_RMS = (1 - alpha) * old + alpha * new_RMS
```
- alpha = 0.35 (smoothing factor)

3. **Gain Mapping** (noise → gain):
```
ratio = smoothed_RMS / RMS_ref
```
- Clamp ratio to [0.1, 12.0]
- Map logarithmically to gain range [min_gain, max_gain]

**Key Parameters:**
```python
smoothing_alpha = 0.35   # How fast gain responds (higher = faster)
rms_ref = 0.02           # Reference noise level
min_gain = 1.0x          # Quiet environment gain
max_gain = 6.0x          # Noisy environment gain
noise_floor = 0.001      # Silence detection threshold
```

**Silence Handling:**
If mic RMS < noise_floor → apply min_gain  
*(Prevents system from boosting when no mic is connected)*

**Why it matters:** In airports, train announcements compete with crowd noise. This automatically raises volume when needed, lowers when quiet.

---

### 3. **AdaptiveClarityBoost Class** - Real-Time High-Frequency Enhancement

**Purpose:** Add extra clarity when ambient noise increases

**How it works:**
1. Create high-shelf filter (>2.5 kHz) using stateful IIR filter
2. Map ambient noise RMS to boost amount:
```
boost = clip(noise_RMS * 10, 0.0, 0.3)
```
3. Smooth boost amount: boost = 0.9 x old + 0.1 x target
4. Apply: output = audio + highs x boost

**Why it matters:** When noise increases, high frequencies get masked first. Adaptive boost compensates in real-time.

---

### 4. **SmartPAEngine Class** - Real-Time Playback System

**Purpose:** Orchestrate real-time processing and playback

**Audio Callback (runs every ~3ms at blocksize=128):**
```python
def callback(indata, outdata, frames, time_info, status):
    # 1. Read microphone input
    mic = indata[:, 0]
    
    # 2. Update leveler with mic RMS
    noise_rms = self.leveler.update_from_frame(mic)
    
    # 3. Compute target gain
    target_gain = self.leveler.compute_gain()
    
    # 4. Smooth gain transition (60% smoothing)
    new_gain = 0.4 × current_gain + 0.6 × target_gain
    
    # 5. Update adaptive clarity boost
    self.clarity_boost.update_boost(noise_rms)
    
    # 6. Fill output buffer with announcement chunk
    announcement_chunk = self.announcement[pointer:pointer+frames]
    
    # 7. Apply adaptive clarity
    chunk = self.clarity_boost.apply(announcement_chunk)
    
    # 8. Apply dynamic gain
    chunk *= new_gain
    
    # 9. Hard limit to prevent clipping
    chunk = clip(chunk, -0.98, 0.98)
    
    # 10. Write to output
    outdata[:, 0] = chunk
```

**Why this architecture?**
- **Non-blocking:** Audio callback must return quickly (<3ms)
- **Stateful:** Leveler and clarity boost maintain internal state
- **Safe:** Hard limiter prevents speaker damage

---

## 📊 Visualization & Diagnostics

The `plot_comparison()` function generates a comprehensive analysis:

### Generated Plots:
1. **Waveforms** (Original vs Final)
   - Shows amplitude changes over time
   - Reveals dynamic range compression

2. **Spectrograms** (0-8 kHz range)
   - Time-frequency representation
   - Shows noise reduction and EQ effects
   - Highlights speech formants

3. **Frequency Spectrum** (Log scale, 50 Hz - 10 kHz)
   - Overlays original vs processed
   - Markers at key frequencies (80 Hz, 300 Hz, 3 kHz, 6 kHz)
   - Shows EQ curve applied

**Output Files:**
- `final_output.wav` - Processed audio after playback (with dynamic leveling applied)
- `audio_comparison.png` - Visual analysis (150 DPI)

---

## 🖥️ How to Run on Windows

### Prerequisites
```cmd
cd "c:\Users\Invincible\Downloads\FAE\FAE Project"
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib sounddevice soundfile
```

### Basic Usage
```cmd
python smart_pa.py --file announcement.wav --sr 44100
```

### Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--file` | Input audio file (required) | - |
| `--sr` | Sample rate in Hz | 44100 |
| `--no-clarity` | Skip offline clarity processing | Enabled |
| `--no-adaptive` | Disable real-time clarity boost | Enabled |
| `--no-graph` | Skip comparison graph generation | Generates |

### Example Commands

**Full processing (recommended):**
```cmd
python smart_pa.py --file "C:\Audio\announcement.wav"
```

**Test dynamic leveling only:**
```cmd
python smart_pa.py --file announcement.wav --no-clarity
```

**Silent processing (no graphs):**
```cmd
python smart_pa.py --file announcement.wav --no-graph
```

### Expected Behavior
1. Script loads and processes audio file
2. Prints processing steps with emoji indicators
3. Starts real-time playback with mic monitoring
4. Console shows real-time gain values (e.g., "⚡ Gain: 2.34")
5. After completion, displays gain statistics
6. Saves output files

### Troubleshooting

**"No module named sounddevice":**
```cmd
python -m pip install sounddevice --user
```

**"No audio devices found":**
- Check Windows audio settings
- Ensure microphone is enabled and set as default
- Try specifying device: modify code to add `device=X` parameter

**"Permission denied" on output files:**
- Close any programs using `final_output.wav` or `audio_comparison.png`
- Run cleanup: `del final_output.wav audio_comparison.png`

---

## 🎓 Key Concepts -  Sound Engineers

### 1. **Why Spectral Processing?**
Time-domain noise gates struggle with stationary noise. Spectral gating operates in frequency domain, allowing selective attenuation of noisy bins while preserving speech formants.

### 2. **Attack/Release in Digital Domain**
The envelope follower uses exponential smoothing:
```
y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
```
This mimics analog RC circuits. Attack/release times convert to alpha via:
```
alpha = 1 - exp(-1 / (t * fs))
```

### 3. **Multiband vs. Single-Band Compression**
Single-band: Bass energy can trigger gain reduction, making treble "duck"  
Multiband: Each band has independent threshold/ratio, preventing cross-band pumping

### 4. **Phase Considerations**
All filters are minimum-phase IIR (Butterworth). Fast but introduces phase shift. For linear-phase, use FIR filters (higher latency).

### 5. **RMS vs. Peak Metering**
RMS better represents perceived loudness:
- Peak: instantaneous maximum
- RMS: energy average (closer to human perception)

### 6. **Psychoacoustic Masking**
High frequencies mask easily in noise. The 2-4 kHz boost and adaptive clarity compensate for masking, improving intelligibility in the critical consonant range.

---

## ⚠️ Limitations & Considerations

### 1. **Noise Profile Estimation**
- **Issue:** Uses first 0.5 seconds for noise floor
- **Problem:** Fails if speech starts immediately or noise is non-stationary
- **Solution:** Implement continuous noise tracking or Voice Activity Detection (VAD)

### 2. **Microphone Placement**
- **Critical:** Mic must capture ambient noise, not PA output
- **Risk:** Feedback loop if mic picks up speakers
- **Recommendation:** Use directional mic pointed away from speakers

### 3. **Computational Efficiency**
- **Issue:** Python loops in envelope follower are slow
- **Impact:** May struggle with very low latency (<128 samples) or high sample rates
- **Solution:** Use Numba JIT compilation or Cython for hot paths

### 4. **Filter Phase Response**
- **Issue:** IIR filters cause phase distortion when recombining bands
- **Impact:** Minimal for speech, but measurable in frequency response
- **Solution:** Use linear-phase FIR filters if latency allows

### 5. **Gain Calibration**
- **Issue:** Gain mapping is heuristic, not calibrated to dB SPL
- **Impact:** May over/under compensate in different rooms
- **Solution:** Add calibration mode with reference tone and SPL meter

### 6. **Latency**
- **Current:** ~3-6ms (blocksize dependent)
- **For live use:** Acceptable for announcements, problematic for two-way communication

### 7. **Hard Limiter**
- **Issue:** Brick-wall limiting at ±0.98 can cause distortion
- **Solution:** Implement look-ahead limiter with soft-knee

---

##  Suggested Improvements & Experiments (future scope)

### Immediate Enhancements

#### 1. **Adaptive Noise Profiling**
Replace static noise estimation with continuous update:
```python
def update_noise_profile(current_profile, new_frame, alpha=0.01):
    # During non-speech segments only (use VAD)
    return (1 - alpha) * current_profile + alpha * np.median(new_frame, axis=1)
```

#### 2. **Voice Activity Detection (VAD)**
Prevent speaker audio from affecting leveler:
```python
def is_speech(frame, threshold=-40):
    spectral_centroid = compute_centroid(frame)
    return spectral_centroid > 1000  # Speech has higher centroid than noise
```

#### 3. **Look-Ahead Limiter**
More transparent peak control:
```python
def lookahead_limit(audio, threshold=0.95, lookahead_ms=5):
    # Analyze future samples, reduce gain before peaks
    # Avoids abrupt clipping distortion
```

#### 4. **GUI Controls**
Add real-time parameter adjustment:
- Gain range sliders (min/max)
- EQ band level controls
- Threshold visualizer
- Real-time spectrogram display

#### 5. **Auto-Calibration**
```python
def calibrate_gain_mapping(mic_input, reference_spl, duration=30):
    # Record ambient noise at known SPL
    # Compute optimal gain curve
    # Save calibration profile
```

### Advanced Experiments

#### 1. **Machine Learning Noise Suppression**
Replace spectral gate with ML model:
- Train on speech/noise datasets
- Use RNNoise or similar architecture
- Deploy via ONNX runtime

#### 2. **Frequency-Dependent Leveling**
Apply different gain curves per band:
```python
# Boost mids more than bass in noise
gain_low = compute_gain(rms) * 0.8
gain_mid = compute_gain(rms) * 1.2
gain_high = compute_gain(rms) * 1.0
```

#### 3. **Stereo Enhancement**
For stereo PA systems:
- Mid/Side processing
- Stereo widening on reverb/effects
- Mono compatibility check

#### 4. **Remote Monitoring**
Add web dashboard:
- Real-time gain graph
- Spectral analysis
- Alert system for feedback/clipping
- OSC/MIDI control integration

#### 5. **A/B Testing Framework**
```python
def ab_compare(original, processed, test_subjects):
    # Blind test protocol
    # Statistical analysis of intelligibility scores
    # Optimize parameters based on results
```

---

## 📐 Mathematical Deep Dive

### STFT (Short-Time Fourier Transform)
$$X[m, k] = \sum_{n=0}^{N-1} w[n] \cdot x[m \cdot H + n] \cdot e^{-j2\pi kn/N}$$

Where:
- $m$ = frame index
- $k$ = frequency bin
- $w[n]$ = window function (Hann)
- $H$ = hop size (overlap)
- $N$ = FFT size (nperseg)

### Butterworth Filter Transfer Function
$$H(s) = \frac{1}{\sqrt{1 + \left(\frac{s}{\omega_c}\right)^{2n}}}$$

Where:
- $\omega_c$ = cutoff frequency
- $n$ = filter order
- $s$ = complex frequency

### Compression Gain Curve
```
G_dB = 0, if L < T
G_dB = (T - L) * (1 - 1/R), if L >= T
```
Where:
- L = input level (dB)
- T = threshold (dB)
- R = ratio
- G = gain reduction (dB)

---

## Summary 

### Smart PA System: Intelligibility-First Announcement Processor

**Problem:** PA announcements become unintelligible in noisy environments

**Solution:** Two-stage processing
1. **Offline Enhancement:** Noise gate → EQ → De-ess → Multiband compression → Exciter
2. **Real-Time Adaptation:** Mic-driven dynamic gain + adaptive clarity boost

**Key Features:**
- ✅ Spectral noise reduction (STFT-based)
- ✅ Vocal-focused EQ (2-4 kHz presence boost)
- ✅ Multiband compression (independent band control)
- ✅ Real-time leveling (automatic volume adjustment)
- ✅ Diagnostic visualization (waveform/spectrogram/spectrum)

**Technologies:** Python, NumPy, SciPy (signal processing), SoundDevice (real-time audio)

**Applications:** Airports, train stations, shopping malls, stadiums, emergency systems

---

## 📚 References & Further Reading

### Digital Signal Processing
- **Oppenheim & Schafer** - "Discrete-Time Signal Processing"
- **Smith, Julius O.** - "Introduction to Digital Filters" (online book)

### Audio Engineering
- **Eargle, John** - "The Microphone Book"
- **Katz, Bob** - "Mastering Audio: The Art and the Science"

### Speech Processing
- **Rabiner & Schafer** - "Theory and Applications of Digital Speech Processing"
- **Loizou, Philipos** - "Speech Enhancement: Theory and Practice"

### Real-Time Audio Programming
- **Boulanger & Lazzarini** - "The Audio Programming Book"
- **Pirkle, Will** - "Designing Audio Effect Plugins in C++"

---

## 🤝 Acknowledgments

This project demonstrates practical application of:
- Broadcast audio processing techniques
- Live sound reinforcement principles
- Adaptive signal processing
- Real-time embedded audio systems

Perfect for educational purposes in:
- Audio engineering courses
- DSP laboratories
- Live sound design classes
- Acoustics and psychoacoustics studies

  **Made by Ayush & Dheeraj**

---

**End of Technical Documentation**

*For questions or improvements, modify the code and experiment! The best way to learn is by tweaking parameters and listening critically.*
