"""
Smart PA System — Complete with Audio Clarity Enhancement
- Pre-processes announcement for maximum intelligibility
- Applies dynamic loudness leveling in real-time
- Optional real-time adaptive clarity boost
"""

import argparse
import threading
import time
import sys
import math
import os
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ----------------------------------------------------------
# AUDIO CLARITY PRE-PROCESSING
# ----------------------------------------------------------

class AudioClarity:
    """Professional audio clarity enhancement pipeline"""
    
    @staticmethod
    def spectral_gate(audio, sr, threshold_db=-40, freq_smoothing=5):
        """Reduce background noise using spectral gating"""
        # STFT parameters
        nperseg = 2048
        noverlap = nperseg // 2
        
        f, t, Zxx = signal.stft(audio, sr, nperseg=nperseg, noverlap=noverlap)
        
        # Compute magnitude and phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Estimate noise floor (use first 0.5 seconds)
        noise_frames = int(0.5 * sr / (nperseg - noverlap))
        noise_profile = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Create gate
        threshold = noise_profile * (10 ** (threshold_db / 20))
        gate = np.where(magnitude > threshold, 1.0, 0.1)  # Reduce by -20dB instead of muting
        
        # Smooth gate over frequency
        for i in range(gate.shape[1]):
            gate[:, i] = uniform_filter1d(gate[:, i], size=freq_smoothing)
        
        # Apply gate
        Zxx_gated = magnitude * gate * np.exp(1j * phase)
        
        # Inverse STFT
        _, audio_gated = signal.istft(Zxx_gated, sr, nperseg=nperseg, noverlap=noverlap)
        
        return audio_gated[:len(audio)]
    
    @staticmethod
    def vocal_eq(audio, sr):
        """Apply vocal-focused EQ for speech intelligibility"""
        
        # HIGH-PASS: Remove rumble below 80 Hz
        sos_hp = signal.butter(4, 80, 'hp', fs=sr, output='sos')
        audio = signal.sosfilt(sos_hp, audio)
        
        # BOOST: 2-4 kHz (presence/clarity range) +6dB
        # This is where consonants and vocal articulation live
        sos_presence = signal.butter(2, [2000, 4000], 'bp', fs=sr, output='sos')
        presence = signal.sosfilt(sos_presence, audio)
        audio = audio + presence * (10 ** (6 / 20) - 1)
        
        # BOOST: 150-300 Hz (warmth/body) +3dB
        sos_low_shelf = signal.butter(2, [150, 300], 'bp', fs=sr, output='sos')
        body = signal.sosfilt(sos_low_shelf, audio)
        audio = audio + body * (10 ** (3 / 20) - 1) * 0.5
        
        # CUT: 200-500 Hz (muddiness) -3dB
        sos_mud = signal.butter(2, [200, 500], 'bp', fs=sr, output='sos')
        mud = signal.sosfilt(sos_mud, audio)
        audio = audio - mud * 0.3
        
        # HIGH SHELF: Boost above 6 kHz +4dB (air/sparkle)
        sos_air = signal.butter(2, 6000, 'hp', fs=sr, output='sos')
        air = signal.sosfilt(sos_air, audio)
        audio = audio + air * (10 ** (4 / 20) - 1) * 0.4
        
        return audio
    
    @staticmethod
    def de_esser(audio, sr, freq=7000, threshold=-20, ratio=3):
        """Reduce harsh sibilance (S/T/CH sounds)"""
        # Isolate sibilant frequencies
        sos_sib = signal.butter(4, freq, 'hp', fs=sr, output='sos')
        sibilants = signal.sosfilt(sos_sib, audio)
        
        # Detect envelope
        envelope = np.abs(signal.hilbert(sibilants))
        envelope_smooth = uniform_filter1d(envelope, size=int(0.01 * sr))
        
        # Create gain reduction
        threshold_linear = 10 ** (threshold / 20)
        gain_reduction = np.ones_like(envelope_smooth)
        
        mask = envelope_smooth > threshold_linear
        excess = envelope_smooth[mask] / threshold_linear
        gain_reduction[mask] = 1.0 / (1.0 + (excess - 1.0) / ratio)
        
        # Apply to sibilants only
        sibilants_reduced = sibilants * gain_reduction
        
        # Subtract original sibilants and add reduced version
        audio_de_essed = audio - sibilants + sibilants_reduced
        
        return audio_de_essed
    
    @staticmethod
    def multiband_compressor(audio, sr):
        """Compress different frequency bands for even dynamics"""
        
        # Split into 3 bands
        sos_low = signal.butter(4, 250, 'lp', fs=sr, output='sos')
        sos_mid = signal.butter(4, [250, 4000], 'bp', fs=sr, output='sos')
        sos_high = signal.butter(4, 4000, 'hp', fs=sr, output='sos')
        
        low = signal.sosfilt(sos_low, audio)
        mid = signal.sosfilt(sos_mid, audio)
        high = signal.sosfilt(sos_high, audio)
        
        # Compress each band
        def compress_band(band, threshold=-15, ratio=3, attack=0.005, release=0.1):
            envelope = np.abs(signal.hilbert(band))
            envelope_smooth = envelope.copy()
            
            # Simple envelope follower
            alpha_attack = 1 - np.exp(-1 / (attack * sr))
            alpha_release = 1 - np.exp(-1 / (release * sr))
            
            for i in range(1, len(envelope_smooth)):
                if envelope[i] > envelope_smooth[i-1]:
                    envelope_smooth[i] = alpha_attack * envelope[i] + (1 - alpha_attack) * envelope_smooth[i-1]
                else:
                    envelope_smooth[i] = alpha_release * envelope[i] + (1 - alpha_release) * envelope_smooth[i-1]
            
            threshold_linear = 10 ** (threshold / 20)
            gain = np.ones_like(envelope_smooth)
            
            mask = envelope_smooth > threshold_linear
            excess_db = 20 * np.log10(envelope_smooth[mask] / threshold_linear + 1e-12)
            reduction_db = excess_db * (1 - 1/ratio)
            gain[mask] = 10 ** (-reduction_db / 20)
            
            return band * gain
        
        low_comp = compress_band(low, threshold=-18, ratio=2.5)
        mid_comp = compress_band(mid, threshold=-12, ratio=3.5)
        high_comp = compress_band(high, threshold=-10, ratio=2.0)
        
        # Recombine with slight boost to mid (where speech lives)
        return low_comp + mid_comp * 1.2 + high_comp
    
    @staticmethod
    def harmonic_exciter(audio, sr, amount=0.15):
        """Add subtle harmonic distortion for presence"""
        # Generate harmonics via soft clipping
        excited = np.tanh(audio * 1.5) * 0.8
        
        # High-pass the excited signal
        sos_hp = signal.butter(4, 2000, 'hp', fs=sr, output='sos')
        excited_highs = signal.sosfilt(sos_hp, excited)
        
        # Mix with original
        return audio * (1 - amount) + (audio + excited_highs * 0.3) * amount
    
    @staticmethod
    def process_announcement(audio, sr, noise_reduction=True, verbose=True):
        """Complete clarity enhancement pipeline"""
        
        if verbose:
            print("🎙️  Audio Clarity Enhancement Pipeline")
            print("=" * 50)
        
        original_peak = np.max(np.abs(audio))
        
        # Step 1: Noise Reduction
        if noise_reduction:
            if verbose:
                print("1️⃣  Applying spectral noise gate...")
            audio = AudioClarity.spectral_gate(audio, sr, threshold_db=-35)
        
        # Step 2: Vocal EQ
        if verbose:
            print("2️⃣  Applying vocal-focused EQ...")
        audio = AudioClarity.vocal_eq(audio, sr)
        
        # Step 3: De-esser
        if verbose:
            print("3️⃣  Reducing harsh sibilance...")
        audio = AudioClarity.de_esser(audio, sr)
        
        # Step 4: Multiband Compression
        if verbose:
            print("4️⃣  Applying multiband compression...")
        audio = AudioClarity.multiband_compressor(audio, sr)
        
        # Step 5: Harmonic Exciter
        if verbose:
            print("5️⃣  Adding harmonic excitement...")
        audio = AudioClarity.harmonic_exciter(audio, sr, amount=0.12)
        
        # Normalize to original peak
        current_peak = np.max(np.abs(audio))
        if current_peak > 0:
            audio = audio * (original_peak / current_peak)
        
        if verbose:
            print("✅ Clarity enhancement complete!")
            print("=" * 50)
        
        return audio

# ----------------------------------------------------------
# FILE CLEANUP UTILITY
# ----------------------------------------------------------

def cleanup_previous_files():
    """Remove previously generated output files"""
    files_to_remove = ["final_output.wav", "audio_comparison.png"]
    
    for filename in files_to_remove:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"🗑️  Removed old file: {filename}")
            except Exception as e:
                print(f"⚠️  Could not remove {filename}: {e}")

# ----------------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------------

def plot_comparison(original, processed, sr, filename="comparison.png"):
    """Generate and save waveform and spectrogram comparison"""
    
    print("\n📊 Generating comparison graphs...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Time arrays
    time_orig = np.arange(len(original)) / sr
    time_proc = np.arange(len(processed)) / sr
    
    # Color scheme
    color_orig = '#FF6B6B'
    color_proc = '#4ECDC4'
    
    # ========== WAVEFORMS ==========
    
    # Original waveform
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_orig, original, color=color_orig, linewidth=0.5, alpha=0.8)
    ax1.set_title('Original Audio - Waveform', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#F8F9FA')
    
    # Processed waveform
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_proc, processed, color=color_proc, linewidth=0.5, alpha=0.8)
    ax2.set_title('Final Output - Waveform (Clarity + Dynamic Leveling)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#F8F9FA')
    
    # ========== SPECTROGRAMS ==========
    
    # Original spectrogram
    ax3 = fig.add_subplot(gs[1, 0])
    f_orig, t_orig, Sxx_orig = signal.spectrogram(original, sr, nperseg=2048, noverlap=1536)
    Sxx_orig_db = 10 * np.log10(Sxx_orig + 1e-10)
    im1 = ax3.pcolormesh(t_orig, f_orig, Sxx_orig_db, shading='gouraud', 
                          cmap='hot', vmin=-80, vmax=0)
    ax3.set_title('Original Audio - Spectrogram', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Frequency (Hz)', fontsize=10)
    ax3.set_xlabel('Time (seconds)', fontsize=10)
    ax3.set_ylim(0, 8000)  # Focus on speech range
    plt.colorbar(im1, ax=ax3, label='Power (dB)')
    
    # Processed spectrogram
    ax4 = fig.add_subplot(gs[1, 1])
    f_proc, t_proc, Sxx_proc = signal.spectrogram(processed, sr, nperseg=2048, noverlap=1536)
    Sxx_proc_db = 10 * np.log10(Sxx_proc + 1e-10)
    im2 = ax4.pcolormesh(t_proc, f_proc, Sxx_proc_db, shading='gouraud', 
                          cmap='hot', vmin=-80, vmax=0)
    ax4.set_title('Final Output - Spectrogram', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Frequency (Hz)', fontsize=10)
    ax4.set_xlabel('Time (seconds)', fontsize=10)
    ax4.set_ylim(0, 8000)
    plt.colorbar(im2, ax=ax4, label='Power (dB)')
    
    # ========== FREQUENCY SPECTRUM ==========
    
    # Compute FFT for both
    n_fft = 8192
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    fft_orig = np.fft.rfft(original, n=n_fft)
    magnitude_orig = 20 * np.log10(np.abs(fft_orig) + 1e-10)
    
    fft_proc = np.fft.rfft(processed, n=n_fft)
    magnitude_proc = 20 * np.log10(np.abs(fft_proc) + 1e-10)
    
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(freqs, magnitude_orig, color=color_orig, linewidth=1.5, 
             alpha=0.7, label='Original Input')
    ax5.plot(freqs, magnitude_proc, color=color_proc, linewidth=1.5, 
             alpha=0.7, label='Final Output (with Clarity + Dynamic Leveling)')
    ax5.set_title('Frequency Spectrum Comparison', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Frequency (Hz)', fontsize=10)
    ax5.set_ylabel('Magnitude (dB)', fontsize=10)
    ax5.set_xlim(50, 10000)
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend(fontsize=11, loc='upper right')
    ax5.set_facecolor('#F8F9FA')
    
    # Add key frequency markers
    speech_ranges = [
        (80, 'Low cutoff'),
        (300, 'Warmth'),
        (3000, 'Presence/Clarity'),
        (6000, 'Air/Sparkle')
    ]
    for freq, label in speech_ranges:
        ax5.axvline(x=freq, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax5.text(freq, ax5.get_ylim()[1] - 5, label, rotation=0, 
                fontsize=8, ha='center', alpha=0.7)
    
    # Overall title
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(f'Smart PA System - Audio Clarity Enhancement Comparison\n{timestamp}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    plt.subplots_adjust(top=0.96)  # Make room for suptitle
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparison saved: {filename}")
    plt.close()

# ----------------------------------------------------------
# RESAMPLING
# ----------------------------------------------------------

def resample_if_needed(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x
    num = int(round(len(x) * float(dst_sr) / src_sr))
    return signal.resample(x, num)

# ----------------------------------------------------------
# DYNAMIC LEVELER
# ----------------------------------------------------------

class DynamicLeveler:
    def __init__(self, smoothing_alpha=0.35, rms_ref=0.02,
                 min_gain=1.0, max_gain=6.0, noise_floor=0.001):                                                          # Min GAIN = 1.0x
        self.alpha = smoothing_alpha
        self.smoothed_rms = 0.0
        self.rms_ref = rms_ref
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.noise_floor = noise_floor  # Threshold for "silence"

    def update_from_frame(self, frame):
        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12)
        self.smoothed_rms = (1 - self.alpha) * self.smoothed_rms + self.alpha * rms
        return self.smoothed_rms

    def compute_gain(self):
        val = self.smoothed_rms
        
        # If essentially silent (no mic input), use minimum gain
        if val < self.noise_floor:
            return self.min_gain
        
        # Scale gain based on noise level
        ratio = max(0.1, min(val / (self.rms_ref + 1e-12), 12.0))
        t = math.log10(ratio) / math.log10(10.0)
        mapped = self.min_gain + (self.max_gain - self.min_gain) * (0.5 + 0.5 * t)
        return max(self.min_gain, min(self.max_gain, mapped))

# ----------------------------------------------------------
# ADAPTIVE REAL-TIME CLARITY BOOST (Optional)
# ----------------------------------------------------------

class AdaptiveClarityBoost:
    """Lightweight real-time high-frequency boost based on noise level"""
    
    def __init__(self, sr, blocksize):
        self.sr = sr
        # Simple high-shelf filter for presence boost
        self.sos = signal.butter(2, 2500, 'hp', fs=sr, output='sos')
        self.zi = signal.sosfilt_zi(self.sos)
        self.boost_amount = 0.0
    
    def update_boost(self, noise_rms):
        """Increase clarity boost as noise increases"""
        # Map noise RMS to boost amount (0.0 to 0.3)
        target = np.clip(noise_rms * 10, 0.0, 0.3)
        # Smooth
        self.boost_amount = 0.9 * self.boost_amount + 0.1 * target
    
    def apply(self, audio_frame):
        """Apply adaptive high-frequency boost"""
        if self.boost_amount < 0.01:
            return audio_frame
        
        # Filter high frequencies
        highs, self.zi = signal.sosfilt(self.sos, audio_frame, zi=self.zi)
        
        # Blend
        return audio_frame + highs * self.boost_amount

# ----------------------------------------------------------
# PA ENGINE
# ----------------------------------------------------------

class SmartPAEngine:
    def __init__(self, announcement, sr, blocksize=128, device=None, 
                 adaptive_clarity=True):
        self.announcement = announcement.astype(np.float32)
        self.sr = sr
        self.blocksize = blocksize
        self.pointer = 0
        self.device = device

        self._gain = 1.0
        self._gain_lock = threading.Lock()
        
        # Track gain history for statistics
        self.gain_history = []
        
        # Store final output for saving
        self.final_output = []

        self.leveler = DynamicLeveler(
            smoothing_alpha=0.35,
            rms_ref=0.02,
            min_gain=1.0,                                                                                              # Min Gain = 0.5
            max_gain=6.0,
            noise_floor=0.001
        )
        
        # Optional adaptive clarity
        self.adaptive_clarity = adaptive_clarity
        if adaptive_clarity:
            self.clarity_boost = AdaptiveClarityBoost(sr, blocksize)

    def set_gain(self, g):
        with self._gain_lock:
            self._gain = g

    def get_gain(self):
        with self._gain_lock:
            return self._gain

    def _fill_output(self, outdata, frames):
        start = self.pointer
        end = start + frames
        out = np.zeros(frames, dtype=np.float32)

        if start < len(self.announcement):
            available = min(len(self.announcement) - start, frames)
            out[:available] = self.announcement[start:start + available]

        self.pointer = end

        # Apply adaptive clarity boost if enabled
        if self.adaptive_clarity:
            out = self.clarity_boost.apply(out)

        # Apply LEVELER gain
        out *= self.get_gain()

        # HARD LIMITER
        out = np.clip(out, -0.98, 0.98)
        
        # Store final output
        self.final_output.extend(out.tolist())

        outdata[:, 0] = out

    def callback(self, indata, outdata, frames, time_info, status):
        mic = indata[:, 0]
        noise_rms = self.leveler.update_from_frame(mic)

        target_gain = self.leveler.compute_gain()

        smooth_alpha = 0.60
        new_gain = (1 - smooth_alpha) * self.get_gain() + smooth_alpha * target_gain
        self.set_gain(new_gain)
        
        # Track gain for statistics
        self.gain_history.append(new_gain)
        
        # Update adaptive clarity boost based on noise
        if self.adaptive_clarity:
            self.clarity_boost.update_boost(noise_rms)

        self._fill_output(outdata, frames)

    def run(self, save_output=False, output_filename="final_output.wav"):
        try:
            with sd.Stream(
                samplerate=self.sr,
                blocksize=self.blocksize,
                dtype='float32',
                channels=1,
                callback=self.callback,
                device=self.device
            ):
                print("\n🔊 Running Smart PA System...")
                print("🎤 Listening to microphone for ambient noise...")
                print("💡 TIP: Make noise near the mic to see dynamic leveling in action!")
                print("📊 Real-time stats:\n")
                while self.pointer < len(self.announcement):
                    time.sleep(0.05)
                    clarity_status = ""
                    if self.adaptive_clarity:
                        clarity_status = f" | Clarity: {self.clarity_boost.boost_amount:.2f}"
                    sys.stdout.write(f"\r⚡ Gain: {self.get_gain():.2f}{clarity_status}  ")
                    sys.stdout.flush()
            
            # Save final output if requested
            if save_output and len(self.final_output) > 0:
                print(f"\n\n💾 Saving final output: {output_filename}")
                final_array = np.array(self.final_output, dtype=np.float32)
                sf.write(output_filename, final_array, self.sr)
                print(f"✅ Final output saved!")
            
            # Calculate and display statistics
            print("\n" + "="*60)
            print("📈 GAIN STATISTICS")
            print("="*60)
            
            if len(self.gain_history) > 0:
                avg_gain = np.mean(self.gain_history)
                min_gain = np.min(self.gain_history)
                max_gain = np.max(self.gain_history)
                std_gain = np.std(self.gain_history)
                
                print(f"   Average Gain Applied: {avg_gain:.2f}x")
                print(f"   Minimum Gain:         {min_gain:.2f}x")
                print(f"   Maximum Gain:         {max_gain:.2f}x")
                print(f"   Std Deviation:        {std_gain:.2f}")
                print(f"   Total Samples:        {len(self.gain_history)}")
                
                # Info about gain behavior
                if max_gain <= 0.51:
                    print("\n   ℹ️  Gain at 0.5x: No microphone input detected")
                    print("   ⚠️  Output volume is reduced - check microphone connection")
                elif avg_gain < 1.0:
                    print("\n   ℹ️  Very quiet environment - minimal amplification")
                elif avg_gain >= 3.0:
                    print("\n   ℹ️  High ambient noise - significant gain boost applied")
            else:
                print("   No gain data collected")
            
            print("="*60)
            print("✅ Playback finished.\n")
            
            return np.array(self.final_output, dtype=np.float32)

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            raise

# ----------------------------------------------------------
# LOADING WITH PRE-PROCESSING
# ----------------------------------------------------------

def load_and_process_announcement(filename, target_sr, apply_clarity=True):
    """Load audio file and apply clarity enhancement"""
    
    print(f"\n📂 Loading: {filename}")
    data, sr = sf.read(filename, always_2d=False)

    if data.ndim > 1:
        print("   Converting to mono...")
        data = np.mean(data, axis=1)

    if sr != target_sr:
        print(f"   Resampling: {sr} Hz → {target_sr} Hz")
        data = resample_if_needed(data, sr, target_sr)
        sr = target_sr

    # Normalize
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.9
    
    # Store original for comparison
    original_data = data.copy()
    
    # Apply clarity enhancement
    if apply_clarity:
        print("")
        data = AudioClarity.process_announcement(data, sr)
        print("")
    else:
        print("⚠️  Clarity enhancement DISABLED\n")

    return data.astype(np.float32), sr, original_data.astype(np.float32)

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart PA System with Audio Clarity Enhancement"
    )
    parser.add_argument('--file', required=True, help='Input audio file')
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate')
    parser.add_argument('--no-clarity', action='store_true', 
                        help='Disable clarity pre-processing')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Disable adaptive real-time clarity boost')
    parser.add_argument('--no-graph', action='store_true',
                        help='Disable saving comparison graph')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("🎤 SMART PA SYSTEM — Audio Clarity + Dynamic Leveling")
    print("="*60)
    
    # Clean up previous output files
    cleanup_previous_files()
    print("")
    
    # Load and pre-process
    ann, sr, original = load_and_process_announcement(
        args.file, 
        args.sr, 
        apply_clarity=not args.no_clarity
    )
    
    # Create engine
    engine = SmartPAEngine(
        announcement=ann, 
        sr=sr, 
        blocksize=128,
        adaptive_clarity=not args.no_adaptive
    )
    
    # Run and get final output
    final_output = engine.run(
        save_output=True,
        output_filename="final_output.wav"
    )
    
    # Generate comparison graph: original vs final output
    if not args.no_graph and len(final_output) > 0:
        #print("\n📊 Generating comparison graphs...")
        # Ensure both arrays are same length for comparison
        min_len = min(len(original), len(final_output))
        plot_comparison(
            original[:min_len], 
            final_output[:min_len], 
            sr, 
            filename="audio_comparison.png"
        )
        print("✅ Comparison complete!\n")

if __name__ == "__main__":
    main()