# Smart PA System

A complete Python-based Public Address (PA) system for speech clarity enhancement and adaptive loudness leveling.

## Features
- **Audio Clarity Pipeline:**
  - Spectral noise gating
  - Vocal-focused EQ
  - De-essing
  - Multiband compression
  - Harmonic exciter
- **Real-Time Dynamic Leveling:**
  - Monitors ambient noise via microphone
  - Automatically adjusts playback gain
- **Optional Adaptive Clarity Boost:**
  - Real-time high-frequency enhancement based on noise
- **Diagnostics:**
  - Saves processed audio and comparison graphs

## Usage

### Prerequisites
Install dependencies:
```cmd
python -m pip install numpy scipy matplotlib sounddevice soundfile
```

### Run the System
```cmd
python smart_pa.py --file <announcement.wav> --sr 44100
```

#### Options
- `--no-clarity` : Skip clarity enhancement
- `--no-adaptive` : Disable adaptive clarity boost
- `--no-graph` : Skip comparison graph

### Output
- `final_output.wav` : Processed announcement
- `audio_comparison.png` : Visual comparison (waveform, spectrogram, spectrum)

## How It Works
1. Loads and pre-processes the input audio for maximum intelligibility
2. During playback, monitors ambient noise and adapts output loudness in real-time
3. Optionally boosts clarity when noise increases
4. Saves results and diagnostics for review

## Applications
- Airports, train stations, malls, stadiums, emergency systems

## Documentation
See `PROJECT_EXPLANATION.md` for a full technical deep dive.

## License
MIT
