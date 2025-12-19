import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, square

# --- Page Config ---
st.set_page_config(
    page_title="Downsampling & Aliasing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS for Embedding ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stCheckbox { margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- Title & Intro ---
st.markdown("### ⬇️ Downsampling & Aliasing")
st.markdown(
    "Why do we need to Low-Pass filter a signal before reducing its sampling rate? "
    "Toggle the filter below to hear and see the difference."
)

# --- Parameters ---
fs = 16000          # sampling frequency [Hz]
f0 = 150            # square-wave fundamental [Hz]
duty = 60           # duty cycle [%]
duration = 0.5      # seconds

# --- Controls ---
with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        factor = st.slider("Downsampling Factor (×)", 1, 16, 4, step=1)
    with c2:
        apply_filter = st.checkbox("Apply Anti-aliasing Filter (Pre-decimation)", value=True)

# --- Computation ---
# 1. Generate Signal
N = int(np.round(duration * fs))
t = np.arange(N) / fs
# Square wave (rich in harmonics)
sig = square(2*np.pi*f0*t, duty=duty/100.0).astype(np.float64)
sig = sig / np.max(np.abs(sig))  # normalize

# 2. Filter Logic
def butter_lowpass(cutoff_hz, fs, order=8):
    return butter(order, cutoff_hz/(0.5*fs), btype="low", output="ba")

if factor < 1: factor = 1
fs_ds = fs // factor # New sampling rate

if apply_filter and factor > 1:
    ny_new = 0.5 * fs_ds
    cutoff = 0.9 * ny_new  # cutoff just below new Nyquist
    b, a = butter_lowpass(cutoff, fs, order=8)
    sig_proc = filtfilt(b, a, sig)
else:
    sig_proc = sig # No filter: High freqs will fold back (alias)

# 3. Downsample (Decimate)
sig_ds = sig_proc[::factor]

# --- FFT Helper ---
def db_spectrum(x, fs):
    N_len = len(x)
    if N_len < 4: return np.array([0]), np.array([-120])
    win = np.hanning(N_len)
    X = np.fft.rfft(x * win)
    mag = (2.0 / N_len) * np.abs(X) / np.mean(win)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    f_axis = np.fft.rfftfreq(N_len, 1.0/fs)
    return f_axis, mag_db

# --- Plotting Setup ---
plt.rcParams.update({'font.size': 8})
def style_plot(fig, ax):
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.tick_params(colors='gray')
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')
    ax.title.set_color('gray')

# --- Plot 1: Time Domain ---
fig1, ax1 = plt.subplots(figsize=(6, 3))
show_time = min(N, int(0.02 * fs)) # 20ms zoom
# Plot original or filtered background
if apply_filter and factor > 1:
    ax1.plot(t[:show_time], sig_proc[:show_time], color="#cccccc", label="Filtered Input", lw=1.5, alpha=0.7)
else:
    ax1.plot(t[:show_time], sig[:show_time], color="#cccccc", label="Original Input", lw=1.5, alpha=0.7)

# Plot Downsampled points
t_ds = np.arange(len(sig_ds)) / fs_ds
points_to_show = int(show_time * fs_ds / fs)
ax1.plot(t_ds[:points_to_show], sig_ds[:points_to_show], 
         "o-", color="#ff4b4b", markersize=4, label=f"Downsampled (fs={fs_ds})", lw=1)

ax1.set_title("Time Domain (20ms Zoom)")
ax1.set_xlabel("Time [s]")
ax1.legend(frameon=False, loc="upper right")
style_plot(fig1, ax1)

# --- Plot 2: Frequency Domain ---
fig2, ax2 = plt.subplots(figsize=(6, 3))
Nfft = min(1 << 14, N)
f_o, db_o = db_spectrum(sig[:Nfft], fs) # Spectrum of pure original
f_d, db_d = db_spectrum(sig_ds[:min(len(sig_ds), Nfft)], fs_ds) # Spectrum of result

ax2.plot(f_o, db_o, color="#cccccc", label="Original", lw=1)
ax2.plot(f_d, db_d, color="#ff4b4b", label="Downsampled", lw=1.2, alpha=0.9)
ax2.set_xlim(0, fs/2) # Show up to original Nyquist to see the cutoff
ax2.set_ylim(-80, 5)
ax2.set_title("Frequency Spectrum")
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("dB")
ax2.legend(frameon=False, loc="upper right")
style_plot(fig2, ax2)

# --- Layout: Plots & Audio ---
col_plots, col_audio = st.columns([2, 1])

with col_plots:
    c_p1, c_p2 = st.columns(2)
    with c_p1: st.pyplot(fig1, use_container_width=True)
    with c_p2: st.pyplot(fig2, use_container_width=True)

with col_audio:
    st.markdown("#### 🎧 Listen")
    st.caption("Original (16 kHz)")
    st.audio((sig * 0.9).astype(np.float32), sample_rate=int(fs))
    
    st.divider()
    
    st.caption(f"Downsampled ({fs_ds} Hz)")
    st.audio((sig_ds * 0.9).astype(np.float32), sample_rate=int(fs_ds))

# --- Interpretation ---
with st.expander("📚 Understanding Aliasing", expanded=True):
    
    st.markdown(f"""
    **Current Settings:**
    * Original Sampling Rate: **{fs} Hz** (Nyquist Limit: {fs/2:.0f} Hz)
    * New Sampling Rate: **{fs_ds} Hz** (Nyquist Limit: {fs_ds/2:.0f} Hz)

    **What is happening?**
    1.  **With Filter OFF:** The square wave has harmonics extending well beyond {fs_ds/2:.0f} Hz. When you downsample, these high frequencies have nowhere to go, so they "fold back" (alias) into the lower frequencies. This creates distinct, metallic-sounding distortion.
    2.  **With Filter ON:** We remove the frequencies above {fs_ds/2:.0f} Hz *before* throwing away samples. The signal looks "blurrier" in the time domain (rounded edges), but it is mathematically correct and sounds cleaner (no metallic artifacts).
    """)
