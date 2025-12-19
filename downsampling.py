import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, square

# --- Page Config & CSS ---
st.set_page_config(
    page_title="Downsampling & Aliasing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("### 📉 Downsampling & Aliasing")
st.markdown(
    "When reducing the sampling rate (downsampling), we must remove high frequencies first. "
    "If we don't, they **alias** (fold back) and distort the signal."
)

# --- Controls (Compact) ---
with st.container(border=True):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        factor = st.slider("Downsampling Factor (M)", 1, 16, 4, step=1)
    with c2:
        st.write("") # Spacer to align checkbox
        st.write("") 
        apply_filter = st.toggle("Apply Anti-aliasing Filter", value=True)
    with c3:
        st.info("💡 **Tip:** Listen to the audio. Without the filter, the pitch changes wrongly!", icon="👂")

# --- Processing ---
fs = 16000          # High enough to hear clear artifacts
f0 = 150            # Fundamental freq
duty = 60           # Duty cycle
duration = 0.5      

# 1. Generate Signal
N = int(np.round(duration * fs))
t = np.arange(N) / fs
# Square wave (rich in harmonics)
sig = square(2*np.pi*f0*t, duty=duty/100.0).astype(np.float64)

# 2. Filter & Downsample Logic
if factor < 1: factor = 1
fs_ds = fs // factor

if apply_filter and factor > 1:
    # Cutoff just below the NEW Nyquist frequency (fs_ds / 2)
    ny_new = 0.5 * fs_ds
    cutoff = 0.9 * ny_new 
    b, a = butter(8, cutoff/(0.5*fs), btype="low") # 8th order butterworth
    sig_proc = filtfilt(b, a, sig)
else:
    sig_proc = sig

# Decimate (Downsample)
sig_ds = sig_proc[::factor]

# --- FFT Helper ---
def get_spectrum(x, fs_local):
    n = len(x)
    win = np.hanning(n)
    # FFT
    mag = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, 1/fs_local)
    # Convert to dB
    mag_db = 20 * np.log10(mag / np.max(mag) + 1e-12)
    return freqs, mag_db

# --- Plotting Setup ---
plt.rcParams.update({'font.size': 8})
col_plot1, col_plot2 = st.columns(2)

# Helper for transparent plots
def style_ax(ax):
    ax.grid(True, alpha=0.2, ls="--")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Transparent background
    ax.set_facecolor("none")

# PLOT 1: Time Domain (Zoomed)
with col_plot1:
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    fig1.patch.set_alpha(0) # Transparent figure
    
    # Zoom window (20ms)
    zoom_sec = 0.02
    zoom_samples = int(zoom_sec * fs)
    
    # Plot Original
    ax1.plot(t[:zoom_samples], sig[:zoom_samples], 
             label="Original", color="#dddddd", lw=1.5, alpha=0.6)
    
    # Plot Filtered (if applicable)
    if apply_filter and factor > 1:
         ax1.plot(t[:zoom_samples], sig_proc[:zoom_samples], 
                  label="Filtered (Pre)", color="#00a8cc", lw=1.5, alpha=0.8)
    
    # Plot Downsampled (Stem-like)
    # We must scale the index for the downsampled array
    zoom_samples_ds = int(zoom_sec * fs_ds)
    t_ds = np.arange(zoom_samples_ds) / fs_ds
    ax1.plot(t_ds, sig_ds[:zoom_samples_ds], 
             'o', label=f"Downsampled (1/{factor})", color="#ff4b4b", markersize=4)
    
    style_ax(ax1)
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_title("Time Domain (20ms zoom)", loc="left", color="gray")
    ax1.set_xlabel("Time [s]")
    st.pyplot(fig1, use_container_width=True)

# PLOT 2: Frequency Domain
with col_plot2:
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    fig2.patch.set_alpha(0)

    # Get Spectra
    f_orig, db_orig = get_spectrum(sig[:4096], fs)
    f_ds, db_ds = get_spectrum(sig_ds[:4096], fs_ds)

    ax2.plot(f_orig, db_orig, label="Original", color="#dddddd", lw=1, alpha=0.6)
    ax2.plot(f_ds, db_ds, label="Downsampled", color="#ff4b4b", lw=1.2)
    
    # Mark the New Nyquist Limit
    ax2.axvline(fs_ds/2, color="#00a8cc", ls="--", lw=1)
    ax2.text(fs_ds/2, 0, " New Nyquist", color="#00a8cc", fontsize=8, rotation=90, va='top')

    style_ax(ax2)
    ax2.set_xlim(0, 4000) # Limit view to relevant frequencies
    ax2.set_ylim(-60, 5)
    ax2.legend(loc="upper right", frameon=False)
    ax2.set_title("Frequency Domain", loc="left", color="gray")
    ax2.set_xlabel("Frequency [Hz]")
    st.pyplot(fig2, use_container_width=True)

# --- Audio Section ---
st.divider()
ac1, ac2, ac3 = st.columns([1, 1, 2])
with ac1:
    st.markdown("**Original (16kHz)**")
    st.audio(sig, sample_rate=fs)
with ac2:
    st.markdown(f"**Downsampled ({fs_ds}Hz)**")
    st.audio(sig_ds, sample_rate=fs_ds)
with ac3:
    # Interpretation
    st.markdown(f"""
    <div style="font-size: 0.9em; color: gray; margin-top: -10px;">
    <b>Observation:</b><br>
    The square wave has harmonics at {f0*3}, {f0*5}, {f0*7} Hz...<br>
    The new Sampling Rate is <b>{fs_ds} Hz</b> (Nyquist = {fs_ds/2} Hz).<br>
    With filter <b>OFF</b>, harmonics above {fs_ds/2} Hz fold back (alias) and sound like "metallic" noise.
    </div>
    """, unsafe_allow_html=True)

# --- Educational Expander ---
with st.expander("📚 Understanding the Math"):
    st.markdown("""
    **What is happening?**
    
    When we downsample by a factor of $M$ (decimation), the new sampling rate becomes $f_s / M$. 
    Any frequency component in the original signal that is above the new Nyquist limit ($f_s / 2M$) cannot be represented correctly.
    """)
    
    st.markdown("""
    Instead of disappearing, these high frequencies "reflect" around the Nyquist frequency, appearing as lower frequencies (ghosts) in the new signal. This is **Aliasing**.
    """)
    
    # 
    
    st.markdown("""
    **The Solution:**
    We apply a **Low-Pass Filter** (Anti-aliasing filter) to remove frequencies above $f_s / 2M$ *before* we throw away the samples. This ensures the remaining signal fits within the new bandwidth limits.
    """)
