import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, square

st.title("⬇️ Downsampling")

# Intro
st.markdown('''Why do we need to low-pass a discrete signal before downsampling it?
            Try it by yourself.
            ''')
# -----------------------
# Fixed signal parameters (as requested)
# -----------------------
fs = 16000          # sampling frequency [Hz]
f0 = 150            # square-wave fundamental [Hz]
duty = 60           # duty cycle [%]
duration = 0.5      # seconds

# -----------------------
# Minimal controls (inline)
# -----------------------
c1, c2 = st.columns(2)
with c1:
    factor = st.slider("Downsampling factor ×", 1, 16, 4, step=1)
with c2:
    apply_filter = st.checkbox("Anti-aliasing low-pass (pre-decimate)", value=True)

# -----------------------
# Generate rectangular (square) wave
# -----------------------
N = int(np.round(duration * fs))
t = np.arange(N) / fs
sig = square(2*np.pi*f0*t, duty=duty/100.0).astype(np.float64)
sig = sig / np.max(np.abs(sig))  # normalize

# -----------------------
# Anti-aliasing filter (optional) and downsampling
# -----------------------
def butter_lowpass(cutoff_hz, fs, order=8):
    return butter(order, cutoff_hz/(0.5*fs), btype="low", output="ba")

if factor < 1:
    factor = 1
fs_ds = fs // factor if factor > 0 else fs

if apply_filter and factor > 1:
    ny_new = 0.5 * fs_ds
    cutoff = 0.9 * ny_new  # just below new Nyquist
    b, a = butter_lowpass(cutoff, fs, order=8)
    sig_proc = filtfilt(b, a, sig)
else:
    sig_proc = sig

sig_ds = sig_proc[::factor]

# -----------------------
# FFT helper (dB)
# -----------------------
def db_spectrum(x, fs):
    N = len(x)
    if N < 4:
        return np.array([0.0]), np.array([-120.0])
    win = np.hanning(N)
    cg = np.sum(win) / N  # coherent gain
    X = np.fft.rfft(x * win)
    mag = (2.0 / N) * np.abs(X) / max(cg, 1e-12)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    f = np.fft.rfftfreq(N, 1.0/fs)
    return f, mag_db

# -----------------------
# Time-domain plot
# -----------------------
show_time = min(N, int(0.02 * fs))  # ~20 ms for clarity
fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(t[:show_time], sig[:show_time], label="Original", alpha=0.9)
if apply_filter and factor > 1:
    ax1.plot(t[:show_time], sig_proc[:show_time], label="Filtered (pre-downsample)", alpha=0.9)
t_ds = np.arange(len(sig_ds)) / fs_ds
ax1.plot(t_ds[:int(show_time * fs_ds / fs)], sig_ds[:int(show_time * fs_ds / fs)],
         "o-", markersize=2, label=f"Downsampled ×{factor}")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.set_title("Time Domain")
st.pyplot(fig1)

# -----------------------
# Frequency-domain plot (dB)
# -----------------------
Nfft = min(1 << 14, N)
f_o, db_o = db_spectrum(sig[:Nfft], fs)
f_d, db_d = db_spectrum(sig_ds[:min(len(sig_ds), Nfft)], fs_ds)

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(f_o, db_o, label="Original (dB)", alpha=0.9)
ax2.plot(f_d, db_d, label=f"Downsampled ×{factor} (dB)", alpha=0.9)
ax2.set_xlim(0, max(f_o[-1], f_d[-1]) if len(f_o) and len(f_d) else fs/2)
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Magnitude [dB]")
ax2.grid(True, which="both", linestyle="--", alpha=0.3)
ax2.legend()
ax2.set_title("Frequency Domain")
st.pyplot(fig2)

# -----------------------
# Listen
# -----------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown('''Original signal''')
    st.audio((sig * 0.9).astype(np.float32), sample_rate=int(fs), format="audio/wav")
with c2:
    st.markdown('''Downsampled signal''')
    st.audio((sig_ds * 0.9).astype(np.float32), sample_rate=int(fs_ds), format="audio/wav")


# -----------------------
# Notes
# -----------------------
ny_orig = fs/2
ny_new = fs_ds/2
with st.expander("Open for comments"):
    st.markdown(
        f"""
    **Sampling**
    - Original fs: **{fs} Hz**  (Nyquist: **{ny_orig:.0f} Hz**)
    - Downsampled fs: **{fs_ds} Hz** (Nyquist: **{ny_new:.0f} Hz**)
    
    **What to observe** 
    - A rectangular wave contains **many odd harmonics**: {f0} Hz, 3·{f0}, 5·{f0}, …
    - With the filter **OFF**, harmonics above the new Nyquist (**{ny_new:.0f} Hz**) will **alias** (fold back) into lower frequencies.
    - With the filter **ON** (an 8th order Butterworth digital filter is used here), those high harmonics are attenuated before decimation, reducing aliasing in the downsampled FFT.
    """
    )



# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt, square

# st.title("⬇️ Downsampling")

# # Intro
# st.markdown('''Why do we need to low-pass a discrete signal before downsampling it?
#             Try it by yourself.
#             ''')
# # -----------------------
# # Fixed signal parameters (as requested)
# # -----------------------
# fs = 8000          # sampling frequency [Hz]
# f0 = 150            # square-wave fundamental [Hz]
# duty = 60           # duty cycle [%]
# duration = 0.5      # seconds

# # -----------------------
# # Minimal controls (inline)
# # -----------------------
# c1, c2 = st.columns(2)
# with c1:
#     factor = st.slider("Downsampling factor ×", 1, 16, 2, step=1)
# with c2:
#     apply_filter = st.checkbox("Anti-aliasing low-pass (pre-decimate)", value=True)

# # -----------------------
# # Generate rectangular (square) wave
# # -----------------------
# N = int(np.round(duration * fs))
# t = np.arange(N) / fs
# sig = square(2*np.pi*f0*t, duty=duty/100.0).astype(np.float64)
# sig = sig / np.max(np.abs(sig))  # normalize

# # -----------------------
# # Anti-aliasing filter (optional) and downsampling
# # -----------------------
# def butter_lowpass(cutoff_hz, fs, order=8):
#     return butter(order, cutoff_hz/(0.5*fs), btype="low", output="ba")

# if factor < 1:
#     factor = 1
# fs_ds = fs // factor if factor > 0 else fs

# if apply_filter and factor > 1:
#     ny_new = 0.5 * fs_ds
#     cutoff = 0.9 * ny_new  # just below new Nyquist
#     b, a = butter_lowpass(cutoff, fs, order=8)
#     sig_proc = filtfilt(b, a, sig)
# else:
#     sig_proc = sig

# sig_ds = sig_proc[::factor]

# # -----------------------
# # FFT helper (dB)
# # -----------------------
# def db_spectrum(x, fs):
#     N = len(x)
#     if N < 4:
#         return np.array([0.0]), np.array([-120.0])
#     win = np.hanning(N)
#     cg = np.sum(win) / N  # coherent gain
#     X = np.fft.rfft(x * win)
#     mag = (2.0 / N) * np.abs(X) / max(cg, 1e-12)
#     mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
#     f = np.fft.rfftfreq(N, 1.0/fs)
#     return f, mag_db

# # -----------------------
# # Time-domain plot
# # -----------------------
# show_time = min(N, int(0.02 * fs))  # ~20 ms for clarity
# fig1, ax1 = plt.subplots(figsize=(10,4))
# ax1.plot(t[:show_time], sig[:show_time], "o-", markersize=2, label="Original", alpha=0.9)
# if apply_filter and factor > 1:
#     ax1.plot(t[:show_time], sig_proc[:show_time], label="Filtered (pre-downsample)", alpha=0.9)
# t_ds = np.arange(len(sig_ds)) / fs_ds
# ax1.plot(t_ds[:int(show_time * fs_ds / fs)], sig_ds[:int(show_time * fs_ds / fs)],
#          "o-", markersize=2, label=f"Downsampled ×{factor}")
# ax1.set_xlabel("Time [s]")
# ax1.set_ylabel("Amplitude")
# ax1.legend()
# ax1.set_title("Time Domain")
# st.pyplot(fig1)

# # -----------------------
# # Frequency-domain plot (dB)
# # -----------------------
# Nfft = min(1 << 14, N)
# f_o, db_o = db_spectrum(sig[:Nfft], fs)
# f_d, db_d = db_spectrum(sig_ds[:min(len(sig_ds), Nfft)], fs_ds)

# fig2, ax2 = plt.subplots(figsize=(10,4))
# ax2.plot(f_o, db_o, label="Original (dB)", alpha=0.9)
# ax2.plot(f_d, db_d, label=f"Downsampled ×{factor} (dB)", alpha=0.9)
# ax2.set_xlim(0, max(f_o[-1], f_d[-1]) if len(f_o) and len(f_d) else fs/2)
# ax2.set_xlabel("Frequency [Hz]")
# ax2.set_ylabel("Magnitude [dB]")
# ax2.grid(True, which="both", linestyle="--", alpha=0.3)
# ax2.legend()
# ax2.set_title("Frequency Domain")
# st.pyplot(fig2)

# # -----------------------
# # Listen
# # -----------------------
# c1, c2 = st.columns(2)
# with c1:
#     st.markdown('''Original signal''')
#     st.audio((sig * 0.9).astype(np.float32), sample_rate=int(fs), format="audio/wav")
# with c2:
#     st.markdown('''Downsampled signal''')
#     st.audio((sig_ds * 0.9).astype(np.float32), sample_rate=int(fs_ds), format="audio/wav")


# # -----------------------
# # Notes
# # -----------------------
# ny_orig = fs/2
# ny_new = fs_ds/2
# with st.expander("Open for comments"):
#     st.markdown(
#         f"""
#     **Sampling**
#     - Original fs: **{fs} Hz**  (Nyquist: **{ny_orig:.0f} Hz**)
#     - Downsampled fs: **{fs_ds} Hz** (Nyquist: **{ny_new:.0f} Hz**)
    
#     **What to observe** 
#     - A rectangular wave contains **many harmonics**
#     - With the filter **OFF**, harmonics above the new Nyquist (**{ny_new:.0f} Hz**) will **alias** (fold back) into lower frequencies.
#     - With the filter **ON** (an 8th order Butterworth digital filter is used here), those high harmonics are attenuated before decimation, reducing aliasing in the downsampled FFT.
    
#     Notice that the downsampled signal in the "OFF" approach seems to be closer to the initial rectangle signal. In practice though, it sounds much worse than with the "ON" approach, 
#     and shows aliased frequency components before the new Nyquist frequency. 
#     """

#     )







