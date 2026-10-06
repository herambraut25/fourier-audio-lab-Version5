import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wave
import io


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Fourier Audio Lab",
    page_icon="🎵",
    layout="wide"
)


# ==========================================================
# CUSTOM STYLE
# ==========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    margin-bottom: 25px;
}

.hero h1 {
    margin-bottom: 5px;
}

.info-card {
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #30363d;
    background: #161b22;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="hero">

# 🎵 Fourier Audio Lab

### Interactive Fourier Analysis • Equalization • Noise Filtering

See how an audio signal is transformed into frequency
components, filtered mathematically, and reconstructed.

</div>
""", unsafe_allow_html=True)


# ==========================================================
# WAV READER
# ==========================================================

def read_wav(file):

    with wave.open(file, "rb") as wav:

        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()

        raw_data = wav.readframes(frames)

    if sample_width == 1:

        audio = np.frombuffer(
            raw_data,
            dtype=np.uint8
        ).astype(np.float32)

        audio -= 128

    elif sample_width == 2:

        audio = np.frombuffer(
            raw_data,
            dtype=np.int16
        ).astype(np.float32)

    elif sample_width == 4:

        audio = np.frombuffer(
            raw_data,
            dtype=np.int32
        ).astype(np.float32)

    else:

        return None, None

    # Stereo → Mono

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels
        )

        audio = np.mean(
            audio,
            axis=1
        )

    # Normalize

    maximum = np.max(
        np.abs(audio)
    )

    if maximum > 0:

        audio = audio / maximum

    return audio, sample_rate


# ==========================================================
# CREATE WAV
# ==========================================================

def create_wav(audio, sample_rate):

    audio = np.clip(
        audio,
        -1,
        1
    )

    audio_int = (
        audio * 32767
    ).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        wav.writeframes(
            audio_int.tobytes()
        )

    return buffer.getvalue()


# ==========================================================
# DEMO AUDIO
# ==========================================================

def create_demo_audio():

    sample_rate = 44100
    duration = 5

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    # Bass
    bass = (
        0.50 *
        np.sin(
            2 * np.pi * 120 * t
        )
    )

    # Mid / vocal-like component
    mid = (
        0.30 *
        np.sin(
            2 * np.pi * 440 * t
        )
    )

    # High-frequency components
    high1 = (
        0.15 *
        np.sin(
            2 * np.pi * 3500 * t
        )
    )

    high2 = (
        0.10 *
        np.sin(
            2 * np.pi * 6500 * t
        )
    )

    # Noise
    noise = (
        0.08 *
        np.random.randn(
            len(t)
        )
    )

    audio = (
        bass
        + mid
        + high1
        + high2
        + noise
    )

    audio /= np.max(
        np.abs(audio)
    )

    return audio, sample_rate


# ==========================================================
# FFT PROCESSING
# ==========================================================

@st.cache_data
def analyze_audio(audio, sample_rate):

    fft_result = np.fft.rfft(
        audio
    )

    frequencies = np.fft.rfftfreq(
        len(audio),
        1 / sample_rate
    )

    magnitude = np.abs(
        fft_result
    )

    time = (
        np.arange(
            len(audio)
        )
        / sample_rate
    )

    return (
        fft_result,
        frequencies,
        magnitude,
        time
    )


# ==========================================================
# INPUT SECTION
# ==========================================================

st.header("🎙️ 1. Input Audio")

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)

use_demo = st.button(
    "🎵 Generate Demo Audio",
    use_container_width=True
)


# ==========================================================
# LOAD AUDIO
# ==========================================================

audio = None
sample_rate = None

if uploaded_file is not None:

    audio, sample_rate = read_wav(
        uploaded_file
    )

elif use_demo:

    audio, sample_rate = create_demo_audio()


# ==========================================================
# MAIN APP
# ==========================================================

if audio is not None:

    st.success("Audio loaded successfully! 🎧")


    # ======================================================
    # ORIGINAL AUDIO
    # ======================================================

    st.header("▶️ 2. Original Audio")

    original_wav = create_wav(
        audio,
        sample_rate
    )

    st.audio(
        original_wav,
        format="audio/wav"
    )


    # ======================================================
    # ANALYZE
    # ======================================================

    (
        fft_result,
        frequencies,
        magnitude,
        time
    ) = analyze_audio(
        audio,
        sample_rate
    )


    # ======================================================
    # WAVEFORM
    # ======================================================

    st.header("🌊 3. Time-Domain Waveform")

    fig, ax = plt.subplots(
        figsize=(14, 4)
    )

    ax.plot(
        time,
        audio
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.set_title(
        "Original Audio Signal"
    )

    ax.grid(
        alpha=0.3
    )

    st.pyplot(
        fig,
        clear_figure=True
    )


    # ======================================================
    # FOURIER ANALYSIS
    # ======================================================

    st.header("🔬 4. Fourier Analysis")

    st.latex(
        r"""
        X[k] =
        \sum_{n=0}^{N-1}
        x[n]e^{-j2\pi kn/N}
        """
    )

    st.write(
        "The Fast Fourier Transform converts the "
        "audio from the time domain into its frequency domain."
    )


    # ======================================================
    # ORIGINAL SPECTRUM
    # ======================================================

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        frequencies,
        magnitude
    )

    ax.set_xlim(
        0,
        min(10000, sample_rate / 2)
    )

    ax.set_xlabel(
        "Frequency (Hz)"
    )

    ax.set_ylabel(
        "Magnitude"
    )

    ax.set_title(
        "Original Fourier Spectrum"
    )

    ax.grid(
        alpha=0.3
    )

    st.pyplot(
        fig,
        clear_figure=True
    )


    # ======================================================
    # EQUALIZER
    # ======================================================

    st.header("🎚️ 5. Fourier Equalizer")

    st.write(
        "Change the frequency weights and observe "
        "how the Fourier spectrum and reconstructed "
        "audio change."
    )


    # ======================================================
    # INTERACTIVE FRAGMENT
    # ======================================================

    @st.fragment
    def equalizer():

        # ----------------------------------------------
        # CONTROLS
        # ----------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            low_gain = st.slider(
                "🔵 Low",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                key="low_gain"
            )

        with col2:

            mid_gain = st.slider(
                "🟢 Mid",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                key="mid_gain"
            )

        with col3:

            high_gain = st.slider(
                "🟠 High",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="high_gain"
            )

        with col4:

            noise_gain = st.slider(
                "🔴 Noise",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key="noise_gain"
            )


        # ----------------------------------------------
        # FREQUENCY WEIGHTS
        # ----------------------------------------------

        weights = np.ones(
            len(frequencies)
        )


        low_region = (
            frequencies < 250
        )

        mid_region = (
            (frequencies >= 250)
            &
            (frequencies < 2000)
        )

        high_region = (
            (frequencies >= 2000)
            &
            (frequencies < 5000)
        )

        noise_region = (
            frequencies >= 5000
        )


        weights[
            low_region
        ] = low_gain

        weights[
            mid_region
        ] = mid_gain

        weights[
            high_region
        ] = high_gain

        weights[
            noise_region
        ] = noise_gain


        # ----------------------------------------------
        # FILTER
        # ----------------------------------------------

        filtered_fft = (
            fft_result * weights
        )


        # ----------------------------------------------
        # INVERSE FOURIER
        # ----------------------------------------------

        filtered_audio = np.fft.irfft(
            filtered_fft,
            n=len(audio)
        )


        # Normalize

        maximum = np.max(
            np.abs(
                filtered_audio
            )
        )

        if maximum > 0:

            filtered_audio = (
                filtered_audio / maximum
            )


        # ----------------------------------------------
        # WEIGHTING FUNCTION
        # ----------------------------------------------

        st.subheader(
            "🧮 Frequency Weighting Function"
        )

        st.latex(
            r"""
            X_{\mathrm{filtered}}[k]
            =
            W[k]X[k]
            """
        )

        fig, ax = plt.subplots(
            figsize=(14, 3.5)
        )

        ax.plot(
            frequencies,
            weights
        )

        ax.set_xlim(
            0,
            min(10000, sample_rate / 2)
        )

        ax.set_ylim(
            -0.05,
            1.05
        )

        ax.set_xlabel(
            "Frequency (Hz)"
        )

        ax.set_ylabel(
            "W[k]"
        )

        ax.set_title(
            "Fourier Frequency Weights"
        )

        ax.grid(
            alpha=0.3
        )

        st.pyplot(
            fig,
            clear_figure=True
        )


        # ----------------------------------------------
        # BEFORE / AFTER
        # ----------------------------------------------

        st.subheader(
            "📊 Before vs After"
        )

        filtered_magnitude = np.abs(
            filtered_fft
        )

        fig, ax = plt.subplots(
            figsize=(14, 4.5)
        )

        ax.plot(
            frequencies,
            magnitude,
            label="Original"
        )

        ax.plot(
            frequencies,
            filtered_magnitude,
            label="Filtered"
        )

        ax.set_xlim(
            0,
            min(10000, sample_rate / 2)
        )

        ax.set_xlabel(
            "Frequency (Hz)"
        )

        ax.set_ylabel(
            "Magnitude"
        )

        ax.set_title(
            "Fourier Spectrum Comparison"
        )

        ax.legend()

        ax.grid(
            alpha=0.3
        )

        st.pyplot(
            fig,
            clear_figure=True
        )


        # ----------------------------------------------
        # AUDIO OUTPUT
        # ----------------------------------------------

        st.subheader(
            "✨ Processed Audio"
        )

        filtered_wav = create_wav(
            filtered_audio,
            sample_rate
        )

        st.audio(
            filtered_wav,
            format="audio/wav"
        )


        # ----------------------------------------------
        # NUMERICAL INFORMATION
        # ----------------------------------------------

        original_energy = np.mean(
            audio ** 2
        )

        filtered_energy = np.mean(
            filtered_audio ** 2
        )

        st.subheader(
            "📈 Live Filter Values"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Low",
                f"{low_gain:.2f}"
            )

        with c2:

            st.metric(
                "Mid",
                f"{mid_gain:.2f}"
            )

        with c3:

            st.metric(
                "High",
                f"{high_gain:.2f}"
            )

        with c4:

            st.metric(
                "Noise",
                f"{noise_gain:.2f}"
            )


        st.caption(
            f"Original energy: {original_energy:.4f}  |  "
            f"Filtered energy: {filtered_energy:.4f}"
        )


        return filtered_audio


    # Run equalizer

    filtered_audio = equalizer()


    # ======================================================
    # MATHEMATICAL EXPLANATION
    # ======================================================

    st.header("📚 6. Mathematical Background")

    st.markdown(
        """
### Step 1 — Fourier Decomposition

The audio signal is represented using sinusoidal
frequency components.

### Step 2 — Frequency Weighting

Each frequency component receives a weight `W[k]`
between 0 and 1.

### Step 3 — Filtering

Unwanted frequency components are attenuated.

### Step 4 — Reconstruction

The inverse Fourier transform converts the
modified frequency representation back into audio.
"""
    )

    st.latex(
        r"""
        f(x)=
        \frac{a_0}{2}
        +
        \sum_{n=1}^{\infty}
        \left[
        a_n\cos(nx)
        +
        b_n\sin(nx)
        \right]
        """
    )

    st.latex(
        r"""
        X_{\mathrm{filtered}}[k]
        =
        W[k]X[k]
        """
    )


    # ======================================================
    # PROJECT CONNECTION
    # ======================================================

    st.header("🎓 Project Concept")

    st.info(
        "The Fourier Series provides the mathematical "
        "foundation for representing a signal as a "
        "combination of sinusoidal components. "
        "For digital audio, the FFT provides an efficient "
        "way to analyze those frequency components."
    )


else:

    st.info(
        "👆 Upload a WAV file or click "
        "**Generate Demo Audio** to begin."
    )