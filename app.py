import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Music Genre Clustering", page_icon="🎵")

MODEL_PATH = Path(__file__).parent / "music_cluster_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

SEGMENTS = {
    0: ("Danceable & Vocal-Heavy", "Very danceable, happy, and rich in vocals. Mostly rap, latin, and r&b."),
    1: ("Instrumental / Electronic", "Almost no vocals, high energy. Mostly EDM."),
    2: ("Loud & Energetic", "The loudest, most energetic, and fastest songs. Mostly rock and EDM."),
    3: ("Calm & Acoustic", "Quiet, low energy, and more acoustic. Mostly r&b, pop, and rock."),
}

# danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
PRESETS = {
    "Enter manually": None,
    "Example: Danceable & Vocal-Heavy": (0.75, 0.71, -6.5, 0.14, 0.15, 0.01, 0.16, 0.65, 114.0),
    "Example: Instrumental / Electronic": (0.66, 0.78, -7.0, 0.07, 0.08, 0.75, 0.19, 0.39, 125.0),
    "Example: Loud & Energetic": (0.55, 0.80, -5.3, 0.09, 0.07, 0.02, 0.24, 0.42, 133.0),
    "Example: Calm & Acoustic": (0.61, 0.43, -10.5, 0.09, 0.51, 0.09, 0.16, 0.40, 113.0),
}

st.title("Music Genre Clustering")
st.write("Enter a song's audio features and the model will assign it to a sound-based cluster.")

choice = st.selectbox("Load an example (optional)", list(PRESETS.keys()))
d = PRESETS[choice] if PRESETS[choice] else (0.65, 0.70, -6.8, 0.10, 0.18, 0.01, 0.19, 0.51, 121.0)

col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, float(d[0]), 0.01)
    energy = st.slider("Energy", 0.0, 1.0, float(d[1]), 0.01)
    loudness = st.slider("Loudness (dB)", -30.0, 2.0, float(d[2]), 0.1)
    speechiness = st.slider("Speechiness", 0.0, 1.0, float(d[3]), 0.01)
    acousticness = st.slider("Acousticness", 0.0, 1.0, float(d[4]), 0.01)

with col2:
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, float(d[5]), 0.01)
    liveness = st.slider("Liveness", 0.0, 1.0, float(d[6]), 0.01)
    valence = st.slider("Valence (positivity)", 0.0, 1.0, float(d[7]), 0.01)
    tempo = st.slider("Tempo (BPM)", 40.0, 240.0, float(d[8]), 1.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
    }])

    cluster = int(model.predict(input_df)[0])
    name, description = SEGMENTS[cluster]
    st.success(f"Cluster: **{name}** (cluster {cluster})")
    st.write(description)
