"""SensoryAware demonstration. Run with: streamlit run app.py"""

import dataclasses
import io
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sensoryaware.audio import AudioLoadError, load_audio  # noqa: E402
from sensoryaware.config import DEFAULT_CONFIG  # noqa: E402
from sensoryaware.decision import decide  # noqa: E402
from sensoryaware.model import YamnetClassifier  # noqa: E402


@st.cache_resource(show_spinner="Loading YAMNet...")
def get_classifier() -> YamnetClassifier:
    return YamnetClassifier()


def build_config():
    st.sidebar.header("Your settings")
    by_name = {category.name: category for category in DEFAULT_CONFIG.categories}
    selected = st.sidebar.multiselect(
        "Sounds that distract you", list(by_name), default=list(by_name)
    )
    min_score = st.sidebar.slider(
        "Score threshold", 0.05, 1.0, DEFAULT_CONFIG.min_score, 0.05
    )
    min_frames = st.sidebar.slider(
        "Minimum frames detected",
        1,
        20,
        DEFAULT_CONFIG.min_frames,
        help="YAMNet scores about two frames per second, and one short sound "
        "usually appears in two or three of them.",
    )
    return dataclasses.replace(
        DEFAULT_CONFIG,
        categories=tuple(by_name[name] for name in selected),
        min_score=min_score,
        min_frames=min_frames,
    )


def show_decision(decision):
    st.subheader("Recommendation")
    if decision.intervention:
        st.success(f"**{decision.intervention.name}**: {decision.intervention.description}")
    else:
        st.info("No intervention recommended.")
    st.write(decision.explanation)
    for warning in decision.warnings:
        st.warning(warning)
    if decision.evidence:
        st.table(
            [
                {
                    "Category": item.category,
                    "Frames detected": f"{item.frames_detected} of {item.total_frames}",
                    "Peak score": f"{item.peak_score:.3f}",
                    "Peak label": item.peak_label or "-",
                    "Met the rule": "Yes" if item.triggered else "No",
                }
                for item in decision.evidence
            ]
        )


st.set_page_config(page_title="Earitation", page_icon="👂", layout="centered")
st.title("👂 Earitation")
st.write(
    "Spot the sounds that irritate your ears in a recording, and get a suggestion "
    "for tuning them out."
)
st.caption(
    "Proof of concept. Suggestions are not medical advice, and a detected sound "
    "does not show that anyone is distracted or distressed. Sounds are only "
    "assessed within the clip."
)
st.info(
    "**How your audio is handled.** Your recording or file is sent to the server "
    "running this app, which is a third-party cloud service if you opened a hosted "
    "link. It is analysed there and is not saved by this app or sent to any other "
    "service."
)

config = build_config()
if not st.checkbox(
    "I understand how my audio is handled, and I will not record or upload other "
    "people without their consent."
):
    st.stop()

source = st.radio(
    "Audio source", ["Upload a file", "Record with microphone"], horizontal=True
)
if source == "Upload a file":
    uploaded = st.file_uploader(
        "Choose an audio file", type=["wav", "flac", "ogg", "mp3"]
    )
else:
    st.caption(
        "Record about 10 seconds; a sound has to recur for several seconds to meet "
        "the default rule. Avoid capturing other people without their consent."
    )
    uploaded = st.audio_input("Record a clip")
if uploaded is None:
    st.stop()

audio_bytes = uploaded.getvalue()
st.audio(audio_bytes)

try:
    waveform = load_audio(io.BytesIO(audio_bytes))
except AudioLoadError as error:
    st.error(str(error))
    st.stop()

try:
    classifier = get_classifier()
except Exception as error:
    st.error(
        "YAMNet could not be loaded. It needs an internet connection the first "
        f"time it runs. Details: {error}"
    )
    st.stop()

with st.spinner("Analysing audio..."):
    predictions = classifier.classify(waveform)

st.subheader("Top sounds detected")
st.table(
    [
        {"Sound": item.label, "Mean score": f"{item.score:.3f}"}
        for item in predictions.top_k(5)
    ]
)
st.caption("Scores are model confidence values, not probabilities.")

show_decision(decide(predictions, config))
