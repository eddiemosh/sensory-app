"""SensoryAware demonstration. Run with: streamlit run app.py"""

import dataclasses
import io
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sensoryaware.audio import AudioLoadError, frame_levels_db, load_audio  # noqa: E402
from sensoryaware.config import DEFAULT_CONFIG  # noqa: E402
from sensoryaware.decision import decide  # noqa: E402
from sensoryaware.detector import (  # noqa: E402
    FrameDetector,
    add_detector_scores,
    use_trained_detector,
)
from sensoryaware.model import YamnetClassifier  # noqa: E402

LABELS = "YAMNet labels"
TRAINED = "Trained detector (experimental)"


@st.cache_resource(show_spinner="Loading YAMNet...")
def get_classifier() -> YamnetClassifier:
    return YamnetClassifier()


@st.cache_resource(show_spinner="Loading the trained detector...")
def get_detector() -> FrameDetector:
    return FrameDetector.load()


def build_config():
    st.sidebar.header("Your settings")
    method = st.sidebar.radio(
        "Detection method",
        [LABELS, TRAINED],
        help="YAMNet labels: named sound labels from Google's pretrained model. "
        "Trained detector: a model trained for this app on the ESC-50 dataset; it "
        "finds clicks far more often but has not been tested on everyday talking "
        "or music. It covers clicking, typing, banging and alarms; chewing and "
        "loud speech always use YAMNet labels. ESC-50 is licensed for "
        "non-commercial use only.",
    )
    base = use_trained_detector(DEFAULT_CONFIG) if method == TRAINED else DEFAULT_CONFIG
    by_name = {category.name: category for category in base.categories}
    selected = st.sidebar.multiselect(
        "Sounds that distract you", list(by_name), default=list(by_name)
    )
    own_thresholds = ", ".join(
        f"{c.name} ({c.min_score:.2f})"
        for c in base.categories
        if c.min_score is not None
    )
    min_score = st.sidebar.slider(
        "Score threshold",
        0.05,
        1.0,
        base.min_score,
        0.05,
        help=f"Applies to every category except those with their own: {own_thresholds}."
        if own_thresholds
        else None,
    )
    min_frames = st.sidebar.slider(
        "Minimum frames detected",
        1,
        20,
        DEFAULT_CONFIG.min_frames,
        help="YAMNet scores about two frames per second, and one short sound "
        "usually appears in two or three of them.",
    )
    for name, category in by_name.items():
        if name in selected and category.min_level_db is not None:
            level = st.sidebar.slider(
                f"{name}: minimum loudness (dBFS)",
                -60,
                0,
                int(category.min_level_db),
                1,
                help="Quieter frames are ignored for this category. 0 is the loudest "
                "a recording can be. The right value depends on your microphone, so "
                "compare a normal and a loud recording to set it.",
            )
            by_name[name] = dataclasses.replace(category, min_level_db=float(level))
    config = dataclasses.replace(
        base,
        categories=tuple(by_name[name] for name in selected),
        min_score=min_score,
        min_frames=min_frames,
    )
    return config, method == TRAINED


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

config, use_detector = build_config()
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
    yamnet, embeddings = classifier.analyse(waveform)
    yamnet = dataclasses.replace(
        yamnet, levels_db=frame_levels_db(waveform, yamnet.num_frames)
    )
    predictions = yamnet
    if use_detector:
        try:
            predictions = add_detector_scores(yamnet, get_detector(), embeddings)
        except Exception as error:
            st.error(f"The trained detector could not be loaded. Details: {error}")
            st.stop()

st.subheader("Top sounds detected")
st.table(
    [
        {"Sound": item.label, "Mean score": f"{item.score:.3f}"}
        for item in yamnet.top_k(5)
    ]
)
st.caption("YAMNet's scores are model confidence values, not probabilities.")

show_decision(decide(predictions, config))
if use_detector:
    st.caption(
        "Clicking, typing, banging and alarms were judged by the trained detector: "
        "a frame counts when its probability is 0.5 or more. It is experimental and "
        "has not been tested on everyday talking or music."
    )
