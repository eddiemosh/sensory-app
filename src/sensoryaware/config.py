"""Default rules. Labels are real YAMNet class names.

The score threshold and frame count are provisional starting points: they have
not been tuned or measured, and should be set from the ESC-50 evaluation.
"""

from sensoryaware.decision import EngineConfig, Intervention, SoundCategory

GENTLE_INSTRUMENTAL = Intervention(
    "gentle_instrumental",
    "Gentle instrumental audio",
    "Soft, low-tempo instrumental music to sit over small repetitive sounds.",
)
STRONGER_MASKING = Intervention(
    "stronger_masking",
    "Stronger masking audio",
    "Steady broadband sound, such as brown noise, to mask sudden loud sounds.",
)
GENERAL_FOCUS = Intervention(
    "general_focus",
    "General focus audio",
    "A general focus playlist for environments with several distracting sounds.",
)

DEFAULT_CONFIG = EngineConfig(
    categories=(
        SoundCategory(
            "clicking",
            "Clicking and ticking",
            ("Clicking", "Tick", "Tick-tock", "Clickety-clack"),
            GENTLE_INSTRUMENTAL.id,
        ),
        SoundCategory(
            "typing",
            "Typing",
            ("Typing", "Computer keyboard", "Typewriter"),
            GENTLE_INSTRUMENTAL.id,
        ),
        SoundCategory(
            "chewing",
            "Chewing",
            ("Chewing, mastication",),
            GENTLE_INSTRUMENTAL.id,
        ),
        SoundCategory(
            "banging",
            "Banging and knocking",
            ("Bang", "Thump, thud", "Knock", "Slam", "Hammer"),
            STRONGER_MASKING.id,
        ),
        SoundCategory(
            "alarms",
            "Alarms and beeps",
            ("Alarm", "Alarm clock", "Beep, bleep", "Buzzer", "Ringtone"),
            STRONGER_MASKING.id,
        ),
    ),
    interventions=(GENTLE_INSTRUMENTAL, STRONGER_MASKING, GENERAL_FOCUS),
    multi_category_intervention=GENERAL_FOCUS.id,
    min_score=0.2,
    min_frames=4,
)
