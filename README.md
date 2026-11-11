# Earitation

Earitation is the app name for **SensoryAware**, the coursework project. The
Python package and the report keep the name `sensoryaware`.

Coursework repository for **ECM3440 Software Engineering 2, CW1**: an engineering
report and proof of concept for SensoryAware, a sound-recognition system that
classifies environmental audio (using YAMNet, trained on ESC-50) and applies a
rule-based decision engine to it.

Deadline: 3 December 2026, 12:00. Target completion: 27 November 2026.

## Layout

```
src/sensoryaware/   application package (ML wrapper, decision engine, UI later)
tests/              pytest test suite
docs/               requirements, user stories, design artefacts, report drafts
data/               local datasets (ESC-50 etc.) - not committed
.github/workflows/  CI (pytest on every push and pull request)
```

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Running the proof of concept

YAMNet is downloaded the first time it is used.

```bash
streamlit run app.py --server.address localhost
```

The flag keeps the app private to your computer; without it, the committed
config binds to all network addresses so that hosting works.

To print the top five classes for one file:

```bash
python scripts/yamnet_top5.py path/to/audio.wav
```

## Hosting

The app can be hosted on Streamlit Community Cloud with `app.py` as the entry
point and Python 3.11 selected under Advanced settings. Hosting means audio is
processed on the host's servers, as the in-app notice explains. Keep the app
restricted to invited viewers.

## Running the tests

```bash
pytest
```

CI runs `pytest -m "not model"`, which leaves out the tests that need the real
YAMNet model.

## Workflow

- Commit small and often; one GitHub issue per milestone.
- Work on feature branches and merge into `main` through pull requests.
- Requirements and user stories are committed before the interface and decision
  engine code, so the design visibly precedes the build.

## Licence note

ESC-50 is believed to be non-commercial. Fine for coursework; check before any
commercial use is discussed in the business analysis.
