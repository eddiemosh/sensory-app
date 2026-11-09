# SensoryAware

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

## Running the tests

```bash
pytest
```

## Workflow

- Commit small and often; one GitHub issue per milestone.
- Work on feature branches and merge into `main` through pull requests.
- Requirements and user stories are committed before the interface and decision
  engine code, so the design visibly precedes the build.

## Licence note

ESC-50 is believed to be non-commercial. Fine for coursework; check before any
commercial use is discussed in the business analysis.
