# Trained frame detector

`frame_detector.npz` is the optional detector the app uses in "Trained
detector" mode (FR13, decision D3).

| | |
| --- | --- |
| Input | YAMNet's 1,024-value embedding for one 0.96 s frame |
| Output | Probability for clicking, typing, banging, alarms and "other" |
| Model | Standardisation + multinomial logistic regression (C = 0.1, balanced class weights) |
| Training data | ESC-50, all 2,000 clips: mouse_click and clock_tick as clicking, keyboard_typing as typing, door_wood_knock as banging, clock_alarm as alarms, the other 45 classes as "other". Quiet frames of target clips (over 20 dB below the clip's loudest) are left out |
| Use in the app | A frame counts at probability 0.5 or more; a category triggers in at least 4 frames |
| Cross-validated result | 80% of target clips detected, 3.0% false alarms on non-target clips (`docs/evaluation.md`) |
| Known limits | Not tested on talking or music; knocking detected in only 45% of clips; no chewing or speech |
| Licence | Derived from ESC-50 (Piczak, 2015), licensed CC BY-NC 3.0: non-commercial use only |

Rebuild with `python scripts/evaluate_app_task.py` (extracts and caches the
frame features) and then `python scripts/train_frame_detector.py`.
