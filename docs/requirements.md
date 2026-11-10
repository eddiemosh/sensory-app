# SensoryAware requirements (first draft)

Derived from the project plan's objectives, scope and testing strategy. Review
and revise before relying on this in the report. SC numbers refer to the plan's
success criteria.

## User stories

| ID | Story |
| --- | --- |
| US1 | As someone who finds certain sounds distracting, I want to analyse a recording of my environment so that I can see which sounds it contains. |
| US2 | As a user, I want to choose which sound categories count as distracting for me, because this varies between people. |
| US3 | As a user, I want a one-off sound to be ignored so that I am not interrupted unnecessarily. |
| US4 | As a user, I want a suitable audio recommendation when a distracting sound recurs. |
| US5 | As a user, I want to know why a recommendation was or was not made so that I can trust and tune it. |

## Functional requirements

| ID | Requirement | Source |
| --- | --- | --- |
| FR1 | The system shall load a user-selected audio file, or a single clip the user records in the interface, and convert it to 16 kHz mono. | US1, SC1 |
| FR2 | The system shall report a useful error for corrupt, unsupported, empty or over-long audio. | SC1 |
| FR3 | The system shall run YAMNet and produce per-frame scores for its 521 classes. | US1, SC2 |
| FR4 | The system shall display the five highest-scoring classes with their scores. | US1, SC3 |
| FR5 | The user shall be able to configure the distracting categories, score threshold and minimum frame count. | US2, SC4 |
| FR6 | A category shall trigger only when its score reaches the threshold in at least the minimum number of frames. | US3, SC4 |
| FR7 | Each triggered category shall lead to its configured intervention. | US4, SC5 |
| FR8 | When two or more categories trigger, the system shall recommend the general focus profile. | US4, SC5 |
| FR9 | When no category triggers, the system shall recommend no intervention. | US3, SC5 |
| FR10 | Every decision shall include a plain-language explanation and per-category evidence. | US5, SC5 |
| FR11 | Configured labels that YAMNet does not have shall be reported and ignored, not cause a failure. | SC4 |

## Non-functional requirements

| ID | Requirement |
| --- | --- |
| NFR1 | Audio shall be processed only by the server running the application, and shall not be written to disk or sent to any other service by the application. |
| NFR2 | The decision engine shall be independent of the interface and the ML framework, so it can be unit tested without either. |
| NFR3 | The same model and input shall produce the same result. |
| NFR4 | The interface shall present interventions as recommendations, and shall not claim to diagnose, treat or measure distraction or distress. |
| NFR5 | Recurrence shall be described as recurrence within the analysed clip, not over real-world time. |
| NFR6 | The interface shall explain how audio is handled and obtain the user's agreement before accepting audio. |
