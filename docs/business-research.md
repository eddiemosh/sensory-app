# Business research notes

Raw material for report section 2 (problem and business analysis). These are
**notes and sources, not the analysis**: the plan requires the analysis and
conclusions in the report to be my own work. Gathered on 2026-11-16 with AI
help (web searches by Claude); every figure below needs checking against its
source before it is cited.

**Reliability key.** **P** = primary source read directly (the publisher,
dataset or official page). **S** = secondary source (news, law firm, blog,
summary), or a primary source seen only through a summary. Prefer P in the
report; follow S back to its primary source first.

---

## 1. Who has the problem (need and market size)

| Finding | Source | Rel. |
| --- | --- | --- |
| Misophonia (strong reactions to specific everyday sounds such as chewing or breathing) causes a significant burden for an estimated **18%** of UK adults (18.4% in press coverage); fewer than 14% of the sample had heard of the condition. Sample of 772 adults matched to UK census figures; measured with the S-Five scale. | Vitoratou S. et al. (2023) "Misophonia in the UK: Prevalence and norms from the S-Five in a UK representative sample", *PLoS ONE* 18(3): e0282777. University of Oxford news release on the study. | S (abstract and press release; full text not read) |
| Auditory processing differences were reported in **over 70%** of autistic children at all three time points from age 3 to 9 (Short Sensory Profile). Other studies report hyperacusis prevalence in autism from 18% to over 80%, depending heavily on the measure used. A 2024 narrative review argues auditory differences shape autistic development more broadly. | The 70% figure is from a 2023 longitudinal study seen only in a search summary; **identify the paper before citing**. Range: meta-analysis by Williams and colleagues (41%), and older studies (18%, 40%), also from search summaries. Review: Poulsen et al. (2024), UCL Discovery. | S |
| In open-plan offices, how **intelligible** nearby speech is predicts distraction better than how loud it is; performance falls as intelligibility rises, and masking that lowers intelligibility improves performance. | "The effect of speech and speech intelligibility on task performance", *Ergonomics* (2006), PubMed 16950722; Roelofsen (2008) "Performance loss in open-plan offices due to noise by speech", *Journal of Facilities Management*; Hedlund et al., Chalmers University (speech exposure reduced working-memory performance compared with silence). | S |
| Popular figures such as "66% less productive near a conversation" could not be traced to a primary study. **Avoid them.** | | |

**Relevance to Earitation.** The app's categories (chewing, typing,
clicking, banging, alarms) are trigger sounds for misophonia and common
office distractions. Note that **speech is not one of the app's categories**,
although the office research says speech is the biggest distractor. Is that a
scope decision to justify, or a gap?

## 2. Possible stakeholders

To be argued in the report; listed here with where evidence above or below
touches them.

- **People sensitive to sounds**: people with misophonia, autistic people,
  people with ADHD, and anyone working in a noisy shared space. They are the
  primary users (section 1).
- **Employers and managers of shared workspaces**: they bear productivity
  costs and have duties to make adjustments for disabled staff. Access to Work
  (section 5) is a funding route.
- **Universities and schools**: libraries, exam halls and study spaces. Not
  researched yet.
- **Other people in the room**: they are recorded without being users, which
  raises privacy issues (section 4).
- **Clinicians such as audiologists and occupational therapists**: only if the
  product made health claims. It does not (NFR4), which keeps it outside
  medical device regulation (section 4).
- **The developer or small company**: costs, licences, hosting (section 3).

## 3. Existing products

### Sound recognition (listens, then alerts)

| Product | What it does | Notes | Source | Rel. |
| --- | --- | --- | --- | --- |
| Apple Sound Recognition (iOS 14 onwards) | Listens for set sounds such as fire and smoke alarms, sirens, cats, dogs, appliances, car horns, doorbells, door knocks, running water, crying babies and shouting, and sends a notification. Custom alarms and appliances can be trained. | Built into iPhone accessibility settings. Aimed at people who are deaf or hard of hearing. | AbilityNet "My Computer My Way" guides (iOS 15 to 18); How-To Geek; AppleInsider (2020) | S |
| Android Sound Notifications (Android 11 onwards) | Similar: about ten sounds including smoke alarms, sirens, doorbells, knocking, dog barks, appliance beeps, running water and crying babies. Alerts by notification, flash or vibration. Runs on the device. | Google says it is not a certified alarm system and that always-on listening uses battery. | Google Accessibility Help, "Sound Notifications" (support.google.com/accessibility/android/answer/10092548) | P |

### Masking and focus audio (plays sound, does not listen)

| Product | Model | Pricing (unverified) | Rel. |
| --- | --- | --- | --- |
| Brain.fm | Focus music, marketed as neuroscience-based | Reported between about $7 and $15 a month; sources conflict | S |
| Endel | Generated soundscapes | Reported about $6 to $7 a month | S |
| Noisli | Mixable background sounds; often recommended for masking speech | Free tier; Pro about $10 a month, billed yearly | S |
| myNoise | Large library of masking noises | Free with donations; paid app unlock | S |

Prices come from review and comparison sites that disagree with each other;
check each vendor's own page before citing them. Marketing claims ("backed by
neuroscience") are not evidence.

### Hardware (not researched yet)

Noise-cancelling headphones and earplugs designed for sound sensitivity are
the obvious alternatives. Access to Work can fund headphones (section 5).

**What was found, stated neutrally.** The sound recognition products listen
and alert, and target deaf and hard-of-hearing users. The focus apps play
audio but do not listen. No product found here does both: listen for a
user-chosen set of distracting sounds and suggest audio in response. The
search was not exhaustive, so this needs a deliberate check (app stores,
academic prototypes) before the report claims a gap.

## 4. Risks, law and regulation

| Topic | Finding | Source | Rel. |
| --- | --- | --- | --- |
| Audio as personal data | Audio recordings in which a person can be identified (for example by voice) are personal data under UK GDPR. | Law-firm commentary (LegalVision); ICO guidance pages on consumer IoT | S |
| Biometric data | The ICO's 2023 draft biometric guidance says a recording counts as biometric (special category) data only when it is analysed to identify or recognise a person, for example a voiceprint. Earitation classifies sound types and does not identify people. An older ICO case (HMRC Voice ID, 2019) treated voice data as biometric because it was used for identification. | BCLP summary of the ICO draft guidance; Taylor Wessing (2019) | S; check whether the ICO guidance has been finalised |
| Household exemption | UK GDPR does not apply to purely personal or household activity. That may cover a person using the app at home, but not a company running the hosted service. | LegalVision | S |
| Medical devices | The MHRA treats software as a medical device according to its **intended purpose**, as stated in labelling, instructions and marketing. General wellbeing apps are usually out of scope. Mental health technology intended to help manage mental ill health can qualify. | GOV.UK "Software and AI as a Medical Device Change Programme roadmap"; GOV.UK "Medical devices: software applications (apps)" (updated 1 July 2023; criteria are in the attached PDF, not yet read); Hill Dickinson and HTN on the 2025 digital mental health guidance | P (roadmap), S (rest) |
| Dataset licence | ESC-50 is licensed under Creative Commons Attribution-NonCommercial (CC BY-NC 3.0). The ESC-10 subset is CC BY. | ESC-50 GitHub README, checked 2026-11-16 | P |
| Model licence | YAMNet code and weights: Apache 2.0. AudioSet ontology and labels: CC BY 4.0. YAMNet uses 521 of AudioSet's 527 classes; six were removed on fairness reviewers' advice. | YAMNet README in tensorflow/models; AudioSet mailing list; Hugging Face model cards | S; check the tensorflow/models LICENSE file |

**Links to the design.** NFR4 (no medical claims) is what keeps the app a
wellbeing tool under the MHRA's intended-purpose test. D2 and NFR1 (no audio
stored) and NFR6 (notice and consent) are the privacy controls. Decision D3
(labels, not an ESC-50-trained model) keeps the non-commercial dataset out of
the shipped product.

## 5. Costs and funding

| Item | Finding | Source | Rel. |
| --- | --- | --- | --- |
| Hosting (proof of concept) | Streamlit Community Cloud is free, with up to 2 CPU cores, 2.7 GB of memory and 50 GB of storage per app. Limits are approximate (as of February 2024) and may change without notice. Apps sleep after 12 hours without traffic. Free workspaces get one private app. | docs.streamlit.io "Manage your app", checked 2026-11-16; Streamlit forum for the private-app limit | P (limits), S (private apps) |
| Measured app needs | About 610 MB of memory with YAMNet loaded (measured locally on 2026-11-10); 0.033 s of YAMNet time per 5-second clip on a laptop CPU (ESC-50 run). | Own measurements, `devlog.md` and `evaluation.md` | P |
| Software licences | TensorFlow, Streamlit and YAMNet are open source (Apache 2.0), so there are no licence fees. ESC-50 is non-commercial but is used only for evaluation. | See section 4 | |
| Funding route for users | Access to Work (UK government) can fund workplace adjustments for disabled and neurodivergent employees, including noise-cancelling headphones when recommended in a needs assessment. Reported annual caps vary (one source says £62,900), so check GOV.UK. | Charity and coaching sites (ADHD UK, RNID, Everway); gov.uk/access-to-work not yet read | S |

Not yet researched: commercial hosting costs at scale, development and
maintenance costs, and a pricing model.

## 6. Still to check before writing the analysis

1. Read the full Vitoratou et al. (2023) paper and note its confidence
   interval. The PubMed Central page was blocked by a browser check.
2. Read gov.uk/access-to-work for current eligibility and the funding cap.
3. Read the MHRA "Medical device stand-alone software including apps" PDF
   for its wellbeing wording.
4. Confirm whether the ICO's biometric guidance is final.
5. Check the tensorflow/models LICENSE file and the YAMNet README.
6. Search app stores and Google Scholar for products or prototypes that both
   detect sounds and respond with audio.
7. Decide how the report handles speech, the biggest office distractor and
   not one of the app's categories.

## Sources consulted

- Vitoratou et al. 2023, via University of Oxford news: https://ox.ac.uk/node/8133630 ; abstract: https://ui.adsabs.harvard.edu/abs/2023PLoSO..1882777V
- Poulsen et al. 2024: https://discovery.ucl.ac.uk/10205392/1/Poulsen%20et%20al%202024.pdf
- Speech intelligibility and task performance (*Ergonomics*, 2006): https://pubmed.ncbi.nlm.nih.gov/16950722/
- Roelofsen 2008: https://www.emeraldinsight.com/doi/abs/10.1108/14725960810885970
- Hedlund et al. (Chalmers): https://research.chalmers.se/publication/553541/file/553541_Fulltext.pdf
- Apple Sound Recognition guides: https://mcmw.abilitynet.org.uk/how-to-detect-sounds-using-your-iphone-or-ipad-with-ios-18 ; https://www.howtogeek.com/693029/how-to-use-sound-recognition-on-iphone/
- Android Sound Notifications: https://support.google.com/accessibility/android/answer/10092548
- Focus app pricing comparisons: https://toolradar.com/compare/brain-fm-vs-noisli ; https://neurolaunch.com/brain-fm-price/
- Audio recording and GDPR: https://legalvision.co.uk/data-privacy-it/audio-recording-cctv-gdpr-dpa-ico/
- ICO draft biometric guidance summary: https://www.bclplaw.com/en-US/events-insights-news/getting-to-know-who-ico-draft-guidance-on-biometric-recognition.html
- MHRA roadmap: https://www.gov.uk/government/publications/software-and-ai-as-a-medical-device-change-programme/software-and-ai-as-a-medical-device-change-programme-roadmap
- MHRA apps guidance: https://www.gov.uk/government/publications/medical-devices-software-applications-apps
- Digital mental health guidance commentary: https://www.hilldickinson.com/insights/articles/regulatory-updates-digital-mental-health-technologies
- ESC-50: https://github.com/karolpiczak/ESC-50
- YAMNet README: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md
- Streamlit Community Cloud limits: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- Access to Work (secondary): https://rnid.org.uk/information-and-support/benefits/access-to-work ; https://www.everway.com/en-gb/govt-grant-schemes/what-is-atw/
