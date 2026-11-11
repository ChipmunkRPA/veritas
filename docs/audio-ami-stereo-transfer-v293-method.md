# Stereo decision transfer on genuine AMI speech

Use all 24 v291 stereo windows from the two genuine AMI meetings. Bind the
completed source replay, its private qualification record, and the exact
capture clock and file commitments. Every planned window remains in the
denominator, including quiet portions and windows without speech admission.
The preparation reads the sealed metadata. The first inference stage opens
only the exact already qualified stereo WAVs. Keep sample coordinates,
activity spans and scores private. [Genuine stereo source preparation](audio-ami-native-stereo-v291-result.md)

Reuse the v279 production mono gate and the unchanged acoustic channel choice
on the two observed microphones. Verify native channel identity, mono PCM
and the case clock before each activity pass. Repeat the channel-selection
calculation with the physical channels swapped. This is 72 activity views
and exactly 6,768 frame forwards. Score every mono view and every selected
channel admitted by the gate, using the frozen v261/v284 classifier and
stability probe. A separate classifier plan binds those eligible views
before scoring, with at most 48 views and 96 forwards. [Channel rule](audio-native-stereo-gate-v279-method.md)
· [Classifier and fixed policy](audio-native-stereo-holdout-v284-method.md)

Compare mono with the unchanged v283 policy. It preserves mono AI or original
directions and can add AI only when mono is unassessed or inconclusive. Count
all AI directions as false alarms on this genuine-origin cohort. Report both
sessions and the combined counts for speech admission, assessed results,
original, false AI, inconclusive and unassessed. The 12 windows within a
meeting share the session. This pilot measures genuine-only coverage and
false AI flags. Synthetic recall and paired positive transfer remain separate.
[Fixed decision policy](audio-ai-only-rescue-v283-method.md)

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_stereo_transfer_v293 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_stereo_transfer_v293 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_stereo_transfer_v293 run-gate
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_stereo_transfer_v293 gate-readout
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_stereo_transfer_v293 seal-classifier
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_stereo_transfer_v293 run-classifier
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_stereo_transfer_v293 readout
```

Both inference stages use the existing offline resource guard. The final
readout validates the terminal receipts and independently reconstructs the
24 paired decisions from the saved score and activity records.
