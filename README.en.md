# NeuroQuest EEG pipeline

[Versión en español](README.md)

Acquisition and classification of emotions (Happy, Neutral, Frustrated) with the OpenBCI Ganglion headband (4 channels, 200 Hz). It is the engineering component of the NeuroQuest / NeuroUX undergraduate thesis project and follows the pipeline in Figure Arq-5 of the thesis. It includes a local web dashboard to inspect sessions, the model and the live signal.

**Status:** the pipeline, the recording protocol and the evaluation are implemented and tested with synthetic data. They still have to be checked with the real headband (see [Checks with the headband](#checks-with-the-headband)). No results from real participants are reported here.

## Pipeline

1. **Acquisition.** BrainFlow reads the real EEG channels of the Ganglion. Row 0 of each packet is the packet counter and is kept separately to detect lost packets.
2. **Filtering.** Butterworth band-pass 0.5–45 Hz, order 4, plus a 60 Hz notch (Colombia's power grid) with Q = 30. The filter is causal and runs over 3 s of previous signal, so there are no transients at the most recent edge of the window.
3. **Artifact rejection.** A window is discarded if any channel exceeds ±100 µV (blinks, movement) or if any channel is flat or saturated (standard deviation below 0.5 µV).
4. **FFT.** Periodogram with a Hamming window over 2 s windows.
5. **Band power.** log10 of δ (1–4 Hz), θ (4–8), α (8–13), β (13–30) and γ (30–45) at Fp1, Fp2, T3 and T4: 20 features. The z-score is done by the model.
6. **Model.** StandardScaler + SVM (or Random Forest).

Everything is set in `config.py`.

## Installation

Requires Python 3.9–3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`setuptools` is pinned below 81 because brainflow 5.12.1 still uses `pkg_resources`. On a Mac with Apple Silicon you may need another PyQt5 version; on an Intel Mac, the one in `requirements.txt` works.

## Usage

```bash
# Record a session with the protocol (with plots). The participant code is required
python main.py --mode gui --participant P03

#   or in the terminal, without plots
python main.py --mode record --participant P03

# Train on the sessions in data/sessions (by default it asks for at least 20 usable sessions)
python main.py --mode train
python main.py --mode train --min-sessions 5 --permutations 100

# Classify live
python main.py --mode realtime --participant P03

# Local web dashboard
python main.py --mode dashboard

# Noise recording with nobody connected (Ganglion-4 Bench noise library): N1, N2, N3, C1, P1-P4
python main.py --mode noise --code N1 --seconds 600 --note "inputs shorted to REF"
```

- **Noise.** `--mode noise` records the requested seconds without a protocol and saves the code, the note and the firmware in the header, under `data/noise/`. How to wire each code is in `paper/noise_protocol.md` of the ganglion4-bench repository.
- **Participant.** The code is typed in the GUI or passed with `--participant` and only accepts letters, numbers, `-` and `_` (for example P03). People's names are never used.
- **Synthetic board.** `--synthetic` uses BrainFlow's synthetic board to test without the headband. Those recordings go to `data/synthetic` and are never mixed with real ones.
- **Port.** Changed with `--port`. The default is `/dev/cu.usbmodem11`, the BLED112 dongle on macOS.

### During a recording

1. Open two terminals. In one, `python main.py --mode dashboard`; in the other, `python main.py --mode gui --participant P03`.
2. Before pressing Start, check the electrodes in the GUI. The "Signal" label says which channel has an artifact, is flat or has much more 60 Hz noise than the others, which almost always means an electrode with poor contact. In the FFT tab, that channel shows a large peak at 60 Hz. Adjust until all four channels look right. The GUI saves nothing until Start, so the dashboard's **Live** page shows the signal from that moment.
3. During the protocol, the GUI announces in the **Cue** (the last 3 s of rest) which emotion comes next, so the stimulus is ready and is presented as soon as the block starts.
4. If a channel fails mid-session, press **Stop**: the session is saved as stopped and left out of training. Fix the electrode and start a new session.
5. If the signal stops arriving, the GUI shows "NO DATA" and the dashboard shows how many seconds it has gone without samples: check the headband battery and the dongle.

### Protocol

A 3 s Cue, which also lets the filters settle, and then, in each round (3 by default), each emotion for 15 s, separated by 8 s of rest and a 3 s Cue. The order of the emotions is shuffled in every round (`SHUFFLE_ORDER`), so the label is not tied to the moment in the session, for example to fatigue or the gel drying. The first 2 s of each stimulus (`ONSET_MARGIN`) are left out of training, because a person presents the stimulus after seeing the phase.

### Files

| What | Where | Format |
|---|---|---|
| Sessions with the protocol | `data/sessions/session_<participant>_<date>.csv` | Raw signal, one row per sample |
| `realtime` recordings | `data/live/live_<date>.csv` | Same as sessions, with the `Live` label |
| Everything recorded with `--synthetic` | `data/synthetic/` | Same |
| Live predictions (realtime and GUI) | `logs/<recording>_predictions.csv` | One row per second: `ok`, `artifact` or `no_data` |
| Model and its evaluation | `models/<model>.pkl`, `.report.json`, `.cv.csv`, `.confusion_matrix.png` | |

**Recording format.**
- **Header:** `# key=value` lines with the sampling rate, the board, the participant, the start time and the planned protocol steps.
- **Columns:** `timestamp,package,Fp1,Fp2,T3,T4,label,round,block`.
- **Closing line:** a final `# completed=...` line if the protocol finished, or `# status=stopped` if it was stopped. Without that line, the session counts as incomplete (it was cut off).
- **Labels:** each sample carries the phase that matches its timestamp, not the phase at the moment it was read.
- **Features:** computed at training time, so changing the pipeline does not require recording again.

`data/`, `models/` and `logs/` are in `.gitignore`. Participant data must not be pushed to the repository.

### Evaluation

- **Which sessions are used.** Only complete ones from `data/sessions`. Stopped, incomplete, synthetic and old-format sessions and empty files are left out, with the reason printed in the terminal and in the report. `--include-partial` also includes stopped and incomplete sessions.
- **Groups.** Windows from the same participant, session or round resemble each other, so they are never split between training and test. The strictest available unit is used: participants (with 3 or more), sessions (with 3 or more) or rounds. With rounds, the result only holds within a session already seen.
- **One rule.** Cross-validation, the hyperparameter search and live mode decide the same way: the class with the highest probability. Confidence is recalibrated with temperature scaling fitted on the grouped validation, so it does not promise more than the model gets right.
- **Chance.** Chance is computed from the classes present in each fold and each session. A session with a single class has no defined chance level and does not count in that comparison. Only `--permutations N` shows whether the model beats chance: it repeats the whole evaluation with labels permuted by block and gives a p-value. The smallest possible p is 1/(N+1). With 19 permutations it reaches exactly 0.05, and p ≤ 0.05 counts as beating chance. For the thesis, 100 is recommended. With 20 sessions, 100 permutations take about 45–60 min on a 4-core Mac.
- **What is saved.** Everything goes to `models/<model>.report.json` and `models/<model>.cv.csv`, which the dashboard shows.

## Dashboard

`python main.py --mode dashboard` opens a page in the browser at `http://127.0.0.1:8501`. It only reads `data/`, `logs/` and `models/` on this computer: it never opens the headband, listens only on this computer and has Streamlit's usage statistics turned off.

- **Sessions.** Each recording with its summary (duration, samples per second, lost packets and discarded windows) and a timeline with the phases, what the model predicted in each window, the discarded windows and the 4 channels. Below, quality per channel and the actual duration of each block against the planned one.
- **Live.** Updates every second with what is being recorded. It shows which block the protocol is in and which emotion comes next, the last 10 s of signal, a warning per channel, and in `realtime` the probabilities over the last minutes.
- **Model.** The accuracy of each session against its chance level, the permutation test, the confusion matrix, the calibration, where it fails (by second of the block, by round and the worst blocks) and how much the power of each emotion changes relative to rest.
- **Compare.** The spectral fingerprint and the discarded windows per session, day or participant.

In Sessions, Model and Compare, **Prepare download** builds an HTML of the page that opens without internet, to send to the advisor or the committee. Each chart also has a camera button to download it as PNG.

## Checks with the headband

What still has to be verified with the real Ganglion, with the dashboard open:

1. **Montage.** When blinking, only Fp1 and Fp2 should move. When touching each electrode, its channel should move.
2. **Samples.** Record a full session. In Sessions, close to 200 samples per second should arrive, and the lost-packet count should make sense. The header stores the firmware reported by BrainFlow (`# firmware=2` or `3`), and `analysis.lost_packets` counts according to that firmware: firmware 2 numbers packets 1–100 and 101–200 (0 is a packet after a reset) and firmware 3 numbers them 0–99 and 100–199. That numbering was taken from BrainFlow 5.12.1's code and still has to be confirmed with the headband. BrainFlow 5.12.1 detects the firmware on its own; newer versions assume 3 unless told otherwise, which is why the connection passes `other_info='fw:auto'`. Before upgrading BrainFlow, check that the header still reports the right firmware.
3. **Block duration.** Each block should last as planned: 15 s of stimulus, 8 s of rest and a 3 s Cue.
4. **Discarded windows.** Check what percentage is discarded. If blinks knock out more than 30 %, decide from the data whether to raise `ARTIFACT_UV`, move the electrodes or handle the EOG. The dashboard's 60 Hz noise thresholds (`LINE_RATIO`, `LINE_MIN_UV` in `analysis.py`) are also tuned with real sessions.
5. **Training.** With 3 or more sessions, ideally from 3 or more people, train with `--permutations 100` and read the Model page.
6. **Normalization.** Arq-5 says "z-score" but not whether it is global or per session. With real data, compare a global z-score against a per-session one (or subtracting the rest baseline) and keep the one that generalizes better.

## Tests

```bash
python test_pipeline.py    # pipeline, recording, training, real time and GUI (headless)
python test_dashboard.py   # every dashboard page, the export and startup
```

**`test_pipeline.py`** generates synthetic sessions where emotions change alpha and beta power, with 60 Hz noise and blinks. It checks:
- filters, artifact rejection and labeling by timestamp;
- which sessions are left out;
- that the model learns when there is signal and gives chance when there is none;
- the single decision rule and the calibration;
- the permutations;
- what training saves;
- the warning when the headband stops sending data;
- the full GUI in offscreen mode with the synthetic board.

**`test_dashboard.py`** runs every page with and without data and with a recording in progress. It also checks that the exported HTML does not depend on the internet and that the dashboard never imports brainflow.
