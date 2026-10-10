"""
Recordings keep the raw EEG (not features), one row per sample, labeled with the protocol phase.
Features are computed when training, so changing the pipeline never forces a new recording.

File format: '# key=value' header lines (sampling_rate, board, participant, started, protocol...),
then timestamp,package,Fp1,Fp2,T3,T4,label,round,block, then a closing '# completed=...' or
'# status=stopped' line.
"""

import csv
import os
import random
import time
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd

import config
from preprocessing import Preprocessor, feature_names


Step = namedtuple('Step', 'label seconds round next')


def protocol_steps():
    """
    Steps of a full recording run. A Cue announces the next emotion so whoever presents the stimulus
    has it ready: the run opens with a Cue (it also lets the filters settle) and every stimulus but the
    last is followed by a Rest and a Cue. Shuffled with SHUFFLE_ORDER, a new order each round.
    """
    stimuli = []
    for round_num in range(1, config.TRAINING_ROUNDS + 1):
        order = list(config.CLASSES)
        if config.SHUFFLE_ORDER:
            random.shuffle(order)
        stimuli += [(emotion, round_num) for emotion in order]
    if not stimuli:
        return []
    steps = [Step(config.CUE_LABEL, max(config.CUE_SECONDS, config.FILTER_PAD), 1, stimuli[0][0])]
    for i, (emotion, round_num) in enumerate(stimuli):
        steps.append(Step(emotion, config.SAMPLING_DURATION, round_num, None))
        if i + 1 < len(stimuli):
            upcoming, next_round = stimuli[i + 1]
            steps.append(Step(config.REST_LABEL, config.REST_DURATION - config.CUE_SECONDS, round_num, None))
            steps.append(Step(config.CUE_LABEL, config.CUE_SECONDS, next_round, upcoming))
    return steps


def protocol_meta(steps):
    """Header lines with the planned steps (label:seconds:round), so the dashboard can follow a run."""
    return {'protocol': 'emotions', 'blocks': len(steps), 'rounds': config.TRAINING_ROUNDS,
            'steps': ';'.join(f'{s.label}:{s.seconds:g}:{s.round}' for s in steps)}


def planned_steps(meta):
    """The steps written by protocol_meta: [(label, seconds, round)] by block, or [] for a live run."""
    steps = []
    for item in filter(None, meta.get('steps', '').split(';')):
        label, seconds, round_num = item.split(':')
        steps.append((label, float(seconds), int(round_num)))
    return steps


def recording_dir(synthetic, live=False):
    if synthetic:
        return config.SYNTHETIC_DIR
    return config.LIVE_DIR if live else config.SESSIONS_DIR


def predictions_path(recording_path):
    """Predictions made during a recording go to logs/, so the session globs never pick them up."""
    stem = os.path.splitext(os.path.basename(recording_path))[0]
    return os.path.join(config.LOGS_DIR, f'{stem}_predictions.csv')


class SessionRecorder:
    """
    Raw EEG of one recording, one row per sample. Each sample takes the phase of the last mark before
    its BrainFlow timestamp (the same clock as time.time()), not the phase current when it was read.
    The file ends with '# completed=...' or '# status=stopped'; without that line it is incomplete.
    """

    def __init__(self, sampling_rate, base_dir=config.SESSIONS_DIR, prefix='session', participant=None,
                 board='ganglion', **meta):
        os.makedirs(base_dir, exist_ok=True)
        now = datetime.now()
        name = '_'.join(part for part in (prefix, participant, now.strftime('%Y%m%d_%H%M%S')) if part)
        self.path = self._create(base_dir, name)
        header = {'sampling_rate': sampling_rate, 'board': board, 'participant': participant or '',
                  'started': now.isoformat(timespec='seconds'), **meta}
        for key, value in header.items():
            self._file.write(f'# {key}={value}\n')
        self._writer = csv.writer(self._file)
        self._writer.writerow(['timestamp', 'package', *config.CHANNEL_NAMES, 'label', 'round', 'block'])
        self._file.flush()
        self._marks = []
        self._mark_times = np.array([])
        self._end = None

    def _create(self, base_dir, name):
        for i in range(1000):
            path = os.path.join(base_dir, f"{name}{f'_{i}' if i else ''}.csv")
            try:
                self._file = open(path, 'x', newline='')
                return path
            except FileExistsError:
                continue
        raise FileExistsError(f"Too many recordings named {name} in {base_dir}")

    def mark(self, label, round_num, block, t=None):
        """From time t (default: now) on, samples belong to this phase."""
        self._marks.append((label, round_num, block))
        self._mark_times = np.append(self._mark_times, time.time() if t is None else t)

    def mark_end(self, t=None):
        """Samples from time t (default: now) on are not recorded."""
        self._end = time.time() if t is None else t

    def write(self, timestamps, samples, package=None):
        """Append the samples that fall inside a phase. Returns how many were written."""
        timestamps = np.asarray(timestamps, dtype=float)
        if not len(timestamps) or not self._marks:
            return 0
        phase = np.searchsorted(self._mark_times, timestamps, side='right') - 1
        keep = phase >= 0
        if self._end is not None:
            keep &= timestamps < self._end
        package = np.full(len(timestamps), -1) if package is None else np.asarray(package)
        for i in np.flatnonzero(keep):
            label, round_num, block = self._marks[phase[i]]
            self._writer.writerow([f'{timestamps[i]:.4f}', int(package[i]), *(f'{v:.3f}' for v in samples[i]),
                                   label, round_num, block])
        self._file.flush()
        return int(keep.sum())

    def close(self, status=None):
        """status 'completed' or 'stopped' writes the closing line; None leaves the file incomplete."""
        if self._file.closed:
            return
        if status == 'completed':
            self._file.write(f"# completed={datetime.now().isoformat(timespec='seconds')}\n")
        elif status:
            self._file.write(f'# status={status}\n')
        self._file.close()


def announce(step):
    if step.label == config.CUE_LABEL:
        print(f"\nNEXT: {step.next} (round {step.round}/{config.TRAINING_ROUNDS}) in {step.seconds:.0f}s, "
              f"get the stimulus ready")
    elif step.label == config.REST_LABEL:
        print(f"REST {step.seconds:.0f}s")
    else:
        print(f"STIMULUS: show '{step.label}' now ({step.seconds:.0f}s)")


def record_session(daq, participant=None, synthetic=False):
    """Run the protocol in the terminal while recording the raw stream."""
    steps = protocol_steps()
    recorder = SessionRecorder(daq.sampling_rate, recording_dir(synthetic), participant=participant,
                               board='synthetic' if synthetic else 'ganglion', **protocol_meta(steps))
    print(f"Recording to {recorder.path}")
    daq.read_new()  # drop what arrived while connecting
    status = 'stopped'
    try:
        for block, step in enumerate(steps):
            recorder.mark(step.label, step.round, block)
            announce(step)
            written = 0
            end = time.time() + step.seconds
            while time.time() < end:
                time.sleep(0.05)
                timestamps, package, samples = daq.read_new()
                written += recorder.write(timestamps, samples, package)
            expected = step.seconds * daq.sampling_rate
            if written < 0.9 * expected:
                print(f"  WARNING: only {written} of about {expected:.0f} samples arrived in this block; "
                      "check the headset and the dongle")
        recorder.mark_end()
        timestamps, package, samples = daq.read_new()
        recorder.write(timestamps, samples, package)
        status = 'completed'
    except KeyboardInterrupt:
        recorder.mark_end()
        print("\nStopped: the session is kept but marked as stopped, so training leaves it out.")
    finally:
        recorder.close(status)
    print(f"Session saved: {recorder.path}")
    return recorder.path


REQUIRED_COLUMNS = ['timestamp', *config.CHANNEL_NAMES, 'label', 'round', 'block']

# Why a recording is left out of training (shown by train and by the dashboard)
SKIP_REASONS = {
    'empty': 'empty file',
    'old_feature_format': 'features CSV from the old GUI, it cannot be retrained: record it again',
    'unknown_format': 'unknown format',
    'synthetic': 'recorded with --synthetic',
    'incomplete': 'incomplete recording (no closing line)',
    'stopped': 'stopped before the end of the protocol',
    'no_windows': 'no usable windows',
    'error': 'could not be read',
}


class SessionError(Exception):
    def __init__(self, code):
        super().__init__(SKIP_REASONS.get(code, code))
        self.code = code


def read_header(path):
    """
    The leading '# key=value' lines, the column names and the closing line written at the end
    ('# completed=...' or '# status=...'). Returns (meta, columns).
    """
    meta = {}
    with open(path, newline='') as f:
        line = f.readline()
        while line.startswith('#'):
            key, _, value = line[1:].strip().partition('=')
            meta[key.strip()] = value.strip()
            line = f.readline()
        columns = line.strip().split(',') if line.strip() else []
    with open(path, 'rb') as f:
        f.seek(max(0, os.path.getsize(path) - 256))
        tail = f.read().decode('utf-8', 'ignore').splitlines()
    for line in tail[-3:]:
        if line.startswith('# completed='):
            meta['completed'] = line.split('=', 1)[1].strip()
        elif line.startswith('# status='):
            meta['status'] = line.split('=', 1)[1].strip()
    return meta, columns


def session_status(meta):
    """'completed', 'stopped' or 'incomplete' (still being written, or cut by a crash)."""
    if 'completed' in meta:
        return 'completed'
    return meta.get('status') or 'incomplete'


def format_problem(columns):
    if not columns:
        return 'empty'
    if any(c.startswith('feature_') for c in columns):
        return 'old_feature_format'
    if any(c not in columns for c in REQUIRED_COLUMNS):
        return 'unknown_format'
    return ''


def load_session(path):
    """(samples, meta) of one recording; meta['sampling_rate'] is a float."""
    meta, columns = read_header(path)
    problem = format_problem(columns)
    if problem:
        raise SessionError(problem)
    df = pd.read_csv(path, comment='#')
    meta['sampling_rate'] = float(meta.get('sampling_rate', config.SAMPLING_RATE))
    return df, meta


TABLE_COLUMNS = (['session', 'name', 'participant', 'block', 'round', 'label', 'start', 'end', 't_start',
                  'artifact', 'reason', 'onset'] + [f'peak_{ch}' for ch in config.CHANNEL_NAMES])


def window_table(path, df=None, meta=None):
    """
    One row per candidate window of a recording, in every block (emotions, Rest, Cue, Live), with
    its artifact flag, the peak uV of each channel and the 20 features. Each window lies fully inside
    its block; the filter pad may reach back into the previous block, exactly like in real time.
    Training and the dashboard both use it, so they never disagree.
    """
    if df is None:
        df, meta = load_session(path)
    fs = meta['sampling_rate']
    prep = Preprocessor(fs)
    raw = df[config.CHANNEL_NAMES].to_numpy(dtype=float)
    step = max(1, int(prep.window * (1 - config.OVERLAP)))
    margin = int(round(config.ONSET_MARGIN * fs))
    names = feature_names()

    rows = []
    for block, part in df.groupby('block', sort=False):
        start, stop = int(part.index[0]), int(part.index[-1]) + 1
        if stop - start != len(part):
            print(f"{os.path.basename(path)}: block {block} is not contiguous, skipped")
            continue
        label = str(part['label'].iat[0])
        emotion = label in config.CLASSES
        for end in range(start + prep.window, stop + 1, step):
            features, artifact, reason, peaks = prep.analyze_window(raw[max(0, end - prep.window - prep.pad):end])
            first = end - prep.window
            rows.append([int(block), int(part['round'].iat[0]), label, first, end, first / fs,
                         artifact, reason, emotion and first < start + margin, *peaks, *features])

    table = pd.DataFrame(rows, columns=TABLE_COLUMNS[3:] + names)
    table.insert(0, 'participant', meta.get('participant', ''))
    table.insert(0, 'name', os.path.splitext(os.path.basename(path))[0])
    table.insert(0, 'session', os.path.relpath(path))
    return table


def training_windows(table):
    """Emotion windows without artifacts and outside the onset margin."""
    keep = table['label'].isin(config.CLASSES) & ~table['artifact'] & ~table['onset']
    return table[keep]


def load_sessions(paths, include_partial=False, allow_synthetic=False):
    """
    Window tables of several recordings, leaving out the ones that must not be trained on.
    Returns (table, excluded); excluded is a list of {'file', 'reason'} with a SKIP_REASONS code.
    """
    tables, excluded = [], []
    for path in paths:
        name = os.path.basename(path)
        try:
            meta, columns = read_header(path)
            code = format_problem(columns)
            if not code and meta.get('board') == 'synthetic' and not allow_synthetic:
                code = 'synthetic'
            if not code and session_status(meta) != 'completed' and not include_partial:
                code = session_status(meta)
            if code:
                raise SessionError(code)
            table = window_table(path)
            if not len(training_windows(table)):
                raise SessionError('no_windows')
        except SessionError as e:
            excluded.append({'file': os.path.relpath(path), 'reason': e.code})
            print(f"{name}: skipped, {e}")
            continue
        except Exception as e:
            excluded.append({'file': os.path.relpath(path), 'reason': 'error'})
            print(f"{name}: skipped, could not be read ({e})")
            continue

        emotion = table[table['label'].isin(config.CLASSES)]
        rejected = int(emotion['artifact'].sum())
        print(f"{name}: {len(training_windows(table))} training windows, {rejected} of {len(emotion)} "
              f"rejected by artifacts ({100 * rejected / max(len(emotion), 1):.0f}%)")
        tables.append(table)

    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=TABLE_COLUMNS)
    return table, excluded
