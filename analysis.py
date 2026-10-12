"""
Computations behind the dashboard, without Streamlit: reading recordings (also while they are being
written), timing and lost packets, signal quality per channel, and the tables behind every chart.
It only reads files; it never opens the board.
"""

import glob
import io
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal

import config
from preprocessing import Preprocessor, feature_names
from sessions import format_problem, planned_steps, predictions_path, read_header, session_status, window_table

# A channel with this many times the 60 Hz noise of the others (and at least LINE_MIN_UV) probably has a
# loose electrode. Not calibrated yet: compare with real sessions (README, revision with the headset).
LINE_RATIO = 3.0
LINE_MIN_UV = 5.0
BYTES_PER_ROW = 90  # generous size of one CSV row, to know how far back to read the tail


def recordings():
    """Every recording (protocol sessions, live runs and synthetic ones), newest first."""
    paths = []
    for folder in (config.SESSIONS_DIR, config.LIVE_DIR, config.SYNTHETIC_DIR):
        paths += glob.glob(os.path.join(folder, '*.csv'))
    return sorted(paths, key=os.path.getmtime, reverse=True)


def reports():
    """Training reports, newest first."""
    return sorted(glob.glob(os.path.join(os.path.dirname(config.MODEL_PATH) or '.', '*.report.json')),
                  key=os.path.getmtime, reverse=True)


def file_key(path):
    """(path, modification time, size): changes whenever the file does, so it keys the caches."""
    stat = os.stat(path)
    return path, stat.st_mtime_ns, stat.st_size


def age_seconds(path, now=None):
    return (time.time() if now is None else now) - os.path.getmtime(path)


def recording_info(path):
    meta, columns = read_header(path)
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        started = datetime.fromisoformat(meta['started'])
    except (KeyError, ValueError):
        started = datetime.fromtimestamp(os.path.getmtime(path))
    return {
        'path': path,
        'name': name,
        'kind': 'live' if name.startswith('live_') else 'session',
        'synthetic': meta.get('board') == 'synthetic',
        'participant': meta.get('participant', ''),
        'started': started,
        'sampling_rate': float(meta.get('sampling_rate', config.SAMPLING_RATE)),
        'status': session_status(meta),
        'board': meta.get('board', ''),
        'firmware': int(meta['firmware']) if meta.get('firmware', '').isdigit() else None,
        'steps': planned_steps(meta),
        'problem': format_problem(columns),
        'meta': meta,
    }


def _parse_rows(text, columns):
    rows = [line for line in text.split('\n') if line and not line.startswith('#') and not line.startswith('timestamp')]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(io.StringIO('\n'.join(rows)), names=columns)


def load_complete(path):
    """Whole recording up to its last complete line (it may still be growing). Returns (df, meta)."""
    meta, columns = read_header(path)
    with open(path, 'rb') as f:
        text = f.read().decode('utf-8', 'ignore')
    text = text[:text.rfind('\n') + 1]
    meta['sampling_rate'] = float(meta.get('sampling_rate', config.SAMPLING_RATE))
    return _parse_rows(text, columns), meta


def read_tail(path, seconds):
    """
    The last `seconds` of a recording that may still be growing: reads from the end of the file,
    drops the first line (cut by the seek) and a last line still being written. Returns (df, meta).
    """
    meta, columns = read_header(path)
    fs = float(meta.get('sampling_rate', config.SAMPLING_RATE))
    meta['sampling_rate'] = fs
    size = os.path.getsize(path)
    start = max(0, size - int(seconds * fs * BYTES_PER_ROW) - 4096)
    with open(path, 'rb') as f:
        f.seek(start)
        text = f.read().decode('utf-8', 'ignore')
    if start > 0:
        text = text[text.find('\n') + 1:]
    text = text[:text.rfind('\n') + 1]
    df = _parse_rows(text, columns)
    if len(df):
        df = df[df['timestamp'] >= df['timestamp'].iloc[-1] - seconds].reset_index(drop=True)
    return df, meta


def read_predictions(path):
    """(predictions, model path) of a predictions file, or (empty frame, '') if there is none."""
    if not path or not os.path.exists(path):
        return pd.DataFrame(), ''
    with open(path) as f:
        first = f.readline()
    model = first.split('=', 1)[1].strip() if first.startswith('# model=') else ''
    return pd.read_csv(path, comment='#'), model


def predictions_for(recording_path):
    path = predictions_path(recording_path)
    return path if os.path.exists(path) else None


# Packet ids of the Ganglion for each compression (18 and 19 bits), as BrainFlow decodes them
# (src/board_controller/openbci/ganglion.cpp, decompress_firmware_2/3). Firmware 2 also sends id 0, an
# uncompressed packet after a reset.
GANGLION_IDS = {2: ((1, 100), (101, 200)), 3: ((0, 99), (100, 199))}


def _packet_ids(package):
    """One id per BLE packet: both samples of a packet share it."""
    return package[np.r_[True, package[1:] != package[:-1]]]


def ganglion_firmware(package):
    """
    Firmware guessed from the packet ids when the header does not say it, or None. Only firmware 2
    reaches 200 or wraps from 100 to 1; only firmware 3 wraps from 99 to 0 or from 199 to 100. A second
    of recording is enough.
    """
    ids = _packet_ids(np.asarray(package, dtype=int))
    pairs = set(zip(ids[:-1].tolist(), ids[1:].tolist()))
    if 200 in ids or (100, 1) in pairs:
        return 2
    if (99, 0) in pairs or (199, 100) in pairs:
        return 3
    return None


def lost_packets(package, board, firmware=None):
    """
    Lost samples according to BrainFlow's package counter, or None when it cannot tell.
    Synthetic board: the counter goes 0-255, one step per sample. Ganglion: every BLE packet carries an
    id that both of its samples share and that cycles inside its compression's range (GANGLION_IDS),
    so `firmware` (from the recording header, or guessed from the ids) is needed. Still to confirm with
    the real headset (README, revision with the headset). Returns (lost samples, expected samples).
    """
    package = np.asarray(package, dtype=int)
    if len(package) < 2 or (package < 0).any():
        return None
    if board == 'synthetic':
        steps = np.diff(package) % 256
        lost = int(np.sum(np.where(steps == 0, 255, steps - 1)))
        return lost, len(package) + lost
    if board == 'ganglion':
        ranges = GANGLION_IDS.get(firmware or ganglion_firmware(package))
        if ranges is None:
            return None

        def span(i):
            return next((r for r in ranges if r[0] <= i <= r[1]), None)

        ids = _packet_ids(package)
        lost = 0
        for a, b in zip(ids[:-1], ids[1:]):
            if span(a) is None or span(a) != span(b):
                continue  # a reset or a change of compression: not a loss
            lost += 2 * int((b - a - 1) % 100)
        return lost, len(package) + lost
    return None


def timing(df, fs, board, firmware=None):
    """
    Duration by the clock, samples per second actually received, stalls and lost samples of a recording.
    Each stall has 'signal_at' (seconds of signal before it, the x of the timeline), 'at' (seconds of clock
    since the start) and 'seconds' (how long nothing arrived).
    """
    ts = df['timestamp'].to_numpy(dtype=float)
    n = len(ts)
    span = ts[-1] - ts[0] if n > 1 else 0.0
    dt = np.diff(ts)
    stalls = [{'signal_at': (i + 1) / fs, 'at': float(ts[i] - ts[0]), 'seconds': float(dt[i])}
              for i in np.flatnonzero(dt > config.GAP_SECONDS)]
    lost = lost_packets(df['package'], board, firmware) if 'package' in df else None
    return {
        'samples': n,
        'seconds': span + 1 / fs if n else 0.0,
        'signal_seconds': n / fs if fs else 0.0,
        'effective_fs': (n - 1) / span if span > 0 else float('nan'),
        'stalls': stalls,
        'lost': None if lost is None else lost[0],
        'expected': None if lost is None else lost[1],
    }


def line_noise(raw, fs):
    """Amplitude (RMS, uV) of the 59-61 Hz band of each channel of the raw signal."""
    if len(raw) < fs:
        return np.full(raw.shape[1], np.nan)
    freqs, psd = signal.welch(raw, fs=fs, nperseg=int(fs), axis=0)
    band = (freqs >= config.NOTCH_FREQ - 1) & (freqs <= config.NOTCH_FREQ + 1)
    return np.sqrt(psd[band].sum(axis=0) * (freqs[1] - freqs[0]))


def raw_spectrum(df, fs):
    """Welch spectrum of the raw signal of each channel: (freqs, psd [freqs, channels])."""
    raw = df[config.CHANNEL_NAMES].to_numpy(dtype=float)
    return signal.welch(raw - raw.mean(axis=0), fs=fs, nperseg=int(2 * fs), axis=0)


def channel_quality(df, fs, table):
    """One row per channel: filtered RMS, 60 Hz noise in the raw signal, windows over the threshold, flat time."""
    raw = df[config.CHANNEL_NAMES].to_numpy(dtype=float)
    filtered = Preprocessor(fs).apply_filters(raw)
    second = int(fs)
    chunks = filtered[:len(filtered) // second * second].reshape(-1, second, raw.shape[1])
    flat = (chunks.std(axis=1) < config.FLAT_UV).mean(axis=0) if len(chunks) else np.zeros(raw.shape[1])
    line = line_noise(raw, fs)
    rows = []
    for i, ch in enumerate(config.CHANNEL_NAMES):
        over = float((table[f'peak_{ch}'] > config.ARTIFACT_UV).mean()) if len(table) else 0.0
        rows.append({'channel': ch, 'rms_uv': float(filtered[int(config.FILTER_PAD * fs):, i].std()),
                     'line_uv': float(line[i]), 'over_pct': 100 * over, 'flat_pct': 100 * float(flat[i])})
    return pd.DataFrame(rows)


def blocks(df, table, info):
    """One row per block: label, round, real and planned seconds, windows, rejected and in the onset margin."""
    fs = info['sampling_rate']
    steps = info['steps']
    ts = df['timestamp'].to_numpy(dtype=float)
    rows = []
    for block, part in df.groupby('block', sort=False):
        windows = table[table['block'] == block]
        planned = steps[int(block)][1] if int(block) < len(steps) else None
        seconds = ts[part.index[-1]] - ts[part.index[0]] + 1 / fs  # by the clock: lost samples do not shorten it
        rows.append({'block': int(block), 'label': str(part['label'].iat[0]), 'round': int(part['round'].iat[0]),
                     'seconds': seconds, 'planned': planned, 'windows': int(len(windows)),
                     'rejected': int(windows['artifact'].sum()), 'onset': int(windows['onset'].sum())})
    return pd.DataFrame(rows)


def filtered_signal(df, fs):
    """The whole recording filtered like the classifier sees it."""
    return Preprocessor(fs).apply_filters(df[config.CHANNEL_NAMES].to_numpy(dtype=float))


def decimate(t, y, max_points):
    """
    Min/max decimation per channel: two points per bucket keep every peak (blinks stay visible).
    Returns a list of (t, y) per channel.
    """
    n, channels = y.shape
    if n <= max_points:
        return [(t, y[:, c]) for c in range(channels)]
    size = int(np.ceil(n / (max_points // 2)))
    m = n // size * size
    chunks = y[:m].reshape(-1, size, channels)
    lo, hi = chunks.argmin(axis=1), chunks.argmax(axis=1)
    base = (np.arange(len(chunks)) * size)[:, None]
    idx = np.stack([base + np.minimum(lo, hi), base + np.maximum(lo, hi)], axis=1).reshape(-1, channels)
    return [(t[idx[:, c]], y[idx[:, c], c]) for c in range(channels)]


def live_status(tail, fs):
    """
    What the operator should know about each channel right now: the newest window filtered like the
    classifier sees it, plus the 60 Hz noise of the last 2 s of raw signal. None until there is enough.
    """
    prep = Preprocessor(fs)
    need = prep.window + prep.pad
    raw = tail[config.CHANNEL_NAMES].to_numpy(dtype=float)
    if len(raw) < need:
        return None
    window = prep.clean_window(raw[-need:])
    line = line_noise(raw[-int(2 * fs):], fs)
    rows = []
    for i, ch in enumerate(config.CHANNEL_NAMES):
        others = np.delete(line, i)
        rows.append({'channel': ch, 'peak_uv': float(np.abs(window[:, i]).max()), 'std_uv': float(window[:, i].std()),
                     'line_uv': float(line[i]),
                     'line_ratio': float(line[i] / max(np.median(others), 1e-6))})
    return pd.DataFrame(rows)


def channel_warnings(status):
    """Sentences (Spanish) for the channels that need attention; empty when all look fine."""
    out = []
    for row in status.itertuples():
        if row.std_uv < config.FLAT_UV:
            out.append(f"{row.channel} está plano: el electrodo puede estar suelto o la entrada saturada.")
        elif row.peak_uv > config.ARTIFACT_UV:
            out.append(f"{row.channel} pasa de ±{config.ARTIFACT_UV:.0f} µV (llegó a {row.peak_uv:.0f} µV): "
                       "parpadeo, movimiento o mal contacto.")
        if row.line_ratio >= LINE_RATIO and row.line_uv >= LINE_MIN_UV:
            out.append(f"{row.channel} tiene {row.line_ratio:.0f} veces más ruido de 60 Hz que los otros canales "
                       f"({row.line_uv:.0f} µV): revisa el contacto de ese electrodo.")
    return out


def progress(tail, info):
    """Where a protocol run is: block, label, round, seconds left and the next emotion."""
    if not len(tail):
        return None
    block = int(tail['block'].iat[-1])
    steps = info['steps']
    current = tail[tail['block'] == block]
    elapsed = current['timestamp'].iat[-1] - current['timestamp'].iat[0] + 1 / info['sampling_rate']
    planned = steps[block][1] if block < len(steps) else None
    upcoming = next((label for label, _, _ in steps[block + 1:] if label in config.CLASSES), None)
    return {'block': block, 'blocks': len(steps), 'label': str(current['label'].iat[-1]),
            'round': int(current['round'].iat[-1]), 'elapsed': elapsed,
            'left': None if planned is None else max(planned - elapsed, 0.0), 'next': upcoming}


def load_report(json_path):
    """(report, cv windows) of a training, or (None, None)."""
    if not json_path or not os.path.exists(json_path):
        return None, None
    with open(json_path) as f:
        report = json.load(f)
    cv_path = json_path[:-len('.report.json')] + '.cv.csv'
    cv = None
    if os.path.exists(cv_path):
        cv = pd.read_csv(cv_path)
        cv['participant'] = cv['participant'].fillna('').astype(str)
    return report, cv


def balanced_accuracy(labels, predicted):
    recalls = [np.mean(predicted[labels == c] == c) for c in np.unique(labels)]
    return float(np.mean(recalls)) if recalls else float('nan')


def session_scores(cv):
    """Balanced accuracy of each session in the grouped validation, with the chance level of that session."""
    rows = []
    for session, part in cv.groupby('session', sort=False):
        labels, predicted = part['label'].to_numpy(), part['predicted'].to_numpy()
        present = len(np.unique(labels))
        rows.append({'session': session, 'name': part['name'].iat[0],
                     'participant': str(part['participant'].iat[0]) if 'participant' in part else '',
                     'windows': int(len(part)), 'balanced_accuracy': balanced_accuracy(labels, predicted),
                     # with a single class left any constant answer scores 100 %: no chance level to compare with
                     'chance': 1 / present if present > 1 else float('nan')})
    return pd.DataFrame(rows)


def error_patterns(cv, tables):
    """
    Where the model fails: accuracy by second inside the block, by round, and the worst blocks.
    tables: window tables of the sessions (to know where each block starts).
    """
    starts = tables.groupby(['session', 'block'])['t_start'].min().rename('block_start')
    cv = cv[cv['session'].isin(tables['session'].unique())].join(starts, on=['session', 'block'])
    cv = cv[cv['block_start'].notna()]
    cv['second'] = (cv['t_start'] - cv['block_start']).round().astype(int)
    by_second = cv.groupby('second')['correct'].agg(['mean', 'size']).reset_index()
    by_round = cv.groupby('round')['correct'].agg(['mean', 'size']).reset_index()
    worst = (cv.groupby(['session', 'name', 'block', 'label'])
             .agg(accuracy=('correct', 'mean'), windows=('correct', 'size'),
                  predicted=('predicted', lambda p: p.value_counts().index[0]))
             .reset_index().sort_values(['accuracy', 'windows'], ascending=[True, False]).head(10))
    return by_second, by_round, worst


def clean(table):
    return table[~table['artifact'] & ~table['onset']]


def emotion_contrast(tables):
    """
    dB of each emotion against the Rest of the same session, per channel and band, averaged over sessions:
    10 x (mean log10 power of the emotion - mean log10 power at rest). Rest windows only (never the Cue).
    Returns one row per (label, channel, band) with mean, standard error and number of sessions.
    """
    names = feature_names()
    rows = []
    for session, table in tables.groupby('session'):
        ok = clean(table)
        rest = ok[ok['label'] == config.REST_LABEL][names]
        if not len(rest):
            continue
        base = rest.mean()
        for label in config.CLASSES:
            part = ok[ok['label'] == label][names]
            if len(part):
                rows.append({'session': session, 'label': label, **(10 * (part.mean() - base)).to_dict()})
    if not rows:
        return pd.DataFrame(columns=['label', 'channel', 'band', 'db', 'se', 'sessions'])
    wide = pd.DataFrame(rows)
    out = []
    for label, part in wide.groupby('label'):
        for name in names:
            channel, band = name.split('_', 1)
            values = part[name].to_numpy()
            out.append({'label': label, 'channel': channel, 'band': band, 'db': float(values.mean()),
                        'se': float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float('nan'),
                        'sessions': int(len(values))})
    return pd.DataFrame(out)


def fingerprint(tables, groups):
    """
    Spectral fingerprint: mean log band power of the clean emotion windows of each group (session, day or
    participant), in dB against the median group. groups maps session -> group name.
    """
    names = feature_names()
    ok = clean(tables)
    ok = ok[ok['label'].isin(config.CLASSES)].copy()
    if not len(ok):
        return pd.DataFrame(columns=names)
    ok['group'] = ok['session'].map(groups)
    means = ok.groupby('group', sort=False)[names].mean()
    return 10 * (means - means.median())


def rejection(tables, groups):
    """Per group: emotion windows, how many were rejected, and the channel behind most rejections."""
    emotion = tables[tables['label'].isin(config.CLASSES)].copy()
    emotion['group'] = emotion['session'].map(groups)
    rows = []
    for group, part in emotion.groupby('group', sort=False):
        bad = part[part['artifact']]
        over = (bad[[f'peak_{ch}' for ch in config.CHANNEL_NAMES]].to_numpy() > config.ARTIFACT_UV).sum(axis=0)
        worst = config.CHANNEL_NAMES[int(np.argmax(over))] if over.sum() else ''
        rows.append({'group': group, 'sessions': int(part['session'].nunique()), 'windows': int(len(part)),
                     'rejected': int(len(bad)), 'rejected_pct': 100 * len(bad) / max(len(part), 1),
                     'flat': int((bad['reason'] == 'flat').sum()), 'worst_channel': worst})
    return pd.DataFrame(rows, columns=['group', 'sessions', 'windows', 'rejected', 'rejected_pct', 'flat', 'worst_channel'])


def table_for(path):
    """Window table of a recording that may still be growing."""
    df, meta = load_complete(path)
    return df, meta, window_table(path, df=df, meta=meta)
