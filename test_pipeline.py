#!/usr/bin/env python3
"""
Offline checks of the pipeline, no Ganglion needed: python test_pipeline.py
Synthetic sessions are written in a temporary folder (emotion = change of alpha/beta power, plus
60 Hz mains and blinks) and go through the same code used with the real board. Each test names the
problem of the review it guards (A1-A17).
"""

import contextlib
import glob
import io
import json
import os
import random
import sys
import tempfile
import threading
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import config
from data_acquisition import DataAcquisition
from model_training import EmotionClassifier, permute_blocks, proba_rule_score
from preprocessing import Preprocessor, feature_names
from sessions import (SessionRecorder, load_session, load_sessions, predictions_path, protocol_meta,
                      protocol_steps, read_header, record_session, session_status, training_windows,
                      window_table)
from utils import Logger, RollingBuffer

FS = config.SAMPLING_RATE
REPO = os.path.dirname(os.path.abspath(__file__))
# Relative alpha / beta amplitude per emotion
PROFILE = {'Happy': (1.6, 0.8), 'Neutral': (1.0, 1.0), 'Frustrated': (0.6, 1.7)}


@contextlib.contextmanager
def overrides(**values):
    old = {key: getattr(config, key) for key in values}
    for key, value in values.items():
        setattr(config, key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(config, key, value)


@contextlib.contextmanager
def workdir():
    """A temporary folder as working directory, so data/, models/ and logs/ land there."""
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as folder:
        os.chdir(folder)
        try:
            yield folder
        finally:
            os.chdir(previous)


def fake_signal(seconds, label, rng, gain, informative=True):
    n = int(round(seconds * FS))
    t = np.arange(n) / FS
    alpha, beta = PROFILE.get(label, (1.0, 1.0)) if informative else (1.0, 1.0)
    pink = np.cumsum(rng.normal(0, 1, (n, config.CHANNELS)), axis=0)
    pink -= np.linspace(pink[0], pink[-1], n)
    x = 0.3 * pink
    x += alpha * 8 * np.sin(2 * np.pi * rng.uniform(9, 11) * t)[:, None]
    x += beta * 4 * np.sin(2 * np.pi * rng.uniform(18, 22) * t)[:, None]
    x += rng.normal(0, 3, (n, config.CHANNELS))
    x += 40 * np.sin(2 * np.pi * 60 * t)[:, None]  # mains, must be removed by the notch
    for start in rng.integers(0, max(n - 60, 1), size=int(seconds // 8)):
        x[start:start + 60, :2] += 250 * np.hanning(60)[:, None]  # blink on Fp1/Fp2
    return x * gain


def fake_session(folder, seed, informative=True, participant='P01', board='ganglion', status='completed',
                 stop_after=None, flat=()):
    """A whole protocol run written through SessionRecorder, with timestamps on its own clock."""
    random.seed(seed)
    rng = np.random.default_rng(seed)
    gain = rng.uniform(0.7, 1.3, config.CHANNELS)  # electrode impedance changes between sessions
    steps = protocol_steps()
    rec = SessionRecorder(FS, base_dir=folder, participant=participant, board=board, **protocol_meta(steps))
    t = 1.79e9 + 1000 * seed
    for block, step in enumerate(steps):
        if stop_after is not None and block >= stop_after:
            break
        x = fake_signal(step.seconds, step.label, rng, gain, informative)
        if step.label in flat:
            x[:] = 7.0
        ts = t + np.arange(len(x)) / FS
        rec.mark(step.label, step.round, block, t=t)
        rec.write(ts, x, package=np.arange(len(x)) % 256)
        t = ts[-1] + 1 / FS
    rec.mark_end(t)
    rec.close(status)
    return rec.path


def arrays(table, groups='session'):
    train = training_windows(table)
    return train[feature_names()].to_numpy(), train['label'].to_numpy(), train[groups].to_numpy(), train


def test_ganglion_channels():
    daq = DataAcquisition(board_id=1)
    assert daq.eeg_channels == [1, 2, 3, 4], daq.eeg_channels
    assert daq.package_channel == 0
    assert daq.sampling_rate == 200


def test_features_and_filters():
    """A8: amplitude and flat channels are both rejected, with their reason."""
    prep = Preprocessor(FS)
    raw = fake_signal(config.WINDOW_SIZE + config.FILTER_PAD, 'Neutral', np.random.default_rng(0), 1.0)
    features, _ = prep.preprocess_window(raw)
    assert features.shape == (len(feature_names()),) == (20,)

    t = np.arange(prep.window + prep.pad) / FS
    mains = 50 * np.sin(2 * np.pi * 60 * t)[:, None].repeat(config.CHANNELS, axis=1)
    assert np.abs(prep.clean_window(mains)).max() < 2.5, "60 Hz should be removed"

    clean = np.random.default_rng(1).normal(0, 10, (prep.window + prep.pad, config.CHANNELS))
    assert prep.analyze_window(clean)[1:3] == (False, '')
    spiky = clean.copy()
    spiky[-100:-60, 0] += 300 * np.hanning(40)
    assert prep.analyze_window(spiky)[1:3] == (True, 'amplitude'), "a 300 uV blink must flag the window"
    flat = clean.copy()
    flat[:, 2] = -187500.0  # a channel stuck at the rail
    _, artifact, reason, _ = prep.analyze_window(flat)
    assert (artifact, reason) == (True, 'flat')


def test_labels_follow_timestamps():
    """A6: a sample takes the phase of its timestamp, not of the moment it was read."""
    with workdir():
        rec = SessionRecorder(FS, base_dir='rec', participant='P01')
        rec.mark('Cue', 1, 0, t=10.0)
        rec.mark('Happy', 1, 1, t=12.0)
        first = np.array([9.0, 9.5, 10.0, 10.5, 11.5, 12.0, 12.5])
        assert rec.write(first, np.ones((len(first), 4)), package=np.arange(len(first))) == 5
        rec.mark_end(13.0)
        late = np.array([12.9, 13.0, 13.5])
        assert rec.write(late, np.ones((len(late), 4))) == 1
        rec.close('stopped')
        df, meta = load_session(rec.path)
        assert df['timestamp'].tolist() == [10.0, 10.5, 11.5, 12.0, 12.5, 12.9]
        assert df['label'].tolist() == ['Cue', 'Cue', 'Cue', 'Happy', 'Happy', 'Happy']
        assert df['package'].tolist()[-1] == -1, "no counter given -> -1"
        assert session_status(meta) == 'stopped' and meta['participant'] == 'P01'


def test_protocol_announces_next_emotion():
    """A6: every stimulus is announced by a Cue that names it."""
    steps = protocol_steps()
    stimuli = [s for s in steps if s.label in config.CLASSES]
    cues = [s for s in steps if s.label == config.CUE_LABEL]
    assert len(stimuli) == len(config.CLASSES) * config.TRAINING_ROUNDS
    assert [c.next for c in cues] == [s.label for s in stimuli]
    assert all(steps[i + 1].label == steps[i].next for i, s in enumerate(steps) if s.label == config.CUE_LABEL)
    assert steps[0].seconds >= config.FILTER_PAD


def test_recordings_left_out():
    """A2, A5, A16: what must not be trained on is skipped with its reason; nothing overwrites."""
    with workdir():
        good = fake_session('s', 1)
        stopped = fake_session('s', 2, status='stopped', stop_after=10)
        cut = fake_session('s', 3, status=None, stop_after=10)
        synthetic = fake_session('s', 4, board='synthetic')
        with open('s/session_20250101_120000.csv', 'w') as f:
            f.write('feature_0000,feature_0001,label\n0.1,0.2,Happy\n')
        open('s/empty.csv', 'w').close()
        paths = sorted(glob.glob('s/*.csv'))

        table, excluded = load_sessions(paths)
        reasons = {os.path.basename(e['file']): e['reason'] for e in excluded}
        assert reasons == {os.path.basename(stopped): 'stopped', os.path.basename(cut): 'incomplete',
                           os.path.basename(synthetic): 'synthetic', 'session_20250101_120000.csv': 'old_feature_format',
                           'empty.csv': 'empty'}, reasons
        assert set(table['session']) == {os.path.relpath(good)}

        table, excluded = load_sessions(paths, include_partial=True, allow_synthetic=True)
        assert len(table['session'].unique()) == 4 and len(excluded) == 2

        a = SessionRecorder(FS, 'x', participant='P01')
        b = SessionRecorder(FS, 'x', participant='P01')
        assert a.path != b.path, "two recordings in the same second must not overwrite each other"
        a.close('completed')
        b.close('completed')

        # A block split in two parts is skipped instead of stretching over other blocks
        with open('broken.csv', 'w') as f:
            f.write(f'# sampling_rate={FS}\n# board=ganglion\n')
            f.write('timestamp,package,Fp1,Fp2,T3,T4,label,round,block\n')
            rng = np.random.default_rng(0)
            for i, block in enumerate([0] * 1000 + [1] * 1000 + [0] * 500):
                label = 'Happy' if block == 0 else 'Rest'
                f.write(f'{i / FS},{i % 256},' + ','.join(f'{v:.3f}' for v in rng.normal(0, 10, 4)) + f',{label},1,{block}\n')
            f.write('# completed=x\n')
        with contextlib.redirect_stdout(io.StringIO()) as out:
            table = window_table('broken.csv')
        assert 'not contiguous' in out.getvalue() and set(table['block']) == {1}


def test_window_table_matches_live():
    """The single windowing code gives exactly what the live path computes."""
    with workdir():
        with overrides(ONSET_MARGIN=0.0):
            path = fake_session('s', 5)
            table = window_table(path)
            df, meta = load_session(path)
            raw = df[config.CHANNEL_NAMES].to_numpy(dtype=float)
            prep = Preprocessor(meta['sampling_rate'])
            for i in np.random.default_rng(0).choice(len(table), 30, replace=False):
                row = table.iloc[i]
                features, artifact = prep.preprocess_window(raw[max(0, row['end'] - prep.window - prep.pad):row['end']])
                assert np.allclose(features, row[feature_names()].to_numpy(dtype=float))
                assert artifact == row['artifact']
            assert not table['onset'].any()
        table = window_table(fake_session('s', 5))
        for _, block in table[table['label'].isin(config.CLASSES)].groupby('block'):
            assert block['onset'].tolist()[:3] == [True, True, False], "first 2 s of a stimulus stay out"
        assert set(table['label']) >= {config.REST_LABEL, config.CUE_LABEL, *config.CLASSES}


def test_training_is_honest():
    """A3: one decision rule everywhere and a confidence that does not oversell."""
    with workdir():
        real = [fake_session('real', s) for s in range(4)]
        noise = [fake_session('noise', s, informative=False) for s in range(4)]

        X, y, groups, _ = arrays(load_sessions(real)[0])
        clf = EmotionClassifier('svm')
        good = clf.evaluate(X, y, groups)
        print(f"  with emotion signal: balanced accuracy {good['balanced_accuracy']:.2f}")
        assert good['tuned'] and good['balanced_accuracy'] > 0.7
        assert (good['y_pred'] == np.array(clf.classes)[good['proba'].argmax(axis=1)]).all()

        Xn, yn, gn, _ = arrays(load_sessions(noise)[0])
        noisy = EmotionClassifier('svm')
        bad = noisy.evaluate(Xn, yn, gn)
        calibration = noisy.calibrate(yn, bad['proba'])
        print(f"  without emotion signal: balanced accuracy {bad['balanced_accuracy']:.2f} (chance 0.33), "
              f"confidence {calibration['mean_confidence_raw']:.2f} -> {calibration['mean_confidence_calibrated']:.2f}")
        assert bad['balanced_accuracy'] < 0.5
        assert calibration['mean_confidence_calibrated'] < 0.45

        clf.calibrate(y, good['proba'])
        clf.train(X, y, groups)
        model = clf.model
        live = [clf.predict(x)[0] for x in X]
        assert np.isclose(proba_rule_score(model, X, y), balanced_accuracy_score(y, live)), "tuning scores the live rule"
        for x in X[::37]:
            label, confidence, probabilities = clf.predict(x)
            assert label == clf.classes[int(np.argmax(probabilities))] and confidence == probabilities.max()
            assert np.isclose(probabilities.sum(), 1)
            raw = clf._proba(model, x.reshape(1, -1))[0]
            assert int(np.argmax(raw)) == int(np.argmax(probabilities)), "temperature never changes the class"


def test_groups_and_chance():
    """A4, A10: chance per fold, and the same tuning rule when evaluating and in the final model."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 20))
    groups = np.repeat(['a', 'b', 'c', 'd'], 30)
    y = np.tile(np.array(config.CLASSES), 40)
    y[(groups == 'd') & (y == 'Frustrated')] = 'Happy'
    clf = EmotionClassifier()
    result = clf.evaluate(X, y, groups)
    fold_d = [f for f in result['folds'] if f['test_groups'] == ['d']][0]
    assert np.isclose(fold_d['chance'], 0.5)
    assert all(np.isclose(f['chance'], 1 / 3) for f in result['folds'] if f['test_groups'] != ['d'])

    try:
        clf.evaluate(X, y, np.zeros(len(y)))
        raise AssertionError("one group must be refused")
    except ValueError:
        pass

    two = np.where(np.isin(groups, ['a', 'b']), 'x', 'y')
    assert not clf.evaluate(X, y, two)['tuned'] and clf.train(X, y, two) == {}
    assert clf.evaluate(X, y, groups)['tuned']


def test_permutations():
    """A4: block-wise permutations keep the design and give an honest p-value."""
    sessions = np.repeat(['s1', 's2'], 60)
    blocks = np.tile(np.repeat(np.arange(6), 10), 2)
    y = np.array(config.CLASSES * 2)[blocks]
    shuffled = permute_blocks(y, sessions, blocks, np.random.default_rng(1))
    for s in ['s1', 's2']:
        for b in range(6):
            assert len(set(shuffled[(sessions == s) & (blocks == b)])) == 1, "a block keeps one label"
        assert Counter(shuffled[sessions == s]) == Counter(y[sessions == s])

    with workdir(), overrides(SAMPLING_DURATION=6.0, REST_DURATION=3.0, CUE_SECONDS=1.0):
        for informative in (True, False):
            paths = [fake_session(f's{informative}', s, informative=informative) for s in range(3)]
            X, y, groups, train = arrays(load_sessions(paths)[0])
            clf = EmotionClassifier()
            observed = clf.evaluate(X, y, groups)['balanced_accuracy']
            with contextlib.redirect_stdout(io.StringIO()):
                null, p = clf.permutation_test(X, y, groups, train['session'].to_numpy(), train['block'].to_numpy(),
                                               observed, 19, tune=False, n_jobs=2)
            print(f"  {'signal' if informative else 'noise '}: observed {observed:.2f}, p = {p:.2f}")
            assert len(null) == 19
            assert (p == 0.05) if informative else (p > 0.05)


def train_command(*extra):
    import main
    sys.argv = ['main.py', '--mode', 'train', *extra]
    with contextlib.redirect_stdout(io.StringIO()) as out:
        try:
            main.handle_training(main.parse_arguments())
            code = 0
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


def test_train_command():
    """A2, A9, A10, A11, A12: what train keeps, the group it uses, what it saves and what it refuses."""
    with workdir():
        for seed, participant in enumerate(['P01', 'P02', 'P03']):
            fake_session(config.SESSIONS_DIR, seed, participant=participant)
        fake_session(config.SESSIONS_DIR, 9, board='synthetic', participant='P09')

        code, out = train_command('--min-sessions', '4', '--model', 'models/m.pkl')
        assert code == 1 and 'usable sessions' in out, out
        code, out = train_command('--min-sessions', '3', '--model', 'models/m.pkl')
        assert code == 0, out
        assert 'No permutation test' in out
        code, out = train_command('--min-sessions', '3', '--model', 'models/p.pkl', '--permutations', '19')
        assert code == 0 and 'p = 0.050 with 19 permutations: above chance' in out, out
        assert json.load(open('models/p.report.json'))['permutations']['n'] == 19

        report = json.load(open('models/m.report.json'))
        for key in ['schema', 'model', 'classes', 'validation', 'data', 'metrics', 'folds', 'per_class',
                    'confusion_matrix', 'calibration', 'permutations', 'config', 'environment']:
            assert key in report, key
        assert report['validation']['group_unit'] == 'participant'
        assert [e['reason'] for e in report['data']['excluded']] == ['synthetic']
        assert np.array(report['confusion_matrix']['counts']).sum() == report['data']['n_windows']
        assert os.path.exists('models/m.confusion_matrix.png')

        cv = pd.read_csv('models/m.cv.csv')
        assert len(cv) == report['data']['n_windows']
        probabilities = cv[[f'p_{c}' for c in config.CLASSES]].to_numpy()
        assert np.allclose(probabilities.sum(axis=1), 1)
        assert (cv['predicted'] == np.array(config.CLASSES)[probabilities.argmax(axis=1)]).all()
        windows = pd.concat([window_table(p) for p in cv['session'].unique()])
        assert len(cv.merge(windows, on=['session', 'start'])) == len(cv)

        with overrides(BANDPASS_LOW=1.0):
            try:
                EmotionClassifier().load_model('models/m.pkl')
                raise AssertionError("a model trained with other filters must be refused")
            except ValueError as e:
                assert 'bandpass' in str(e)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            EmotionClassifier().load_model('models/m.pkl', sampling_rate=250)
        assert 'Warning' in out.getvalue()

        with overrides(TRAINING_ROUNDS=1):
            single = fake_session('one', 7)
        with overrides(MIN_CLASS_WINDOWS=5):
            code, out = train_command('--csv', single, '--min-sessions', '1', '--model', 'models/x.pkl')
        assert code == 1 and 'Only one group' in out and 'Traceback' not in out, out

    sys.argv = ['main.py', '--mode', 'record', '--participant', 'Diego Burbano']
    import main
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            main.parse_arguments()
            raise AssertionError("a name is not a participant code")
        except SystemExit as e:
            assert e.code == 2


def test_record_warns_when_samples_stop():
    """A1: a silent board is reported block by block and the session still closes cleanly."""
    class SilentBoard:
        sampling_rate = FS

        def read_new(self):
            return np.array([]), np.array([]), np.zeros((0, config.CHANNELS))

    with workdir(), overrides(SAMPLING_DURATION=0.3, REST_DURATION=0.3, CUE_SECONDS=0.1, TRAINING_ROUNDS=1,
                              FILTER_PAD=0.1):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            path = record_session(SilentBoard(), participant='P01')
        assert out.getvalue().count('WARNING') == len(protocol_steps())
        assert session_status(read_header(path)[0]) == 'completed'


def test_noise_recording_with_synthetic_board():
    """Noise library: a timed recording with its code and note in the header and no protocol."""
    from sessions import record_noise

    with workdir():
        daq = DataAcquisition(board_id=-1)
        with contextlib.redirect_stdout(io.StringIO()):
            daq.initialize_board()
            daq.start_streaming()
            try:
                path = record_noise(daq, 'N3', seconds=2.0, note='10 kOhm\n between + and -', synthetic=True)
                try:
                    record_noise(daq, 'X9', seconds=0.1)
                    raise AssertionError("an unknown code is refused")
                except ValueError:
                    pass
            finally:
                daq.stop_streaming()
                daq.close()
        meta, _ = read_header(path)
        assert path.startswith(os.path.join(config.NOISE_DIR, 'synthetic', 'noise_N3_'))
        assert meta['protocol'] == 'noise' and meta['code'] == 'N3' and meta['note'] == '10 kOhm between + and -'
        assert session_status(meta) == 'completed'
        df = pd.read_csv(path, comment='#')
        expected = 2.0 * daq.sampling_rate  # the synthetic board runs at 250 Hz
        assert set(df['label']) == {'noise'} and 0.75 * expected < len(df) <= 1.1 * expected, len(df)


def test_realtime_with_synthetic_board():
    """A1, A11: realtime records its signal and predictions, and never classifies a frozen buffer."""
    from real_time_classification import RealTimeClassifier

    with workdir():
        paths = [fake_session('s', seed) for seed in range(3)]
        X, y, groups, _ = arrays(load_sessions(paths)[0])
        clf = EmotionClassifier()
        clf.train(X, y, groups)
        clf.save_model('models/m.pkl', sampling_rate=FS)

        rtc = RealTimeClassifier('models/m.pkl', board_id=-1, synthetic=True)
        with contextlib.redirect_stdout(io.StringIO()):
            thread = threading.Thread(target=rtc.start_classification)
            thread.start()
            time.sleep(8)
            rtc.is_running = False
            thread.join(15)
        live = glob.glob(os.path.join(config.SYNTHETIC_DIR, 'live_*.csv'))
        assert len(live) == 1
        meta, _ = read_header(live[0])
        df, _ = load_session(live[0])
        assert session_status(meta) == 'completed' and meta['board'] == 'synthetic'
        assert len(df) > 5 * 250 and set(df['label']) == {config.LIVE_LABEL}
        predictions = pd.read_csv(predictions_path(live[0]), comment='#')
        assert 'ok' in set(predictions['status']), predictions['status'].value_counts()

        prep = Preprocessor(FS)
        buffer = RollingBuffer(prep.window + prep.pad, config.CHANNELS)
        buffer.push(np.arange(1000) / FS, np.random.default_rng(0).normal(0, 10, (1000, config.CHANNELS)))
        buffer.last_arrival -= 5
        rtc.classifier.predict = lambda features: (_ for _ in ()).throw(AssertionError("frozen buffer classified"))
        logger = Logger('logs/frozen.csv', config.CLASSES)
        with contextlib.redirect_stdout(io.StringIO()):
            rtc.process_window(prep, buffer, logger)
        assert pd.read_csv('logs/frozen.csv', comment='#')['status'].tolist() == ['no_data']


def test_lost_packets_and_stalls():
    """A7: the package counter tells how many samples were lost; long pauses show as stalls."""
    import analysis

    counter = np.arange(1000) % 256
    assert analysis.lost_packets(counter, 'synthetic') == (0, 1000)
    assert analysis.lost_packets(np.delete(counter, np.arange(300, 310)), 'synthetic') == (10, 1000)
    ids = np.repeat(np.arange(300) % 100 + 101, 2)  # Ganglion: two samples per packet, ids 101-200
    assert analysis.lost_packets(ids, 'ganglion') == (0, 600)
    assert analysis.lost_packets(np.delete(ids, np.arange(40, 46)), 'ganglion') == (6, 600)
    assert analysis.lost_packets(np.full(10, -1), 'synthetic') is None


def test_lost_packets_follow_the_ganglion_firmware():
    """Firmware 2 numbers packets 1-100 / 101-200, firmware 3 numbers them 0-99 / 100-199 (BrainFlow)."""
    import analysis

    fw2 = np.repeat(np.arange(350) % 100 + 1, 2)  # 18-bit packets of firmware 2, wrapping 100 -> 1
    fw3 = np.repeat(np.arange(350) % 100, 2)      # 18-bit packets of firmware 3, wrapping 99 -> 0
    fw3_19 = np.repeat(np.arange(350) % 100 + 100, 2)
    assert analysis.ganglion_firmware(fw2) == 2
    assert analysis.ganglion_firmware(fw3) == 3 and analysis.ganglion_firmware(fw3_19) == 3
    assert analysis.ganglion_firmware(fw2[:100]) is None, "less than one cycle cannot tell"
    for ids, firmware in [(fw2, 2), (fw3, 3), (fw3_19, 3)]:
        across = np.delete(ids, np.arange(196, 206))  # 5 packets lost across the wrap of the counter
        before = np.delete(ids, np.arange(196, 200))  # the 2 packets just before the wrap
        assert analysis.lost_packets(ids, 'ganglion') == (0, 700)
        assert analysis.lost_packets(across, 'ganglion') == (10, 700)
        assert analysis.lost_packets(across, 'ganglion', firmware) == (10, 700)
        assert analysis.lost_packets(before, 'ganglion', firmware) == (4, 700)
    # Read as firmware 2, a firmware 3 recording takes id 0 for a reset and misses the loss before it
    assert analysis.lost_packets(np.delete(fw3, np.arange(196, 200)), 'ganglion', 2) == (0, 696)
    assert analysis.lost_packets(fw2[:100], 'ganglion') is None, "unknown firmware: no count"
    reset = np.r_[fw2[:61], 0, fw2[61:]]  # firmware 2 sends id 0 (one sample) after a reset
    assert analysis.lost_packets(reset, 'ganglion') == (0, 701)

    ts = np.r_[np.arange(400) / FS, 2.5 + np.arange(400) / FS]
    timing = analysis.timing(pd.DataFrame({'timestamp': ts, 'package': np.arange(800) % 256}), FS, 'synthetic')
    stall = timing['stalls'][0]
    assert len(timing['stalls']) == 1 and abs(stall['seconds'] - 0.505) < 0.01
    assert abs(stall['at'] - 1.995) < 0.01 and abs(stall['signal_at'] - 2.0) < 0.01
    assert abs(timing['seconds'] - 4.5) < 0.01 and abs(timing['signal_seconds'] - 4.0) < 0.01, "duration by the clock"


def test_buffer_refills_after_silence():
    """A1: after a cut the live window waits for fresh samples instead of mixing old and new signal."""
    buffer = RollingBuffer(100, config.CHANNELS, reset_after=config.NO_DATA_SECONDS)
    buffer.push(np.arange(150) / FS, np.ones((150, config.CHANNELS)))
    assert buffer.full
    buffer.last_arrival -= 5
    buffer.push(np.arange(10) / FS, np.ones((10, config.CHANNELS)))
    assert not buffer.full and buffer.received == 10
    buffer.push(np.arange(90) / FS, np.ones((90, config.CHANNELS)))
    assert buffer.full


def test_read_tail_while_writing():
    """The dashboard reads a recording while the recorder keeps appending to it."""
    import analysis

    with workdir():
        rec = SessionRecorder(FS, 'live', prefix='live', protocol='live')
        rec.mark(config.LIVE_LABEL, 0, 0, t=0.0)
        stop = threading.Event()

        def writer():
            t = 0.0
            while not stop.is_set():
                ts = t + np.arange(10) / FS
                rec.write(ts, np.ones((10, config.CHANNELS)), package=np.arange(10))
                t = ts[-1] + 1 / FS
                time.sleep(0.005)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            rows = []
            for _ in range(30):
                df, _ = analysis.read_tail(rec.path, 2)
                assert list(df.columns) == ['timestamp', 'package', *config.CHANNEL_NAMES, 'label', 'round', 'block']
                assert not df.isna().any().any() and (df['label'] == config.LIVE_LABEL).all()
                rows.append(len(df))
                time.sleep(0.01)
        finally:
            stop.set()
            thread.join()
        rec.close('completed')
        assert max(rows) > 0
        df, _ = analysis.read_tail(rec.path, 1)
        assert abs(df['timestamp'].iat[-1] - df['timestamp'].iat[0] - 1) < 0.02


def test_gui_offscreen():
    """A1, A14, A15: participant code, labels, stop, close, unload and a silent board in the real GUI."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from pyqtgraph.Qt import QtCore, QtWidgets
    import gui_app

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def quiet(mode, context, message):  # the offscreen plugin complains about window hints
        if 'propagateSizeHints' not in message:
            sys.stderr.write(message + '\n')
    QtCore.qInstallMessageHandler(quiet)

    def run_events(seconds):
        QtCore.QTimer.singleShot(int(seconds * 1000), app.quit)
        app.exec_()

    with workdir():
        paths = [fake_session('s', seed) for seed in range(3)]
        X, y, groups, _ = arrays(load_sessions(paths)[0])
        clf = EmotionClassifier()
        clf.train(X, y, groups)
        clf.save_model('models/m.pkl', sampling_rate=FS)

        daq = DataAcquisition(board_id=-1)
        with contextlib.redirect_stdout(io.StringIO()), \
                overrides(SAMPLING_DURATION=3.0, REST_DURATION=2.0, CUE_SECONDS=1.0, TRAINING_ROUNDS=1):
            daq.initialize_board()
            daq.start_streaming()
            try:
                win = gui_app.EEGGui(daq, 'models/m.pkl')
                win.show()  # closeAllWindows only closes visible windows, as in run_gui
                assert win.classifier_loaded, "--model is loaded when the window opens"
                assert not win.btn_start.isEnabled()
                win.txt_participant.setText('Diego Burbano')
                assert not win.btn_start.isEnabled()
                win.txt_participant.setText('P01')
                assert win.btn_start.isEnabled()
                win.chk_live.setChecked(True)

                total = sum(step.seconds for step in protocol_steps())
                win.btn_start.click()
                run_events(total + 2)
                assert not win.protocol.active
                first = glob.glob(os.path.join(config.SESSIONS_DIR, 'session_P01_*.csv'))
                assert len(first) == 1
                df, meta = load_session(first[0])
                assert session_status(meta) == 'completed'
                labels = df.groupby('block', sort=False)['label'].first().tolist()
                assert labels[0] == config.CUE_LABEL and sum(l in config.CLASSES for l in labels) == 3
                last = df[df['block'] == df['block'].max()]
                assert len(last) >= 0.9 * config.SAMPLING_DURATION * daq.sampling_rate, "tail of the last block"
                predictions = pd.read_csv(predictions_path(first[0]), comment='#')
                assert len(predictions) and set(predictions['label']) <= {config.CUE_LABEL, config.REST_LABEL, *config.CLASSES}

                # A write error stops the recording; the next session gets its own predictions file
                win.btn_start.click()
                run_events(1.5)
                win.recording_failed(OSError(28, 'No space left on device'))
                assert win.recorder is None and 'Recording failed' in win.lbl_status.text()
                win.btn_start.click()
                run_events(2)
                newest = max(glob.glob(os.path.join(config.SESSIONS_DIR, '*.csv')), key=os.path.getmtime)
                assert os.path.exists(predictions_path(newest)), "predictions of the new session in their own file"
                win.btn_stop.click()
                win.on_unload_model()
                assert win.lbl_predict.text() == "Prediction: - (conf -)"

                win.btn_start.click()
                run_events(1.5)
                daq.stop_streaming()
                run_events(1.6)
                assert 'NO DATA' in win.lbl_signal.text(), win.lbl_signal.text()
                app.closeAllWindows()  # what Ctrl+C does
                sessions = sorted(glob.glob(os.path.join(config.SESSIONS_DIR, '*.csv')), key=os.path.getmtime)
                statuses = [session_status(read_header(p)[0]) for p in sessions]
                assert statuses == ['completed', 'stopped', 'stopped', 'stopped'], statuses
            finally:
                daq.stop_streaming()
                daq.close()


if __name__ == '__main__':
    os.chdir(REPO)
    only = sys.argv[1:]
    for name, fn in list(globals().items()):
        if name.startswith('test_') and (not only or name in only):
            print(name, flush=True)
            fn()
    print("All tests passed")
