#!/usr/bin/env python3
"""
EEG Emotion Classification System
Main entry point with command-line interface
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
import webbrowser

import numpy as np

import config

REPO = os.path.dirname(os.path.abspath(__file__))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='EEG Emotion Classification System')

    parser.add_argument('--mode', choices=['gui', 'record', 'noise', 'train', 'realtime', 'dashboard'], required=True,
                        help='gui: protocol with plots | record: protocol in the terminal | '
                             'train: model from recorded sessions | realtime: live classification | '
                             'dashboard: local web page with the sessions, the model and the live signal')
    parser.add_argument('--port', default=config.SERIAL_PORT,
                        help=f'Serial port for OpenBCI board (default: {config.SERIAL_PORT})')
    parser.add_argument('--synthetic', action='store_true',
                        help=f'Use the BrainFlow synthetic board instead of the Ganglion; its recordings '
                             f'go to {config.SYNTHETIC_DIR} and never mix with real ones')
    parser.add_argument('--participant',
                        help='Participant code such as P03 (never a name); required to record real sessions')
    parser.add_argument('--model', default=config.MODEL_PATH,
                        help=f'Path to model file (default: {config.MODEL_PATH})')
    parser.add_argument('--model-type', choices=['svm', 'rf'], default='svm',
                        help='Model type for training (default: svm)')
    parser.add_argument('--csv', help='Train from this single session CSV instead of --csv-dir')
    parser.add_argument('--csv-dir', default=config.SESSIONS_DIR,
                        help=f'Directory with the session CSVs (default: {config.SESSIONS_DIR})')
    parser.add_argument('--min-sessions', '--min-csv', dest='min_sessions', type=int, default=20,
                        help='Minimum number of usable sessions (complete, with windows of every emotion) '
                             'required to train (default: 20)')
    parser.add_argument('--include-partial', action='store_true',
                        help='Also train on sessions that were stopped or cut before the end')
    parser.add_argument('--permutations', type=int, default=0,
                        help='Permutation test against chance, e.g. 100 (repeats the whole evaluation N times)')
    parser.add_argument('--web-port', type=int, default=8501, help='Port of the dashboard (default: 8501)')
    parser.add_argument('--code', help='noise: recording code of the noise protocol (N1, N2, N3, C1, P1-P4)')
    parser.add_argument('--seconds', type=float, default=300.0, help='noise: length of the recording (default: 300)')
    parser.add_argument('--note', default='', help='noise: what was connected (resistor, distance, tone...)')

    args = parser.parse_args()
    if args.participant and not re.match(config.PARTICIPANT_PATTERN, args.participant):
        parser.error('--participant must be a code like P03 (letters, digits, - or _, up to 16 characters), '
                     'never a name')
    return args


def board_id(args):
    from brainflow.board_shim import BoardIds
    return BoardIds.SYNTHETIC_BOARD.value if args.synthetic else config.BOARD_ID


def handle_training(args):
    """Evaluate with grouped cross-validation, then fit the final model on everything."""
    from model_training import EmotionClassifier, build_report, cv_table, report_paths, save_evaluation
    from preprocessing import feature_names
    from sessions import load_sessions, read_header, training_windows

    # Synthetic sessions only enter when their folder or file is given on purpose
    explicit = bool(args.csv) or os.path.normpath(args.csv_dir) != os.path.normpath(config.SESSIONS_DIR)
    paths = [args.csv] if args.csv else sorted(glob.glob(os.path.join(args.csv_dir, '*.csv')))
    print(f"Found {len(paths)} recordings in {args.csv or args.csv_dir}")

    table, excluded = load_sessions(paths, include_partial=args.include_partial, allow_synthetic=explicit)
    train = training_windows(table)
    if train.empty:
        print("No usable windows in the sessions.")
        sys.exit(1)

    labels_per_session = train.groupby('session')['label'].agg(set)
    usable = [s for s, labels in labels_per_session.items() if set(config.CLASSES) <= labels]
    if len(usable) < args.min_sessions:
        print(f"Need at least {args.min_sessions} usable sessions (complete, with windows of every emotion); "
              f"found {len(usable)}. Use --min-sessions to change it.")
        sys.exit(1)

    counts = train['label'].value_counts()
    short = [c for c in config.CLASSES if counts.get(c, 0) < config.MIN_CLASS_WINDOWS]
    if short:
        print(f"Too few clean windows of {', '.join(short)} (minimum {config.MIN_CLASS_WINDOWS}): "
              f"{ {c: int(counts.get(c, 0)) for c in config.CLASSES} }")
        sys.exit(1)

    # Windows of the same participant, session or round are correlated, so a group never goes to
    # both sides of a split. The most demanding unit available is used.
    participants = train['participant'].fillna('').astype(str)
    if (participants != '').all() and participants.nunique() >= 3:
        unit, groups = 'participant', participants.to_numpy()
    elif train['session'].nunique() >= 3:
        unit, groups = 'session', train['session'].to_numpy()
    else:
        unit, groups = 'round', (train['session'] + '#' + train['round'].astype(str)).to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        print("Only one group of windows: record at least 2 sessions (or one session with 2 complete rounds).")
        sys.exit(1)

    X = train[feature_names()].to_numpy()
    y = train['label'].to_numpy()
    print(f"\n{len(y)} windows | { {c: int(counts.get(c, 0)) for c in config.CLASSES} } | "
          f"test groups: {unit} ({n_groups})")

    classifier = EmotionClassifier(args.model_type)
    results = classifier.evaluate(X, y, groups, n_jobs=-1)
    calibration = classifier.calibrate(y, results['proba'])
    print("\nGrouped cross-validation (rule: argmax of the probabilities, the same as live):")
    print(f"Accuracy:          {results['accuracy']:.2f}")
    print(f"Balanced accuracy: {results['balanced_accuracy']:.2f}  (chance = {results['chance']:.2f})")
    print(f"Macro F1:          {results['f1_macro']:.2f}")
    print(f"Per fold:          {np.round(results['fold_scores'], 2)}")
    print(f"Hyperparameters:   {'tuned inside each fold' if results['tuned'] else 'fixed (fewer than 4 groups)'}")
    print(f"Confidence:        {calibration['mean_confidence_raw']:.2f} on average, "
          f"{calibration['mean_confidence_calibrated']:.2f} after calibration (T = {calibration['temperature']:.2f}); "
          f"accuracy {calibration['accuracy']:.2f}")
    print("\n" + results['report'])

    permutation = None
    if args.permutations > 0:
        print(f"Permutation test, {args.permutations} runs of the whole evaluation:")
        null, p_value = classifier.permutation_test(
            X, y, groups, train['session'].to_numpy(), train['block'].to_numpy(),
            results['balanced_accuracy'], args.permutations, tune=results['tuned'])
        if p_value is not None:
            permutation = {'n': int(len(null)), 'p_value': p_value, 'null': null.tolist(),
                           'null_p95': float(np.percentile(null, 95))}
            verdict = 'above chance' if p_value <= 0.05 else 'not distinguishable from chance'
            print(f"p = {p_value:.3f} with {len(null)} permutations: {verdict} "
                  f"(95% of the permutations stay below {permutation['null_p95']:.2f})")
    else:
        print("No permutation test: this alone does not show that the model beats chance "
              "(use --permutations 100).")
    if unit == 'round':
        print("Groups are rounds of the same session(s): this measures recognition inside a session already "
              "seen, not on new people or days.")

    _, _, cm_path = report_paths(args.model)
    classifier.plot_confusion_matrix(results['confusion_matrix'], cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")

    params = classifier.train(X, y, groups, tune=results['tuned'], n_jobs=-1)
    if params:
        print(f"Final model hyperparameters: {params}")

    sessions = []
    for session, part in train.groupby('session', sort=False):
        whole = table[(table['session'] == session) & table['label'].isin(config.CLASSES)]
        meta, _ = read_header(session)
        sessions.append({'file': session, 'name': part['name'].iat[0], 'participant': part['participant'].iat[0],
                         'board': meta.get('board', ''), 'sampling_rate': float(meta.get('sampling_rate', 0)),
                         'windows': int(len(part)), 'emotion_windows': int(len(whole)),
                         'rejected': int(whole['artifact'].sum()), 'onset': int(whole['onset'].sum())})
    rates = [s['sampling_rate'] for s in sessions if s['sampling_rate']]
    sampling_rate = max(set(rates), key=rates.count) if rates else None
    synthetic = any(s['board'] == 'synthetic' for s in sessions)

    classifier.save_model(args.model, sampling_rate=sampling_rate)
    report = build_report(args.model, classifier, results, calibration, permutation, params, train, sessions,
                          excluded, unit, n_groups, synthetic, sampling_rate)
    json_path, cv_path = save_evaluation(args.model, report, cv_table(train, results, classifier.classes))
    print(f"Evaluation saved to {json_path} and {cv_path}")


def handle_record(args):
    from data_acquisition import DataAcquisition
    from sessions import record_session

    if not args.synthetic and not args.participant:
        print("Real sessions need a participant code: --participant P03")
        sys.exit(1)
    with DataAcquisition(args.port, board_id(args)) as daq:
        daq.start_streaming()
        record_session(daq, participant=args.participant, synthetic=args.synthetic)


def handle_noise(args):
    """A timed recording with nobody connected (noise library of Ganglion-4 Bench)."""
    from data_acquisition import DataAcquisition
    from sessions import NOISE_CODES, record_noise

    if args.code not in NOISE_CODES:
        print(f"Noise recordings need a code from the protocol: --code {' | '.join(NOISE_CODES)}")
        sys.exit(1)
    with DataAcquisition(args.port, board_id(args)) as daq:
        daq.start_streaming()
        record_noise(daq, args.code, args.seconds, args.note, synthetic=args.synthetic)


def handle_realtime(args):
    """Handle real-time classification mode."""
    from real_time_classification import run_real_time_classification

    try:
        print("Starting real-time emotion classification...")
        run_real_time_classification(args.model, args.port, board_id(args),
                                     participant=args.participant, synthetic=args.synthetic)
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        print("Please train a model first using --mode train")
        sys.exit(1)
    except ValueError as e:
        print(f"Real-time classification failed: {e}")
        sys.exit(1)


def handle_gui(args):
    """Handle GUI mode."""
    from gui_app import run_gui

    print("Launching GUI for EEG acquisition, FFT, and protocol-driven recording...")
    run_gui(args.port, board_id(args), args.model, participant=args.participant, synthetic=args.synthetic)


def handle_dashboard(args):
    """Start the Streamlit dashboard on this computer only and open it in the browser."""
    import importlib.util
    import urllib.request

    if importlib.util.find_spec('streamlit') is None:
        print("The dashboard needs streamlit: pip install -r requirements.txt")
        sys.exit(1)
    url = f"http://127.0.0.1:{args.web_port}"
    # The privacy options go on the command line too, so they hold from any working directory
    command = [sys.executable, '-m', 'streamlit', 'run', os.path.join(REPO, 'dashboard.py'),
               '--server.address', '127.0.0.1', '--server.port', str(args.web_port),
               '--server.headless', 'true', '--browser.gatherUsageStats', 'false',
               '--client.toolbarMode', 'minimal', '--server.fileWatcherType', 'none']
    process = subprocess.Popen(command, cwd=REPO)
    try:
        for _ in range(120):
            if process.poll() is not None:
                sys.exit(process.returncode)
            try:
                urllib.request.urlopen(url + '/_stcore/health', timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        print(f"\nDashboard at {url} (Ctrl+C to stop)")
        webbrowser.open(url)
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()


def main():
    """Main function."""
    args = parse_arguments()

    print("EEG Emotion Classification System")
    print("=" * 40)

    handlers = {'train': handle_training, 'record': handle_record, 'noise': handle_noise, 'realtime': handle_realtime,
                'gui': handle_gui, 'dashboard': handle_dashboard}
    try:
        handlers[args.mode](args)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()
