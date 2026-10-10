#!/usr/bin/env python3
"""
Dashboard checks without a browser: python test_dashboard.py
Every page runs through Streamlit's AppTest with recordings, with empty folders and with a recording in
progress; the HTML export and the promise of never touching the headset are checked too.
"""

import contextlib
import glob
import io
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np
from plotly.offline import get_plotlyjs
from streamlit.testing.v1 import AppTest

import config
import main
from sessions import SessionRecorder, predictions_path
from test_pipeline import fake_session, workdir
from utils import Logger

REPO = os.path.dirname(os.path.abspath(__file__))


def run_page(page, folder, clicks=()):
    """Runs one page function in its own app, from the given working folder. clicks: button labels to press."""
    script = (f"import os, sys\nsys.path.insert(0, {REPO!r})\nos.chdir({folder!r})\n"
              f"import dashboard\ndashboard.{page}()\n")
    at = AppTest.from_string(script, default_timeout=120)
    at.run()
    for label in clicks:
        [button for button in at.button if button.label == label][0].click().run()
    assert not at.exception, at.exception
    return at


def text_of(at):
    return ' '.join(m.value for m in at.markdown)


def charts(at):
    return len(at.get('plotly_chart'))


def make_recordings():
    """Three real sessions, one synthetic, a trained model and GUI predictions for one session."""
    paths = [fake_session(config.SESSIONS_DIR, seed, participant=f'P0{seed + 1}') for seed in range(3)]
    fake_session(config.SYNTHETIC_DIR, 7, board='synthetic', participant=None)
    sys.argv = ['main.py', '--mode', 'train', '--min-sessions', '3']
    with contextlib.redirect_stdout(io.StringIO()):
        main.handle_training(main.parse_arguments())
    logger = Logger(predictions_path(paths[0]), config.CLASSES, config.MODEL_PATH)
    t0 = 1.79e9 + 10
    for i in range(30):
        probabilities = np.roll([0.6, 0.3, 0.1], i % 3)
        logger.log(t0 + i, 'ok', config.CLASSES[int(np.argmax(probabilities))], probabilities.max(), probabilities,
                   label='Happy', block=1)
    return paths


def live_recording(seconds=8, fs=200):
    """A realtime run whose file was just written, with its predictions."""
    rec = SessionRecorder(fs, config.LIVE_DIR, prefix='live', participant='P01', protocol='live')
    now = time.time()
    rec.mark(config.LIVE_LABEL, 0, 0, t=now - seconds)
    ts = now - seconds + np.arange(int(seconds * fs)) / fs
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 10, (len(ts), config.CHANNELS)) + 8 * np.sin(2 * np.pi * 10 * ts)[:, None]
    rec.write(ts, signal, package=np.arange(len(ts)) % 256)
    logger = Logger(predictions_path(rec.path), config.CLASSES, config.MODEL_PATH)
    for i in range(seconds):
        logger.log(now - seconds + i, 'ok' if i % 4 else 'artifact', 'Neutral', 0.55, [0.25, 0.55, 0.2], label='Live', block=0)
    return rec


def test_never_opens_the_board():
    out = subprocess.run([sys.executable, '-B', '-c', 'import sys; import dashboard; print("brainflow" in sys.modules)'],
                         cwd=REPO, capture_output=True, text=True)
    assert out.stdout.strip() == 'False', out.stdout + out.stderr


def test_empty_folders():
    with workdir() as folder:
        sessions = run_page('page_sessions', folder)
        assert 'Todavía no hay grabaciones' in text_of(sessions) and charts(sessions) == 0
        live = run_page('page_live', folder)
        assert 'Ahora no se está grabando nada' in text_of(live) and 'python main.py --mode gui' in text_of(live)
        model = run_page('page_model', folder)
        assert 'Todavía no hay un modelo evaluado' in text_of(model) and charts(model) == 0
        compare = run_page('page_compare', folder)
        assert 'hacen falta al menos dos sesiones' in text_of(compare) and charts(compare) == 0


def test_pages_with_recordings():
    with workdir() as folder:
        paths = make_recordings()

        sessions = run_page('page_sessions', folder, clicks=['Preparar descarga'])
        text = text_of(sessions)
        assert 'ventanas de emoción' in text and 'muestras' in text, text
        assert charts(sessions) >= 3, charts(sessions)
        exported = [v for k, v in sessions.session_state.filtered_state.items() if isinstance(v, bytes)]
        assert len(exported) == 1
        page = exported[0].decode('utf-8')
        plotly_js = get_plotlyjs()
        assert page.count(plotly_js[:500]) == 1, "plotly.js inlined exactly once"
        rest = page.replace(plotly_js, '')
        assert not re.search(r'<script[^>]+src=', rest) and '#000001' not in rest, "no external script, no Streamlit template"
        assert any(color in rest for color in ('#eb6834', '#2a78d6', '#4a3aa7')), "emotion colors in the figures"
        assert len(page) < 12e6

        model = run_page('page_model', folder, clicks=['Preparar descarga'])
        text = text_of(model)
        assert 'exactitud balanceada' in text and 'participantes completos' in text, text
        assert 'No se hizo la prueba de permutación' in text
        assert charts(model) >= 3

        compare = run_page('page_compare', folder)
        assert 'Hay 3 sesiones de 3 participantes' in text_of(compare), text_of(compare)
        assert charts(compare) >= 1

        # A session deleted after training: the model page still works and keeps its label
        os.remove(paths[1])
        model = run_page('page_model', folder)
        assert 'P02' in ' '.join(str(v) for v in model.dataframe[0].value.iloc[:, 0]), model.dataframe[0].value


def test_sessions_without_emotion_windows():
    """Comparar and Sesiones with sessions stopped before the first stimulus."""
    with workdir() as folder:
        for seed in range(2):
            fake_session(config.SESSIONS_DIR, seed, participant=f'P0{seed + 1}', status='stopped', stop_after=1)
        compare = run_page('page_compare', folder)
        assert 'no tienen ventanas limpias de emoción' in text_of(compare), text_of(compare)
        sessions = run_page('page_sessions', folder)
        assert 'Se detuvo antes de terminar' in text_of(sessions)


def test_live_page():
    with workdir() as folder:
        rec = live_recording()
        live = run_page('page_live', folder)
        text = text_of(live)
        assert 'se está grabando' in text and 'La última predicción' in text, text
        assert charts(live) == 2

        for seconds, expected in ((8, 2), (20, 2), (40, 3)):  # the zoom only appears past 35 s
            short = live_recording(seconds)
            short.close('completed')
            sessions = run_page('page_sessions', folder)
            assert charts(sessions) == expected, (seconds, charts(sessions))
        rec.close('completed')
        old = time.time() - 120
        for path in glob.glob(os.path.join(config.LIVE_DIR, '*.csv')):
            os.utime(path, (old, old))
        live = run_page('page_live', folder)
        assert 'Ahora no se está grabando nada' in text_of(live) and 'terminó hace 2 min' in text_of(live), text_of(live)

        # A session without its closing line whose headset went silent a minute ago is not "finished"
        silent = live_recording(10)
        minute_ago = time.time() - 60
        os.utime(silent.path, (minute_ago, minute_ago))
        live = run_page('page_live', folder)
        assert 'no recibe muestras nuevas desde hace 1 min' in text_of(live), text_of(live)
        assert 'Ahora no se está grabando nada' not in text_of(live)


def test_launch_from_another_folder():
    """main.py --mode dashboard serves only on 127.0.0.1 and works from any working directory."""
    import urllib.request

    port = 8611
    with workdir():
        env = dict(os.environ, BROWSER='true')  # do not open a real browser in the test
        process = subprocess.Popen([sys.executable, '-B', os.path.join(REPO, 'main.py'), '--mode', 'dashboard',
                                    '--web-port', str(port)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            for _ in range(60):
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1)
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                raise AssertionError("the dashboard did not start")
            assert urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=2).status == 200
        finally:
            os.killpg(process.pid, signal.SIGINT)  # what Ctrl+C does: main.py and streamlit both stop
            output = process.communicate(timeout=30)[0].decode()
        assert 'Network URL' not in output and 'External URL' not in output, output


if __name__ == '__main__':
    os.chdir(REPO)
    only = sys.argv[1:]
    for name, fn in list(globals().items()):
        if name.startswith('test_') and (not only or name in only):
            print(name, flush=True)
            fn()
    print("All dashboard tests passed")
