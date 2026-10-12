"""
Local dashboard (Streamlit), started with: python main.py --mode dashboard
It reads data/, logs/ and models/ of this computer; it never opens the headset and sends nothing out.
Pages: Sesiones, En vivo, Modelo and Comparar. Text in Spanish, emotion names as in the data.
"""

import glob
import hashlib
import html
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis
import config
from preprocessing import Preprocessor, feature_names

REPO = os.path.dirname(os.path.abspath(__file__))

# Emotion colors validated for contrast and color-blind separation; Rest, Cue and Live stay gray
COLORS = {'Happy': '#eb6834', 'Neutral': '#2a78d6', 'Frustrated': '#4a3aa7',
          'Rest': '#c3c2b7', 'Cue': '#898781', 'Live': '#c3c2b7'}
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
DIVERGING = [[0, '#8c510a'], [0.5, '#f0efec'], [1, '#01665e']]  # away from the emotion hues
GRAYS = [[0, '#f4f3ef'], [1, '#52514e']]
MONTHS = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
SKIPPED = {'empty': 'archivo vacío', 'old_feature_format': 'formato viejo de features, hay que regrabarla',
           'unknown_format': 'formato desconocido', 'synthetic': 'grabada con --synthetic',
           'incomplete': 'incompleta, sin línea de cierre', 'stopped': 'detenida antes del final',
           'no_windows': 'sin ventanas útiles', 'error': 'no se pudo leer'}
UNITS = {'participant': ('participantes', 'participantes completos'), 'session': ('sesiones', 'sesiones completas'),
         'round': ('rondas', 'rondas completas')}

# Streamlit 1.39 drops the figure template on screen, so every style goes straight into the layout
LAYOUT = dict(
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color=INK2),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, separators=',.',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(color=INK2)),
    hoverlabel=dict(bgcolor='#ffffff', bordercolor=GRID, font=dict(color=INK, size=13)),
    margin=dict(l=72, r=24, t=48, b=48),
)
AXES = dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=GRID, tickcolor=AXIS, tickfont=dict(color=INK2),
            title_font=dict(color=INK2), automargin=True)
LEGEND_ORDER = {label: rank for rank, label in enumerate(['Happy', 'Neutral', 'Frustrated', 'Rest', 'Cue', 'Live'])}

CSS = """
body { margin: 0; background: #f9f9f7; color: #0b0b0b; font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; background: #fcfcfb; }
h1 { font-size: 28px; margin: 0 0 12px; } h2 { font-size: 19px; margin: 32px 0 8px; }
p { color: #52514e; max-width: 75ch; } code { font-size: 13px; background: #f0efec; padding: 1px 4px; border-radius: 3px; }
table { border-collapse: collapse; margin: 12px 0; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid #e1e0d9; padding: 6px 12px; text-align: left; }
th { color: #52514e; font-weight: 600; } .foot { margin-top: 40px; font-size: 13px; color: #898781; }
"""


# Spanish formatting, without depending on the system locale
def num(value, decimals=0):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '-'
    value = round(float(value), decimals) + 0.0  # never '-0'
    return f'{value:,.{decimals}f}'.replace(',', ' ').replace('.', ',').replace(' ', '.')


def pct(fraction, decimals=0):
    return f'{num(100 * fraction, decimals)} %'


def duration(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds} s'
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f'{minutes} min {rest} s' if rest else f'{minutes} min'
    hours, minutes = divmod(minutes, 60)
    return f'{hours} h {minutes} min'


def when(moment):
    return f'{moment.day} {MONTHS[moment.month - 1]} {moment.year}, {moment:%H:%M}'


def recording_label(info):
    kind = 'En vivo' if info['kind'] == 'live' else 'Sesión'
    if info['synthetic']:
        return f"{kind} sintética, {when(info['started'])}"
    return f"{kind} de {info['participant'] or 'participante sin código'}, {when(info['started'])}"


def short_label(info, seconds=False):
    who = 'sintética' if info['synthetic'] else (info['participant'] or 'sin código')
    moment = info['started']
    return f"{who}, {moment.day} {MONTHS[moment.month - 1]} {moment:%H:%M:%S}" if seconds else \
        f"{who}, {moment.day} {MONTHS[moment.month - 1]} {moment:%H:%M}"


def unique_labels(infos):
    """Short labels for recordings; seconds and then a counter are added where two would look the same."""
    labels = [short_label(info) for info in infos]
    labels = [short_label(info, seconds=True) if labels.count(label) > 1 else label for info, label in zip(infos, labels)]
    seen, out = {}, []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f'{label} ({seen[label]})')
    return out


def info_from_name(name, participant, synthetic):
    """What a label needs about a recording that is no longer on disk, read from its file name."""
    stamp = re.search(r'(\d{8}_\d{6})', name)
    started = datetime.strptime(stamp.group(1), '%Y%m%d_%H%M%S') if stamp else datetime(1970, 1, 1)
    return {'synthetic': synthetic, 'participant': participant, 'started': started}


def recording_phrase(info):
    """'la sesión de P03 del 9 oct 2026 a las 14:03', for the middle of a sentence."""
    moment = info['started']
    kind = 'la grabación en vivo' if info['kind'] == 'live' else 'la sesión'
    who = 'sintética' if info['synthetic'] else f"de {info['participant'] or 'un participante sin código'}"
    return f"{kind} {who} del {moment.day} {MONTHS[moment.month - 1]} {moment.year} a las {moment:%H:%M}"


def slug(text):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', text).strip('_').lower()


def plot_config(name):
    return {'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
            'toImageButtonOptions': {'format': 'png', 'scale': 3, 'filename': slug(name)}}


@st.cache_data(max_entries=64, show_spinner=False)
def cached_recording(key):
    """Info, samples and window table of a recording; the key (path, mtime, size) refreshes it."""
    info = analysis.recording_info(key[0])
    df, _, table = analysis.table_for(key[0])
    return info, df, table


@st.cache_data(max_entries=16, show_spinner=False)
def cached_report(key):
    return analysis.load_report(key[0])


@st.cache_resource(show_spinner=False)
def plotly_js():
    from plotly.offline import get_plotlyjs
    return get_plotlyjs()


def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)


def build_html(title, parts):
    """Self-contained page: plotly.js inlined once, the same figures and tables shown on screen."""
    body = []
    for kind, *content in parts:
        if kind == 'text':
            body.append(f'<p>{inline(content[0])}</p>')
        elif kind == 'heading':
            body.append(f'<h2>{html.escape(content[0])}</h2>')
        elif kind == 'figure':
            fig, name = content
            body.append(fig.to_html(full_html=False, include_plotlyjs=False, config=plot_config(name),
                                    default_width='100%'))
        elif kind == 'table':
            body.append(content[0].to_html(index=False, escape=True, border=0))
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style><script>{plotly_js()}</script></head>'
            f'<body><main><h1>{html.escape(title)}</h1>{"".join(body)}'
            f'<p class="foot">Generado el {when(datetime.now())} con el panel de NeuroQuest EEG.</p>'
            f'</main></body></html>').encode('utf-8')


class Page:
    """Writes text, figures and tables to the page and keeps them for the HTML export."""

    def __init__(self, title):
        self.title = title
        self.parts = []
        st.title(title)
        self.slot = st.empty()

    def text(self, markdown):
        st.markdown(markdown)
        self.parts.append(('text', markdown))

    def heading(self, text):
        st.subheader(text)
        self.parts.append(('heading', text))

    def figure(self, fig, name):
        st.plotly_chart(fig, use_container_width=True, theme=None, config=plot_config(name), key=f'fig_{slug(name)}')
        self.parts.append(('figure', fig, name))

    def table(self, df, decimals=None):
        """decimals: {column: digits} for the numeric columns, shown with Spanish separators."""
        df = df.copy()
        for column, digits in (decimals or {}).items():
            df[column] = df[column].astype(float).round(digits) + 0.0  # never '-0,0'
        styler = df.style
        exported = df.copy()
        for column, digits in (decimals or {}).items():
            styler = styler.format(precision=digits, decimal=',', thousands='.', subset=[column], na_rep='-')
            exported[column] = df[column].map(lambda v, d=digits: num(v, d))
        st.dataframe(styler, hide_index=True, use_container_width=True)
        self.parts.append(('table', exported))

    def export(self, file_name, subtitle='', signature=''):
        """signature: anything that changes when the data behind the page does (file keys), so a prepared
        download never outlives the data it was built from."""
        key = f'export_{slug(file_name)}_{hashlib.md5(repr(signature).encode()).hexdigest()[:10]}'
        with self.slot.container():
            if st.button('Preparar descarga', key=f'{key}_prepare',
                         help='Arma un HTML con esta página que abre sin internet, para enviarlo o guardarlo'):
                st.session_state[key] = build_html(f'{self.title}: {subtitle}' if subtitle else self.title, self.parts)
            if key in st.session_state:
                st.download_button('Descargar esta página (HTML)', st.session_state[key], file_name=file_name,
                                   mime='text/html', key=f'{key}_download')


def empty(message):
    st.markdown(message)


# Figures
def figure(rows=1, cols=1, **kwargs):
    fig = make_subplots(rows=rows, cols=cols, **kwargs) if rows > 1 or cols > 1 else go.Figure()
    fig.update_layout(template='none', **LAYOUT)
    fig.update_xaxes(showline=True, ticks='outside', **AXES)
    fig.update_yaxes(showline=False, **AXES)
    return fig


def block_spans(df, fs):
    """(start s, end s, label, round, block) for every block of a recording."""
    spans = []
    index = np.arange(len(df))
    for block, part in df.groupby('block', sort=False):
        spans.append((index[part.index[0]] / fs, (index[part.index[-1]] + 1) / fs, str(part['label'].iat[0]),
                      int(part['round'].iat[0]), int(block)))
    return spans


def add_channel_rows(fig, traces, first_row, fs_spans, stalls=(), limit=150):
    """Filtered signal of the 4 channels with the +-threshold lines and the emotion blocks shaded."""
    for c, (t, y) in enumerate(traces):
        row = first_row + c
        fig.add_trace(go.Scatter(x=t, y=np.round(y, 1), mode='lines', line=dict(color=INK2, width=1),
                                 name=config.CHANNEL_NAMES[c], showlegend=False,
                                 hovertemplate='%{x:.1f} s, %{y:.0f} µV<extra>' + config.CHANNEL_NAMES[c] + '</extra>'),
                      row=row, col=1)
        for level in (config.ARTIFACT_UV, -config.ARTIFACT_UV):
            fig.add_hline(y=level, line=dict(color=MUTED, width=1), row=row, col=1)
        for start, end, label, _, _ in fs_spans:
            if label in config.CLASSES:
                fig.add_vrect(x0=start, x1=end, fillcolor=COLORS[label], opacity=0.08, line_width=0, row=row, col=1)
        for at, _ in stalls:
            fig.add_vline(x=at, line=dict(color=MUTED, width=1), row=row, col=1)
        fig.update_yaxes(range=[-limit, limit], title_text=config.CHANNEL_NAMES[c], row=row, col=1,
                         tickvals=[-config.ARTIFACT_UV, 0, config.ARTIFACT_UV])


def fig_timeline(df, table, info, cv, predictions):
    """Phases, cross-validated predictions, live predictions and rejected windows above the 4 channels."""
    fs = info['sampling_rate']
    spans = block_spans(df, fs)
    t = np.arange(len(df)) / fs
    traces = analysis.decimate(t, analysis.filtered_signal(df, fs), 3000)
    ribbons = ['Fases']
    if cv is not None and len(cv):
        ribbons.append('Modelo')
    if predictions is not None and len(predictions):
        ribbons.append('En vivo')
    ribbons.append('Descartadas')
    n = len(ribbons) + config.CHANNELS
    fig = figure(n, 1, shared_xaxes=True, vertical_spacing=0.012,
                 row_heights=[0.22] * len(ribbons) + [1] * config.CHANNELS)

    shown = set()
    steps = info['steps']
    for start, end, label, round_num, block in spans:
        upcoming = next((s[0] for s in steps[block + 1:] if s[0] in config.CLASSES), None) if steps else None
        hover = f'Cue: sigue {upcoming}' if label == config.CUE_LABEL and upcoming else f'{label}, ronda {round_num}'
        fig.add_trace(go.Bar(x=[end - start], base=[start], y=['Fases'], orientation='h', width=0.9,
                             marker=dict(color=COLORS.get(label, MUTED), line=dict(width=0)),
                             name=label, legendgroup=label, showlegend=label not in shown,
                             legendrank=LEGEND_ORDER.get(label, 99),
                             hovertemplate=f'{hover}<br>{num(start)}-{num(end)} s<extra></extra>'), row=1, col=1)
        shown.add(label)

    row = 2
    if 'Modelo' in ribbons:
        for label in config.CLASSES:
            part = cv[cv['predicted'] == label]
            if not len(part):
                continue
            probability = part[f'p_{label}'].to_numpy()
            fig.add_trace(go.Scatter(
                x=part['t_start'] + config.WINDOW_SIZE / 2, y=['Modelo'] * len(part), mode='markers',
                marker=dict(symbol='square', size=9, color=COLORS[label], line=dict(color=SURFACE, width=1)),
                legendgroup=label, showlegend=False, customdata=np.stack([part['label'], probability * 100], axis=1),
                hovertemplate='real %{customdata[0]}, predicha ' + label + ' (%{customdata[1]:.0f} %)<extra></extra>'),
                row=row, col=1)
        row += 1
    if 'En vivo' in ribbons:
        ok = predictions[predictions['status'] == 'ok']
        for label in config.CLASSES:
            part = ok[ok['predicted'] == label]
            if len(part):
                fig.add_trace(go.Scatter(
                    x=part['t'], y=['En vivo'] * len(part), mode='markers', showlegend=False, legendgroup=label,
                    marker=dict(symbol='square', size=9, color=COLORS[label], line=dict(color=SURFACE, width=1)),
                    customdata=part['confidence'] * 100,
                    hovertemplate=label + ' (%{customdata:.0f} %)<extra>en vivo</extra>'), row=row, col=1)
        row += 1

    bad = table[table['artifact']]
    if len(bad):
        peaks = bad[[f'peak_{ch}' for ch in config.CHANNEL_NAMES]].to_numpy()
        worst = np.array(config.CHANNEL_NAMES)[peaks.argmax(axis=1)]
        text = [f'{ch} llegó a {p:.0f} µV' if reason == 'amplitude' else 'un canal quedó plano'
                for ch, p, reason in zip(worst, peaks.max(axis=1), bad['reason'])]
        fig.add_trace(go.Scatter(x=bad['t_start'] + config.WINDOW_SIZE / 2, y=['Descartadas'] * len(bad),
                                 mode='markers', marker=dict(symbol='line-ns', size=12, line=dict(color=INK, width=2)),
                                 text=text, showlegend=False, hovertemplate='%{text}<extra>descartada</extra>'),
                      row=row, col=1)
    else:
        fig.add_trace(go.Scatter(x=[0], y=['Descartadas'], mode='markers', marker=dict(opacity=0),
                                 showlegend=False, hoverinfo='skip'), row=row, col=1)

    for r in range(1, len(ribbons) + 1):
        fig.update_yaxes(showgrid=False, row=r, col=1, fixedrange=True)
    stalls = [(s['signal_at'], s['seconds']) for s in analysis.timing(df, fs, info['board'], info['firmware'])['stalls']]
    add_channel_rows(fig, traces, len(ribbons) + 1, spans, stalls)
    fig.update_xaxes(title_text='Tiempo desde el inicio (s)', row=n, col=1)
    fig.update_layout(height=140 * config.CHANNELS + 36 * len(ribbons) + 120, barmode='overlay', bargap=0,
                      hovermode='closest', legend=dict(traceorder='normal'))
    return fig


def fig_zoom(df, info, start, seconds):
    """A few seconds at full resolution."""
    fs = info['sampling_rate']
    first, last = int(start * fs), int((start + seconds) * fs)
    filtered = analysis.filtered_signal(df, fs)[first:last]
    t = np.arange(first, first + len(filtered)) / fs
    spans = [s for s in block_spans(df, fs) if s[1] > start and s[0] < start + seconds]
    fig = figure(config.CHANNELS, 1, shared_xaxes=True, vertical_spacing=0.02)
    spans = [(max(a, start), min(b, start + seconds), label, r, k) for a, b, label, r, k in spans]
    add_channel_rows(fig, [(t, filtered[:, c]) for c in range(config.CHANNELS)], 1, spans)
    fig.update_xaxes(range=[start, start + seconds])
    fig.update_xaxes(title_text='Tiempo desde el inicio (s)', row=config.CHANNELS, col=1)
    fig.update_layout(height=520, margin=dict(t=24))
    return fig


def fig_spectra(df, fs):
    freqs, psd = analysis.raw_spectrum(df, fs)
    fig = figure(1, config.CHANNELS, shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=config.CHANNEL_NAMES)
    for c in range(config.CHANNELS):
        fig.add_trace(go.Scatter(x=freqs, y=psd[:, c], mode='lines', line=dict(color=INK2, width=1.5), showlegend=False,
                                 hovertemplate='%{x:.1f} Hz<extra>' + config.CHANNEL_NAMES[c] + '</extra>'), row=1, col=c + 1)
        fig.add_vline(x=config.NOTCH_FREQ, line=dict(color=MUTED, width=1), row=1, col=c + 1)
        fig.update_xaxes(title_text='Hz', row=1, col=c + 1)
    fig.update_yaxes(type='log', exponentformat='power')
    fig.update_yaxes(title_text='µV²/Hz, señal cruda', row=1, col=1)
    fig.update_layout(height=260, margin=dict(t=40))
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=13, color=INK2)
    return fig


def fig_live_signal(tail, fs, seconds=10):
    """The last seconds of the 4 channels, filtered like the classifier sees them."""
    raw = tail[config.CHANNEL_NAMES].to_numpy(dtype=float)
    filtered = Preprocessor(fs).apply_filters(raw)[-int(seconds * fs):]
    labels = tail['label'].to_numpy()[-len(filtered):] if 'label' in tail else None
    t = (np.arange(len(filtered)) - len(filtered)) / fs
    spans = []
    if labels is not None and len(labels):
        edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        for a, b in zip(np.r_[0, edges], np.r_[edges, len(labels)]):
            spans.append((t[a], t[b - 1], str(labels[a]), 0, 0))
    fig = figure(config.CHANNELS, 1, shared_xaxes=True, vertical_spacing=0.02)
    add_channel_rows(fig, [(t, filtered[:, c]) for c in range(config.CHANNELS)], 1, spans)
    fig.update_xaxes(title_text='Segundos hasta ahora', row=config.CHANNELS, col=1, range=[-seconds, 0])
    fig.update_layout(height=480, margin=dict(t=16), transition_duration=0)
    return fig


def fig_live_predictions(predictions, minutes=5):
    """Probabilities of each emotion over the last minutes; gaps where the window was rejected or no data came."""
    now = predictions['timestamp'].iloc[-1]
    recent = predictions[predictions['timestamp'] >= now - 60 * minutes].copy()
    recent['t'] = recent['timestamp'] - now
    fig = figure(2, 1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.12, 1])
    ok = recent['status'] == 'ok'
    for label in config.CLASSES:
        part = recent[ok & (recent['predicted'] == label)]
        if len(part):
            fig.add_trace(go.Scatter(x=part['t'], y=['Predicción'] * len(part), mode='markers', showlegend=False,
                                     legendgroup=label, marker=dict(symbol='square', size=9, color=COLORS[label]),
                                     hovertemplate=label + '<extra></extra>'), row=1, col=1)
        probability = recent[f'p_{label}'].where(ok)
        fig.add_trace(go.Scatter(x=recent['t'], y=probability * 100, mode='lines', connectgaps=False, name=label,
                                 legendgroup=label, line=dict(color=COLORS[label], width=2),
                                 hovertemplate=label + ' %{y:.0f} %<extra></extra>'), row=2, col=1)
    rejected = recent[~ok]
    if len(rejected):
        fig.add_trace(go.Scatter(x=rejected['t'], y=['Predicción'] * len(rejected), mode='markers', showlegend=False,
                                 marker=dict(symbol='line-ns', size=10, line=dict(color=MUTED, width=2)),
                                 text=rejected['status'].map({'artifact': 'artefacto', 'no_data': 'sin datos'}),
                                 hovertemplate='%{text}<extra></extra>'), row=1, col=1)
    fig.update_yaxes(showgrid=False, fixedrange=True, row=1, col=1)
    fig.update_yaxes(range=[0, 100], title_text='Probabilidad (%)', row=2, col=1)
    fig.update_xaxes(title_text='Segundos hasta la última predicción', row=2, col=1)
    fig.update_layout(height=380, hovermode='x unified', margin=dict(t=40))
    return fig


def fig_session_scores(scores):
    order = scores.sort_values('balanced_accuracy')
    fig = figure()
    fig.add_trace(go.Scatter(x=order['chance'] * 100, y=order['label'], mode='markers', name='azar de esa sesión',
                             marker=dict(symbol='circle-open', size=12, color=MUTED, line=dict(width=2)),
                             hovertemplate='azar %{x:.0f} %<extra></extra>'))
    fig.add_trace(go.Scatter(x=order['balanced_accuracy'] * 100, y=order['label'], mode='markers',
                             name='exactitud balanceada', marker=dict(size=11, color=INK2, line=dict(color=SURFACE, width=2)),
                             customdata=order['windows'],
                             hovertemplate='%{x:.0f} % en %{customdata} ventanas<extra></extra>'))
    mean = scores['balanced_accuracy'].mean() * 100
    fig.add_vline(x=mean, line=dict(color=AXIS, width=1), annotation_position='bottom right',
                  annotation=dict(text=f'media {num(mean)} %', font=dict(color=INK2)))
    fig.update_xaxes(range=[0, 100], title_text='Exactitud balanceada en la validación agrupada (%)')
    fig.update_yaxes(showgrid=False)
    fig.update_layout(height=max(220, 34 * len(scores) + 120))
    return fig


def fig_confusion(matrix, labels):
    counts = np.array(matrix)
    rows = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, rows, out=np.zeros_like(counts, dtype=float), where=rows > 0)
    text = [[f'{counts[i, j]}<br>{num(100 * share[i, j])} %' for j in range(len(labels))] for i in range(len(labels))]
    fig = figure()
    fig.add_trace(go.Heatmap(z=share * 100, x=labels, y=labels, colorscale=GRAYS, zmin=0, zmax=100, text=text,
                             texttemplate='%{text}', showscale=False, xgap=2, ygap=2,
                             hovertemplate='real %{y}, predicha %{x}: %{text}<extra></extra>'))
    fig.update_traces(textfont=dict(size=14))
    fig.update_yaxes(autorange='reversed', title_text='Real', title_standoff=16, showgrid=False)
    fig.update_xaxes(title_text='Predicha', side='top', showgrid=False, showline=False, ticks='')
    fig.update_layout(height=340, width=None, margin=dict(t=72))
    return fig


def fig_failures(by_second, by_round, chance):
    fig = figure(1, 2, horizontal_spacing=0.12, column_widths=[0.7, 0.3],
                 subplot_titles=['Por segundo dentro del bloque', 'Por ronda'])
    fig.add_trace(go.Scatter(x=by_second['second'], y=by_second['mean'] * 100, mode='lines+markers', showlegend=False,
                             line=dict(color=INK2, width=2), marker=dict(size=8, color=INK2, line=dict(color=SURFACE, width=2)),
                             customdata=by_second['size'],
                             hovertemplate='segundo %{x}: %{y:.0f} % (%{customdata} ventanas)<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=by_round['round'].astype(str), y=by_round['mean'] * 100, mode='markers', showlegend=False,
                             marker=dict(size=11, color=INK2, line=dict(color=SURFACE, width=2)),
                             customdata=by_round['size'],
                             hovertemplate='ronda %{x}: %{y:.0f} % (%{customdata} ventanas)<extra></extra>'), row=1, col=2)
    for col in (1, 2):
        fig.add_hline(y=chance * 100, line=dict(color=MUTED, width=1), row=1, col=col)
        fig.update_yaxes(range=[0, 100], row=1, col=col)
    fig.update_yaxes(title_text='Ventanas acertadas (%)', row=1, col=1)
    fig.update_xaxes(type='category', title_text='Ronda', row=1, col=2)
    fig.update_xaxes(title_text='Segundo en que empieza la ventana', row=1, col=1)
    fig.add_annotation(x=1, xref='x domain', y=chance * 100, yref='y', text='azar', showarrow=False,
                       xanchor='right', yanchor='bottom', font=dict(color=MUTED))
    fig.update_layout(height=320, margin=dict(t=56))
    for annotation in fig.layout.annotations:
        if annotation.text != 'azar':
            annotation.font = dict(size=13, color=INK2)
    return fig


def fig_contrast(contrast):
    bands = list(config.BANDS)
    fig = figure(1, config.CHANNELS, shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=config.CHANNEL_NAMES)
    offsets = {label: (i - 1) * 0.18 for i, label in enumerate(config.CLASSES)}
    for c, channel in enumerate(config.CHANNEL_NAMES):
        for label in config.CLASSES:
            part = contrast[(contrast['channel'] == channel) & (contrast['label'] == label)].set_index('band').reindex(bands)
            x = np.arange(len(bands)) + offsets[label]
            fig.add_trace(go.Scatter(
                x=x, y=part['db'], mode='markers', name=label, legendgroup=label, showlegend=c == 0,
                marker=dict(size=9, color=COLORS[label], line=dict(color=SURFACE, width=2)),
                error_y=dict(type='data', array=part['se'], visible=True, color=COLORS[label], thickness=1.5, width=0),
                customdata=np.stack([bands, part['sessions']], axis=1),
                hovertemplate=label + ', %{customdata[0]}: %{y:.1f} dB (%{customdata[1]} sesiones)<extra>' + channel + '</extra>'),
                row=1, col=c + 1)
        fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=1, col=c + 1)
        fig.update_xaxes(tickvals=list(range(len(bands))), ticktext=bands, row=1, col=c + 1, showgrid=False)
    fig.update_yaxes(title_text='dB frente al reposo', row=1, col=1)
    fig.update_layout(height=360, margin=dict(t=40, b=40),
                      legend=dict(orientation='h', yanchor='top', y=-0.32, xanchor='left', x=0))
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=13, color=INK2)
    return fig


def fig_fingerprint(matrix):
    names = list(matrix.columns)
    limit = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy())))) if len(matrix) else 1.0
    fig = figure()
    fig.add_trace(go.Heatmap(z=matrix.to_numpy(), x=names, y=list(matrix.index), colorscale=DIVERGING, zmid=0,
                             zmin=-limit, zmax=limit, xgap=2, ygap=2,
                             colorbar=dict(title=dict(text='dB', side='right'), thickness=10, outlinewidth=0),
                             hovertemplate='%{y}, %{x}: %{z:.1f} dB<extra></extra>'))
    for c in range(1, config.CHANNELS):
        fig.add_vline(x=c * len(config.BANDS) - 0.5, line=dict(color=SURFACE, width=6))
    fig.update_xaxes(tickangle=-45, showgrid=False, showline=False, ticks='')
    fig.update_yaxes(showgrid=False, autorange='reversed')
    fig.update_layout(height=max(260, 30 * len(matrix) + 170), margin=dict(l=170, b=110))
    return fig


def fig_rejection(rejected):
    order = rejected.iloc[::-1]
    fig = figure()
    fig.add_trace(go.Bar(x=order['rejected_pct'], y=order['group'], orientation='h', width=0.5,
                         marker=dict(color=INK2, line=dict(width=0)), showlegend=False,
                         customdata=np.stack([order['rejected'], order['windows']], axis=1),
                         hovertemplate='%{x:.0f} % (%{customdata[0]} de %{customdata[1]} ventanas)<extra></extra>'))
    fig.update_xaxes(title_text='Ventanas de emoción descartadas (%)', rangemode='tozero')
    fig.update_yaxes(showgrid=False)
    fig.update_layout(height=max(200, 30 * len(order) + 110), margin=dict(l=170))
    return fig


# Pages
def page_sessions():
    page = Page('Sesiones')
    paths = analysis.recordings()
    if not paths:
        page.text(f"Todavía no hay grabaciones en `{config.SESSIONS_DIR}`. Cuando grabes una sesión con "
                  "`python main.py --mode gui --participant P01` (o con `--mode record`), va a aparecer aquí con "
                  "su señal, las fases del protocolo y las ventanas descartadas por artefactos.")
        return
    infos = [analysis.recording_info(p) for p in paths]
    choice = st.selectbox('Grabación', range(len(infos)), format_func=lambda i: recording_label(infos[i]),
                          key='session_choice')
    info = infos[choice]
    if info['problem']:
        page.text(f"`{info['name']}` no se puede mostrar: {SKIPPED.get(info['problem'], info['problem'])}.")
        return
    with st.spinner('Leyendo la grabación...'):
        info, df, table = cached_recording(analysis.file_key(info['path']))
    if not len(df):
        page.text("Se creó el archivo, pero todavía no tiene muestras.")
        return

    fs = info['sampling_rate']
    timing = analysis.timing(df, fs, info['board'], info['firmware'])
    predictions = None
    predictions_path = analysis.predictions_for(info['path'])
    if predictions_path:
        predictions, _ = analysis.read_predictions(predictions_path)
        predictions['t'] = predictions['timestamp'] - df['timestamp'].iat[0]
    report, cv = cached_report(analysis.file_key(analysis.reports()[0])) if analysis.reports() else (None, None)
    session_cv = cv[cv['session'] == os.path.relpath(info['path'])] if cv is not None else None

    page.text(summary_sentence(info, timing, table))
    page.figure(fig_timeline(df, table, info, session_cv, predictions), f"linea_de_tiempo_{info['name']}")
    if session_cv is not None and len(session_cv):
        held_out = {'participant': 'cuando su participante quedó por fuera del entrenamiento',
                    'session': 'cuando esta sesión quedó por fuera del entrenamiento',
                    'round': 'cuando la ronda de cada ventana quedó por fuera del entrenamiento (las otras rondas de '
                             'esta sesión sí se usaron para entrenar)'}.get(report['validation']['group_unit'], '')
        page.text(f"La fila Modelo muestra lo que predijo el modelo de `{report['model']['path']}` para cada ventana "
                  f"{held_out}; donde su color no coincide con el de la fase, se equivocó.")

    total = len(df) / fs
    if total > 35:  # shorter recordings already show every peak in the timeline
        seconds = 30
        start = st.slider('Ver de cerca, desde el segundo', 0, int(total) - seconds, 0, step=1,
                          key=f"zoom_{info['name']}", help=f'Muestra {seconds} s con todas las muestras')
        st.plotly_chart(fig_zoom(df, info, start, seconds), use_container_width=True, theme=None,
                        config=plot_config(f"detalle_{info['name']}"), key='fig_zoom')

    page.heading('Calidad por canal')
    quality = analysis.channel_quality(df, fs, table)
    for sentence in quality_sentences(quality):
        page.text(sentence)
    page.figure(fig_spectra(df, fs), f"espectros_{info['name']}")
    page.table(quality.rename(columns={'channel': 'Canal', 'line_uv': 'Ruido de 60 Hz (µV)', 'rms_uv': 'Amplitud RMS (µV)',
                                       'over_pct': 'Ventanas fuera de ±100 µV (%)', 'flat_pct': 'Tiempo plano (%)'}),
               {'Ruido de 60 Hz (µV)': 1, 'Amplitud RMS (µV)': 1, 'Ventanas fuera de ±100 µV (%)': 0, 'Tiempo plano (%)': 0})

    if info['kind'] == 'session':
        page.heading('Bloques del protocolo')
        blocks = analysis.blocks(df, table, info)
        page.table(blocks.rename(columns={'block': 'Bloque', 'label': 'Fase', 'round': 'Ronda', 'seconds': 'Duración (s)',
                                          'planned': 'Planeada (s)', 'windows': 'Ventanas', 'rejected': 'Descartadas',
                                          'onset': 'En el margen inicial'}),
                   {'Duración (s)': 1, 'Planeada (s)': 0})
    page.export(f"neuroquest_{info['name']}.html", recording_label(info),
                (analysis.file_key(info['path']), analysis.reports()[:1] and analysis.file_key(analysis.reports()[0])))


def summary_sentence(info, timing, table):
    fs = info['sampling_rate']
    who = 'sintética' if info['synthetic'] else f"de {info['participant'] or 'un participante sin código'}"
    kind = 'Grabación en vivo' if info['kind'] == 'live' else 'Sesión'
    parts = [f"{kind} {who} del {when(info['started'])}: duró {duration(timing['seconds'])} y llegaron "
             f"{num(timing['samples'])} muestras, {num(timing['effective_fs'], 1)} por segundo "
             f"(la placa promete {num(fs)})."]
    if timing['lost'] is not None:
        lost = timing['lost']
        note = ' según el contador de paquetes; en la Ganglion esa cuenta falta confirmarla con la diadema' \
            if info['board'] == 'ganglion' else ' según el contador de paquetes'
        parts.append(f"No se perdió ninguna muestra{note}." if lost == 0 else
                     f"Se perdieron {num(lost)} muestras ({pct(lost / max(timing['expected'], 1), 1)}){note}.")
    if timing['stalls']:
        longest = max(timing['stalls'], key=lambda s: s['seconds'])
        parts.append(f"Hubo {len(timing['stalls'])} pausa{'s' if len(timing['stalls']) > 1 else ''} en la llegada de datos; "
                     f"la más larga, de {num(longest['seconds'], 1)} s, en el segundo {num(longest['at'])}.")
    emotion = table[table['label'].isin(config.CLASSES)]
    if len(emotion):
        bad = emotion[emotion['artifact']]
        sentence = f"De {num(len(emotion))} ventanas de emoción se descartaron {num(len(bad))} ({pct(len(bad) / len(emotion))})"
        if len(bad):
            amplitude = bad[bad['reason'] == 'amplitude']
            if len(amplitude):
                over = (amplitude[[f'peak_{ch}' for ch in config.CHANNEL_NAMES]] > config.ARTIFACT_UV).sum()
                top = over.sort_values(ascending=False)
                channels = ' y '.join(c.replace('peak_', '') for c in top.index[:2] if top[c] > 0)
                sentence += f": {num(len(amplitude))} por pasar de ±{config.ARTIFACT_UV:.0f} µV, sobre todo en {channels}"
            flat = int((bad['reason'] == 'flat').sum())
            if flat:
                sentence += f"{' y' if len(amplitude) else ':'} {num(flat)} por un canal plano"
        parts.append(sentence + '.')
    status = info['status']
    left_out = '' if info['synthetic'] else ', así que el entrenamiento la deja por fuera'
    if info['synthetic']:
        parts.append("Es una grabación sintética: sirve para probar el programa y no entra al entrenamiento.")
    if status == 'incomplete' and analysis.age_seconds(info['path']) < 5:
        parts.append("Se está grabando ahora mismo; su señal en vivo está en la página En vivo.")
    elif info['kind'] == 'session' and status == 'stopped':
        parts.append(f"Se detuvo antes de terminar el protocolo{left_out}.")
    elif info['kind'] == 'session' and status == 'incomplete':
        parts.append(f"No tiene línea de cierre (se cortó antes de terminar){left_out}.")
    return ' '.join(parts)


def quality_sentences(quality):
    out = []
    lines = quality['line_uv'].to_numpy()
    for i, row in quality.iterrows():
        others = np.median(np.delete(lines, i))
        if row['line_uv'] >= analysis.LINE_MIN_UV and row['line_uv'] >= analysis.LINE_RATIO * max(others, 1e-6):
            out.append(f"{row['channel']} tiene {num(row['line_uv'] / max(others, 1e-6))} veces más ruido de 60 Hz que "
                       f"la mediana de los otros canales ({num(row['line_uv'], 1)} µV): su electrodo probablemente tenía mal contacto.")
        if row['flat_pct'] >= 5:
            out.append(f"{row['channel']} estuvo plano el {num(row['flat_pct'])} % del tiempo: suelto o saturado.")
        if row['over_pct'] >= 30:
            out.append(f"{row['channel']} pasó de ±{config.ARTIFACT_UV:.0f} µV en el {num(row['over_pct'])} % de las ventanas.")
    if not out:
        out.append("Ningún canal se sale de lo normal: ruido de 60 Hz parecido entre canales, sin tramos planos y "
                   "con pocas ventanas fuera del umbral.")
    return out


def page_live():
    st.title('En vivo')
    st.caption('Se actualiza cada segundo con lo que se está grabando en este computador.')
    live_view()


@st.fragment(run_every=1)
def live_view():
    paths = analysis.recordings()
    start = ("Para ver la señal aquí, abre otra terminal y ejecuta `python main.py --mode gui --participant P01` para "
             "grabar una sesión con el protocolo, o `python main.py --mode realtime` para clasificar en vivo "
             "(con `--synthetic` se prueba sin la diadema). Esta página revisa cada segundo y muestra la señal en "
             "cuanto llega la primera muestra.")
    if not paths:
        empty("Ahora no se está grabando nada. " + start)
        return
    path = paths[0]
    age = analysis.age_seconds(path)
    info = analysis.recording_info(path)
    # Without its closing line the file may belong to a session still running whose headset went silent:
    # keep warning about it for 10 minutes instead of calling it finished
    if (info['status'] != 'incomplete' and age > 30) or age > 600:
        ended = {'completed': 'terminó', 'stopped': 'se detuvo'}.get(info['status'], 'quedó incompleta, sin línea de cierre,')
        empty(f"Ahora no se está grabando nada. {start} La última grabación, {recording_phrase(info)}, "
              f"{ended} hace {duration(age)}; está en la página Sesiones.")
        return
    tail, _ = analysis.read_tail(path, 16)
    if not len(tail):
        empty(f"Se creó `{info['name']}` pero todavía no llegan muestras; si sigue así, revisa la diadema y el dongle.")
        return
    fs = info['sampling_rate']

    if info['status'] == 'completed':
        state = f"terminó hace {duration(age)}"
    elif info['status'] == 'stopped':
        state = f"se detuvo hace {duration(age)}"
    elif age > 3:
        state = (f"no recibe muestras nuevas desde hace {duration(age)}: si la sesión sigue abierta, revisa la "
                 "batería de la diadema y el dongle")
    else:
        state = 'se está grabando'
    predictions, _ = analysis.read_predictions(analysis.predictions_for(path))

    if info['kind'] == 'session':
        where = analysis.progress(tail, info)
        sentence = f"{recording_phrase(info)[0].upper()}{recording_phrase(info)[1:]} {state}."
        if where and state == 'se está grabando':
            sentence += f" Bloque {where['block'] + 1} de {where['blocks']}: {where['label']}"
            sentence += f", ronda {where['round']}" if where['label'] in config.CLASSES else ''
            if where['left'] is not None:
                sentence += f"; quedan {num(where['left'])} s"
            if where['next']:
                sentence += f"; la siguiente emoción es {where['next']}"
            sentence += '.'
        counts = running_counts(path)
        if counts:
            sentence += f" Van {num(counts[0])} ventanas de emoción y se han descartado {num(counts[1])}."
        st.markdown(sentence)
    else:
        sentence = f"{recording_phrase(info)[0].upper()}{recording_phrase(info)[1:]} {state}."
        ok = predictions[predictions['status'] == 'ok'] if len(predictions) else predictions
        if len(predictions):
            last = predictions.iloc[-1]
            ago = time.time() - last['timestamp']
            if last['status'] == 'ok':
                moment = 'hace un momento' if ago < 1.5 else f'hace {duration(ago)}'
                sentence += f" La última predicción, {moment}, fue {last['predicted']} con {num(100 * last['confidence'])} %."
            elif last['status'] == 'artifact':
                sentence += " La última ventana se descartó por un artefacto."
            else:
                sentence += " Desde la última predicción no llegan datos."
            if len(ok):
                common = ok['predicted'].value_counts()
                sentence += f" En total van {num(len(ok))} predicciones; la más frecuente es {common.index[0]} ({pct(common.iat[0] / len(ok))})."
        st.markdown(sentence)
        if len(predictions):
            st.plotly_chart(fig_live_predictions(predictions), use_container_width=True, theme=None,
                            config=plot_config('predicciones_en_vivo'), key='live_predictions')

    status = analysis.live_status(tail, fs)
    if status is not None:
        warnings = analysis.channel_warnings(status)
        st.markdown(' '.join(warnings) if warnings else "Los cuatro canales se ven bien: sin artefactos, sin tramos "
                    "planos y con ruido de 60 Hz parecido entre ellos.")
    if info['kind'] == 'session' and len(predictions):
        last = predictions.iloc[-1]
        if last['status'] == 'ok':
            st.markdown(f"La GUI está clasificando en vivo: la última predicción fue {last['predicted']} "
                        f"({num(100 * last['confidence'])} %) durante {last['label'] or 'el reposo'}.")
    st.plotly_chart(fig_live_signal(tail, fs), use_container_width=True, theme=None,
                    config=plot_config('senal_en_vivo'), key='live_signal')


def running_counts(path):
    """(emotion windows, rejected) of the recording in progress, recomputed at most every 5 s."""
    cached = st.session_state.get('live_counts')
    if cached and cached[0] == path and time.time() - cached[1] < 5:
        return cached[2]
    counts = None
    if os.path.getsize(path) < 20e6:
        try:
            _, _, table = analysis.table_for(path)
            emotion = table[table['label'].isin(config.CLASSES)]
            counts = (len(emotion), int(emotion['artifact'].sum()))
        except Exception:
            counts = None
    st.session_state['live_counts'] = (path, time.time(), counts)
    return counts


def page_model():
    page = Page('Modelo')
    paths = analysis.reports()
    if not paths:
        page.text("Todavía no hay un modelo evaluado. Al entrenar con `python main.py --mode train`, el entrenamiento "
                  "guarda su evaluación junto al modelo, y aquí vas a ver la exactitud de cada sesión frente al azar, "
                  "la matriz de confusión y en qué ventanas falla.")
        if glob.glob(os.path.join(os.path.dirname(config.MODEL_PATH) or '.', '*.pkl')):
            page.text("Hay un modelo guardado sin evaluación: se entrenó antes de que el entrenamiento la guardara. "
                      "Vuelve a entrenarlo para verla aquí.")
        return
    choice = paths[0] if len(paths) == 1 else st.selectbox(
        'Modelo', paths, format_func=lambda p: os.path.basename(p).replace('.report.json', ''), key='model_choice')
    report, cv = cached_report(analysis.file_key(choice))
    metrics, validation = report['metrics'], report['validation']
    unit_many, unit_whole = UNITS.get(validation['group_unit'], (validation['group_unit'],) * 2)
    data = report['data']
    model_name = 'SVM' if report['model']['type'] == 'svm' else 'Random Forest'

    text = (f"El modelo {model_name} de `{report['model']['path']}` se evaluó dejando por fuera {unit_whole}: "
            f"{validation['n_folds']} pliegues con {num(len(data['sessions']))} sesiones y {num(data['n_windows'])} ventanas. "
            f"Su exactitud balanceada fue {pct(metrics['balanced_accuracy'])}, frente a {pct(metrics['chance'])} del azar.")
    permutation = report.get('permutations')
    if permutation:
        verdict = 'supera al azar' if permutation['p_value'] <= 0.05 else 'no se distingue del azar'
        text += (f" En {permutation['n']} permutaciones, p = {num(permutation['p_value'], 3)}: {verdict} "
                 f"(el 95 % de las permutaciones quedó por debajo de {pct(permutation['null_p95'])}).")
    else:
        text += (" No se hizo la prueba de permutación, así que esto solo no muestra que supere al azar "
                 "(entrena con `--permutations 100`).")
    scores = analysis.session_scores(cv) if cv is not None else pd.DataFrame()
    if len(scores):
        below = int((scores['balanced_accuracy'] <= scores['chance']).sum())  # NaN chance (one class) never counts
        text += (f" Por sesión va de {pct(scores['balanced_accuracy'].min())} a {pct(scores['balanced_accuracy'].max())}"
                 + (f", y {below} {'queda' if below == 1 else 'quedan'} en su azar o por debajo." if below else '.'))
    page.text(text)
    if validation['group_unit'] == 'round':
        page.text("Ojo: se dejaron por fuera rondas de las mismas sesiones, así que esto dice qué tan bien se reconoce "
                  "la emoción dentro de una sesión ya vista, no en una persona o un día nuevos.")
    if data.get('synthetic'):
        page.text("Se entrenó con grabaciones sintéticas: sirve para probar el programa, no dice nada de emociones.")

    if len(scores):
        infos = [analysis.recording_info(row.session) if os.path.exists(row.session) else
                 info_from_name(row.name, row.participant, data.get('synthetic', False))
                 for row in scores.itertuples()]
        scores['label'] = unique_labels(infos)
        page.figure(fig_session_scores(scores), 'exactitud_por_sesion')
        page.table(scores[['label', 'windows', 'balanced_accuracy', 'chance']].assign(
            balanced_accuracy=scores['balanced_accuracy'] * 100, chance=scores['chance'] * 100).rename(columns={
                'label': 'Sesión', 'windows': 'Ventanas', 'balanced_accuracy': 'Exactitud balanceada (%)',
                'chance': 'Azar (%)'}), {'Exactitud balanceada (%)': 0, 'Azar (%)': 0})

    page.heading('Matriz de confusión')
    page.figure(fig_confusion(report['confusion_matrix']['counts'], report['confusion_matrix']['labels']), 'matriz_de_confusion')
    per_class = pd.DataFrame([{'Clase': c, 'Precisión': v['precision'], 'Sensibilidad': v['recall'], 'F1': v['f1-score'],
                               'Ventanas': int(v['support'])} for c, v in report['per_class'].items()])
    page.table(per_class, {'Precisión': 2, 'Sensibilidad': 2, 'F1': 2})
    calibration = report.get('calibration') or {}
    if calibration:
        page.text(f"Confianza: antes de calibrar, el modelo decía estar seguro en promedio al "
                  f"{pct(calibration['mean_confidence_raw'])}, pero acertó el {pct(calibration['accuracy'])}. "
                  f"Con la calibración (T = {num(calibration['temperature'], 2)}) la confianza media pasa a "
                  f"{pct(calibration['mean_confidence_calibrated'])}, que es la que se muestra en vivo.")

    if cv is not None and len(cv):
        tables = []
        for session in cv['session'].unique():
            if os.path.exists(session):
                tables.append(cached_recording(analysis.file_key(session))[2].assign(session=session))
        if tables:
            tables = pd.concat(tables, ignore_index=True)
            page.heading('Dónde falla')
            by_second, by_round, worst = analysis.error_patterns(cv, tables)
            page.figure(fig_failures(by_second, by_round, metrics['chance']), 'donde_falla')
            page.text("Los bloques con más errores:")
            page.table(worst.assign(accuracy=worst['accuracy'] * 100)[['name', 'block', 'label', 'predicted', 'accuracy', 'windows']]
                       .rename(columns={'name': 'Sesión', 'block': 'Bloque', 'label': 'Real', 'predicted': 'Predicha más veces',
                                        'accuracy': 'Acertadas (%)', 'windows': 'Ventanas'}), {'Acertadas (%)': 0})

            contrast = analysis.emotion_contrast(tables)
            if len(contrast):
                page.heading('Diferencias de potencia frente al reposo')
                sessions = int(contrast['sessions'].max())
                page.text(f"Potencia de cada emoción frente al reposo de la misma sesión, en dB, por banda y canal "
                          f"(promedio de {sessions} {'sesiones' if sessions != 1 else 'sesión'}; las barras son el error "
                          "estándar). Solo usa ventanas limpias y el reposo anterior al anuncio de la emoción siguiente.")
                page.figure(fig_contrast(contrast), 'potencia_frente_al_reposo')
        else:
            page.text("Las sesiones de este modelo ya no están en esta carpeta, así que no se puede ver dónde falla.")

    page.heading('Cómo se entrenó')
    params = report['model'].get('final_params') or {}
    used = report['config']
    tuning = ('Los hiperparámetros se eligieron dentro de cada pliegue de entrenamiento' if report['model']['tuned']
              else 'Los hiperparámetros quedaron fijos porque había menos de 4 grupos')
    if params:
        shown = [f"{key.replace('clf__', '')} = {str(value).replace('.', ',')}" for key, value in params.items()]
        tuning += '; el modelo final quedó con ' + ', '.join(shown)
    page.text(f"{tuning}. Señal filtrada entre {num(used['bandpass'][0], 1)} y {num(used['bandpass'][1])} Hz con notch de "
              f"{num(used['notch'])} Hz, ventanas de {num(used['window'])} s, umbral de artefactos de "
              f"±{num(used['artifact_uv'])} µV y {num(used['onset_margin'])} s de margen al inicio de cada estímulo.")
    def group_name(group):
        path, _, round_num = group.partition('#')
        name = os.path.splitext(os.path.basename(path))[0]
        return f'{name} (ronda {round_num})' if round_num else name
    folds = pd.DataFrame([{'Pliegue': f['fold'] + 1, f'{unit_many.capitalize()} de prueba': ', '.join(group_name(g) for g in f['test_groups']),
                           'Ventanas': f['n_test'], 'Exactitud balanceada (%)': 100 * f['balanced_accuracy'],
                           'Azar (%)': 100 * f['chance']} for f in report['folds']])
    page.table(folds, {'Exactitud balanceada (%)': 0, 'Azar (%)': 0})
    if data.get('excluded'):
        page.text("Grabaciones que quedaron por fuera: " + '; '.join(
            f"`{os.path.basename(e['file'])}` ({SKIPPED.get(e['reason'], e['reason'])})" for e in data['excluded']) + '.')
    page.export(f"neuroquest_modelo_{slug(os.path.basename(choice).replace('.report.json', ''))}.html",
                report['model']['path'], analysis.file_key(choice))


def page_compare():
    page = Page('Comparar')
    infos = [analysis.recording_info(p) for p in analysis.recordings()]
    sessions = [i for i in infos if i['kind'] == 'session' and not i['problem'] and not i['synthetic']]
    note = ''
    if len(sessions) < 2:
        synthetic = [i for i in infos if i['kind'] == 'session' and not i['problem'] and i['synthetic']]
        if len(synthetic) >= 2 and not sessions:
            sessions, note = synthetic, "Todavía no hay sesiones reales, así que se comparan las sintéticas. "
        else:
            page.text(f"Para comparar hacen falta al menos dos sesiones y hay {len(sessions)} en `{config.SESSIONS_DIR}`. "
                      "La próxima que grabes va a aparecer aquí junto a las demás, con su huella espectral y cuántas "
                      "ventanas se descartaron.")
            return
    options = ['Sesión', 'Día'] + (['Participante'] if any(i['participant'] for i in sessions) else [])
    group_by = st.radio('Agrupar por', options, horizontal=True, key='compare_group')
    with st.spinner('Leyendo las sesiones...'):
        loaded = []
        for info in sessions:
            try:
                loaded.append((info, cached_recording(analysis.file_key(info['path']))[2]))
            except Exception:
                continue
    tables = pd.concat([table for _, table in loaded], ignore_index=True)
    if group_by == 'Día':
        groups = {table['session'].iat[0]: f"{info['started'].day} {MONTHS[info['started'].month - 1]} {info['started'].year}"
                  for info, table in loaded if len(table)}
    elif group_by == 'Participante':
        groups = {table['session'].iat[0]: info['participant'] or ('sintética' if info['synthetic'] else 'sin código')
                  for info, table in loaded if len(table)}
    else:
        labels = unique_labels([info for info, _ in loaded])
        groups = {table['session'].iat[0]: label for (info, table), label in zip(loaded, labels) if len(table)}

    rejected = analysis.rejection(tables, groups)
    days = sorted({i['started'].date() for i, _ in loaded})
    people = {i['participant'] for i, _ in loaded if i['participant']}
    text = note + (f"Hay {len(loaded)} sesiones" + (f" de {len(people)} participantes" if people else '')
                   + (f", entre el {days[0].day} {MONTHS[days[0].month - 1]} y el {days[-1].day} {MONTHS[days[-1].month - 1]}"
                      if len(days) > 1 else f", del {days[0].day} {MONTHS[days[0].month - 1]}") + '.')
    if len(rejected) and rejected['rejected'].sum() == 0:
        text += " No se descartó ninguna ventana de emoción en ninguna sesión."
    elif len(rejected):
        worst = rejected.sort_values('rejected_pct').iloc[-1]
        text += (f" Las ventanas descartadas van de {num(rejected['rejected_pct'].min())} % a {num(rejected['rejected_pct'].max())} % "
                 f"(mediana {num(rejected['rejected_pct'].median())} %); el grupo con más descartes es {worst['group']}"
                 + (f", sobre todo por {worst['worst_channel']}." if worst['worst_channel'] else '.'))
    page.text(text)

    page.heading('Huella espectral')
    matrix = analysis.fingerprint(tables, groups)
    if len(matrix):
        page.text("Potencia de cada banda y canal en las ventanas limpias de emoción, en dB respecto de la mediana de los "
                  "grupos: un grupo muy distinto en todas las bandas de un canal suele ser un electrodo, no una emoción.")
        page.figure(fig_fingerprint(matrix), 'huella_espectral')
    else:
        page.text("Estas sesiones no tienen ventanas limpias de emoción todavía, así que no hay huella que comparar. "
                  "Aparece en cuanto haya al menos una sesión con bloques de emoción sin artefactos.")

    page.heading('Ventanas descartadas')
    if len(rejected) and rejected['rejected'].sum():
        page.figure(fig_rejection(rejected), 'ventanas_descartadas')
    if len(rejected):
        page.table(rejected.rename(columns={'group': 'Grupo', 'sessions': 'Sesiones', 'windows': 'Ventanas de emoción',
                                        'rejected': 'Descartadas', 'rejected_pct': 'Descartadas (%)', 'flat': 'Por canal plano',
                                        'worst_channel': 'Canal con más descartes'}), {'Descartadas (%)': 0})
    if len(matrix):
        page.table(matrix.reset_index().rename(columns={'group': 'Grupo'}), {name: 1 for name in feature_names()})
    page.export(f"neuroquest_comparar_{slug(group_by)}.html", f'agrupado por {group_by.lower()}',
                (group_by, [analysis.file_key(info['path']) for info, _ in loaded]))


def main():
    st.set_page_config(page_title='NeuroQuest EEG', layout='wide')
    os.chdir(REPO)
    st.sidebar.caption('Lee `data/`, `logs/` y `models/` de este computador; no envía nada a internet.')
    pages = [st.Page(page_sessions, title='Sesiones', default=True),
             st.Page(page_live, title='En vivo', url_path='en-vivo'),
             st.Page(page_model, title='Modelo', url_path='modelo'),
             st.Page(page_compare, title='Comparar', url_path='comparar')]
    st.navigation(pages).run()


if __name__ == '__main__':
    main()
