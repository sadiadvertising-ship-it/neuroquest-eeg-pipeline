import json
import os
import platform
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from matplotlib.figure import Figure
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from preprocessing import feature_names

# With fewer groups the inner search would hold out a single group, so hyperparameters stay fixed,
# both when evaluating and in the final model.
MIN_GROUPS_TO_TUNE = 4


def signal_config(sampling_rate=None):
    """Everything that shapes the features. Saved with the model and in the report."""
    return {
        'sampling_rate': sampling_rate,
        'bandpass': [config.BANDPASS_LOW, config.BANDPASS_HIGH],
        'notch': config.NOTCH_FREQ,
        'artifact_uv': config.ARTIFACT_UV,
        'flat_uv': config.FLAT_UV,
        'filter_pad': config.FILTER_PAD,
        'onset_margin': config.ONSET_MARGIN,
        'window': config.WINDOW_SIZE,
        'overlap': config.OVERLAP,
        'bands': {name: list(edges) for name, edges in config.BANDS.items()},
        'channels': list(config.CHANNEL_NAMES),
        'features': feature_names(),
    }


def versions():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'scikit-learn': sklearn.__version__, 'pandas': pd.__version__}


def proba_rule_score(estimator, X, y):
    """Balanced accuracy of argmax(predict_proba), the rule used live (SVC.predict votes differently)."""
    proba = estimator.predict_proba(X)
    return balanced_accuracy_score(y, estimator.classes_[proba.argmax(axis=1)])


def soften(proba, temperature):
    """Temperature scaling: same argmax, confidence closer to what the grouped validation supports."""
    logits = np.log(np.clip(proba, 1e-6, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def fit_temperature(proba, y_index):
    """Temperature that minimizes the log-loss of the out-of-group probabilities."""
    rows = np.arange(len(proba))

    def log_loss(log_t):
        return -np.mean(np.log(soften(proba, np.exp(log_t))[rows, y_index] + 1e-12))

    return float(np.exp(minimize_scalar(log_loss, bounds=(np.log(0.05), np.log(20)), method='bounded').x))


def permute_blocks(y, sessions, blocks, rng):
    """Shuffle the labels between whole blocks inside each session (the windows of a block move together)."""
    y = np.asarray(y).copy()
    for session in np.unique(sessions):
        in_session = sessions == session
        ids = np.unique(blocks[in_session])
        labels = [y[in_session & (blocks == b)][0] for b in ids]
        for b, label in zip(ids, rng.permutation(labels)):
            y[in_session & (blocks == b)] = label
    return y


class EmotionClassifier:
    """z-score of the log band powers (Figure Arq-5, block 5) + SVM or Random Forest."""

    def __init__(self, model_type='svm'):
        self.model_type = model_type
        self.model = None
        self.classes = list(config.CLASSES)
        self.temperature = 1.0
        self.signal = None

    def _estimator(self, tune, n_groups, n_jobs):
        if self.model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
            grid = {'clf__max_depth': [None, 5, 10], 'clf__min_samples_leaf': [1, 5]}
        else:
            clf = SVC(probability=True, class_weight='balanced', random_state=42)
            grid = {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 0.01, 0.1]}
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        if not tune or n_groups < 2:
            return pipe
        return GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=min(3, n_groups)),
                            scoring=proba_rule_score, n_jobs=n_jobs)

    def _fit(self, X, y, groups, tune, n_jobs=1):
        model = self._estimator(tune, len(np.unique(groups)), n_jobs)
        if isinstance(model, GridSearchCV):
            model.fit(X, y, groups=groups)
            return model.best_estimator_, model.best_params_
        return model.fit(X, y), {}

    def _proba(self, model, X):
        """Probabilities in the order of self.classes (0 for a class the model never saw)."""
        proba = model.predict_proba(X)
        out = np.zeros((len(X), len(self.classes)))
        for j, label in enumerate(model.classes_):
            out[:, self.classes.index(label)] = proba[:, j]
        return out

    def evaluate(self, X, y, groups, tune=None, n_jobs=1):
        """
        Grouped cross-validation of the rule used live, argmax(predict_proba): test windows never
        share a group with the training windows, and hyperparameters are tuned inside each training
        fold only. tune=None tunes when there are at least MIN_GROUPS_TO_TUNE groups.
        """
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError("Need at least 2 groups (participants, sessions or rounds) to evaluate")
        if tune is None:
            tune = n_groups >= MIN_GROUPS_TO_TUNE
        labels = np.array(self.classes)
        proba = np.zeros((len(y), len(self.classes)))
        fold_of = np.zeros(len(y), dtype=int)
        folds = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
            for k, (train, test) in enumerate(GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups)):
                model, params = self._fit(X[train], y[train], groups[train], tune, n_jobs)
                proba[test] = self._proba(model, X[test])
                fold_of[test] = k
                folds.append({
                    'fold': k,
                    'test_groups': sorted(str(g) for g in np.unique(groups[test])),
                    'n_test': int(len(test)),
                    'balanced_accuracy': float(balanced_accuracy_score(y[test], labels[proba[test].argmax(1)])),
                    'chance': 1 / len(np.unique(y[test])) if len(np.unique(y[test])) > 1 else None,
                    'params': params,
                })
            y_pred = labels[proba.argmax(axis=1)]
            present = [c for c in self.classes if c in set(y)]
            return {
                'tuned': bool(tune),
                'accuracy': float(accuracy_score(y, y_pred)),
                'balanced_accuracy': float(balanced_accuracy_score(y, y_pred)),
                'f1_macro': float(f1_score(y, y_pred, labels=present, average='macro')),
                'chance': 1 / len(present),
                'folds': folds,
                'fold_scores': np.array([f['balanced_accuracy'] for f in folds]),
                'report': classification_report(y, y_pred, labels=self.classes, zero_division=0),
                'per_class': classification_report(y, y_pred, labels=self.classes, zero_division=0,
                                                   output_dict=True),
                'confusion_matrix': confusion_matrix(y, y_pred, labels=self.classes),
                'y_pred': y_pred,
                'proba': proba,
                'fold': fold_of,
            }

    def calibrate(self, y, proba):
        """Fit the temperature on out-of-group probabilities. Returns what it changed."""
        y_index = np.array([self.classes.index(label) for label in y])
        self.temperature = fit_temperature(proba, y_index)
        return {
            'temperature': self.temperature,
            'accuracy': float(np.mean(proba.argmax(1) == y_index)),
            'mean_confidence_raw': float(proba.max(1).mean()),
            'mean_confidence_calibrated': float(soften(proba, self.temperature).max(1).mean()),
        }

    def permutation_test(self, X, y, groups, sessions, blocks, observed, n, tune, n_jobs=-1, seed=0):
        """
        Null distribution of the balanced accuracy: labels are shuffled between whole blocks inside
        each session and the full procedure (tuning included) runs again. Returns (null, p_value).
        Ctrl+C keeps the permutations already done.
        """
        def one(i):
            y_perm = permute_blocks(y, sessions, blocks, np.random.default_rng(seed + i))
            return self.evaluate(X, y_perm, groups, tune=tune)['balanced_accuracy']

        null, started = [], time.time()
        try:
            for score in Parallel(n_jobs=n_jobs, return_as='generator')(delayed(one)(i) for i in range(n)):
                null.append(score)
                done = len(null)
                if done == 1 or done % 5 == 0 or done == n:
                    left = (time.time() - started) / done * (n - done)
                    print(f"  permutation {done}/{n}, about {left / 60:.0f} min left", flush=True)
        except KeyboardInterrupt:
            print(f"\n  Stopped after {len(null)} permutations")
        null = np.array(null)
        p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1)) if len(null) else None
        return null, p_value

    def train(self, X, y, groups, tune=None, n_jobs=1):
        """Final model on all the data, with the same tuning rule as the evaluation."""
        groups = np.asarray(groups)
        if tune is None:
            tune = len(np.unique(groups)) >= MIN_GROUPS_TO_TUNE
        self.model, params = self._fit(X, y, groups, tune, n_jobs)
        return params

    def plot_confusion_matrix(self, cm, save_path):
        fig = Figure(figsize=(6, 5))
        ax = fig.subplots()
        im = ax.imshow(cm, cmap='Blues')
        ticks = range(len(self.classes))
        ax.set_xticks(ticks, self.classes)
        ax.set_yticks(ticks, self.classes)
        for i in ticks:
            for j in ticks:
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion matrix (grouped cross-validation)')
        fig.colorbar(im)
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path)

    def save_model(self, path=config.MODEL_PATH, sampling_rate=None):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'classes': self.classes,
            'temperature': self.temperature,
            'signal': signal_config(sampling_rate),
            'versions': versions(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path=config.MODEL_PATH, sampling_rate=None):
        """Refuses a model whose features were computed differently; warns on another sampling rate."""
        data = joblib.load(path)
        saved = data.get('signal')
        if saved is None:
            raise ValueError("The model was saved by an older version; retrain it with --mode train")
        current = signal_config()
        changed = [key for key in current if key != 'sampling_rate' and saved.get(key) != current[key]]
        if list(data['classes']) != list(config.CLASSES):
            changed.append('classes')
        if changed:
            raise ValueError(f"The model was trained with other settings ({', '.join(changed)}); "
                             "retrain it with --mode train")
        self.model = data['model']
        self.model_type = data['model_type']
        self.classes = list(data['classes'])
        self.temperature = data.get('temperature', 1.0)
        self.signal = saved
        trained_fs = saved.get('sampling_rate')
        if sampling_rate and trained_fs and abs(trained_fs - sampling_rate) > 0.5:
            print(f"Warning: the model was trained at {trained_fs:g} Hz and the board runs at {sampling_rate:g} Hz")
        print(f"Model loaded from {path}")

    def predict(self, features):
        """Returns (emotion, confidence, probabilities in the order of self.classes)."""
        proba = soften(self._proba(self.model, np.asarray(features).reshape(1, -1)), self.temperature)[0]
        best = int(np.argmax(proba))
        return self.classes[best], float(proba[best]), proba


def report_paths(model_path):
    """Files written next to the model: (report.json, cv.csv, confusion_matrix.png)."""
    stem = os.path.splitext(model_path)[0]
    return stem + '.report.json', stem + '.cv.csv', stem + '.confusion_matrix.png'


def cv_table(train, results, classes):
    """One row per evaluated window with its fold, prediction and out-of-group probabilities."""
    cv = train[['session', 'name', 'participant', 'block', 'round', 'label', 'start', 'end', 't_start']].copy()
    cv['fold'] = results['fold']
    cv['predicted'] = results['y_pred']
    cv['correct'] = cv['predicted'] == cv['label']
    for j, label in enumerate(classes):
        cv[f'p_{label}'] = results['proba'][:, j]
    return cv


def _plain(value):
    """numpy values -> plain Python, for json."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_report(model_path, classifier, results, calibration, permutation, params, train, sessions,
                 excluded, unit, n_groups, synthetic, sampling_rate):
    counts = train['label'].value_counts()
    return _plain({
        'schema': 1,
        'created': datetime.now().isoformat(timespec='seconds'),
        'model': {'path': model_path, 'type': classifier.model_type, 'final_params': params,
                  'tuned': results['tuned'], 'temperature': classifier.temperature},
        'classes': classifier.classes,
        'validation': {'method': 'GroupKFold', 'n_folds': len(results['folds']), 'group_unit': unit,
                       'n_groups': n_groups, 'rule': 'argmax(predict_proba)',
                       'tuning': 'GridSearchCV inside each training fold' if results['tuned']
                       else f'fixed hyperparameters (fewer than {MIN_GROUPS_TO_TUNE} groups)'},
        'data': {'n_windows': int(len(train)), 'class_counts': {c: int(counts.get(c, 0)) for c in classifier.classes},
                 'synthetic': synthetic, 'sessions': sessions, 'excluded': excluded},
        'metrics': {'accuracy': results['accuracy'], 'balanced_accuracy': results['balanced_accuracy'],
                    'f1_macro': results['f1_macro'], 'chance': results['chance']},
        'folds': results['folds'],
        'per_class': {c: results['per_class'][c] for c in classifier.classes},
        'confusion_matrix': {'labels': classifier.classes, 'rows': 'true', 'columns': 'predicted',
                             'counts': results['confusion_matrix']},
        'calibration': calibration,
        'permutations': permutation,
        'config': signal_config(sampling_rate),
        'environment': versions(),
    })


def save_evaluation(model_path, report, cv):
    """report.json and cv.csv, written to a temporary file first so a crash never leaves half a file."""
    json_path, cv_path, _ = report_paths(model_path)
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    with open(json_path + '.tmp', 'w') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    os.replace(json_path + '.tmp', json_path)
    cv.to_csv(cv_path + '.tmp', index=False)
    os.replace(cv_path + '.tmp', cv_path)
    return json_path, cv_path
