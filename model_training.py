import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception as _tf_err:
    tf = None
    keras = None
    TF_AVAILABLE = False

import optuna
import config

class EmotionClassifier:
    def __init__(self, model_type='hybrid'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classes = config.CLASSES
        self.label_encoder.fit(self.classes)

    def create_cnn_lstm_model(self, input_shape, num_classes, params):
        """
        Build a Dense neural network for 1D feature vectors.
        The original CNN+LSTM expected 3D inputs (timesteps, features), but our pipeline
        provides flattened feature vectors. This dense model aligns with the current data.
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not available; deep model cannot be created.")

        inputs = keras.layers.Input(shape=input_shape)  # e.g., (n_features,)
        x = keras.layers.Dense(params.get('dense_units', 128), activation='relu')(inputs)
        x = keras.layers.Dropout(params.get('dropout', 0.2))(x)
        x = keras.layers.Dense(max(32, params.get('dense_units', 128) // 2), activation='relu')(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=params.get('learning_rate', config.LEARNING_RATE)
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            'dense_units': trial.suggest_int('dense_units', 64, 256),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }
        model = self.create_cnn_lstm_model(X_train.shape[1:], len(self.classes), params)
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=10, batch_size=config.BATCH_SIZE, verbose=0)
        score = model.evaluate(X_val, y_val, verbose=0)[1]  # Accuracy
        return score

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        if not TF_AVAILABLE:
            # Fallback defaults when TensorFlow isn't available
            return {
                'dense_units': 128,
                'dropout': 0.2,
                'learning_rate': config.LEARNING_RATE
            }
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                       n_trials=config.TUNING_TRIALS)
        return study.best_params

    def prepare_data(self, features, labels):
        smote = SMOTE(random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        features_scaled = self.scaler.fit_transform(features_resampled)
        labels_encoded = self.label_encoder.transform(labels_resampled)
        return features_scaled, labels_encoded

    def train_model(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        X_train_scaled, y_train_encoded = self.prepare_data(X_train, y_train)
        X_test_scaled = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Deep feature extractor (optional)
        if TF_AVAILABLE:
            best_params = self.tune_hyperparameters(
                X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded
            )
            dl_model = self.create_cnn_lstm_model(
                X_train_scaled.shape[1:], len(self.classes), best_params
            )
            dl_model.fit(
                X_train_scaled, y_train_encoded,
                validation_data=(X_test_scaled, y_test_encoded),
                epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=1
            )

            # Extract features from DL model (penultimate layer)
            feature_extractor = keras.Model(
                inputs=dl_model.input, outputs=dl_model.layers[-2].output
            )
            X_train_dl = feature_extractor.predict(X_train_scaled, verbose=0)
            X_test_dl = feature_extractor.predict(X_test_scaled, verbose=0)
            self.feature_extractor = feature_extractor
        else:
            print("TensorFlow not available. Skipping deep feature extraction and using scaled features.")
            X_train_dl = X_train_scaled
            X_test_dl = X_test_scaled
            self.feature_extractor = None

        # Train classical models for ensemble
        svm = SVC(probability=True)
        rf = RandomForestClassifier()

        # Ensemble
        self.model = VotingClassifier(estimators=[('svm', svm), ('rf', rf)], voting='soft')
        self.model.fit(X_train_dl, y_train_encoded)

        y_pred = self.model.predict(X_test_dl)
        y_pred_proba = self.model.predict_proba(X_test_dl)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)

        return X_test_dl, y_test, y_pred_decoded, y_pred_proba

    def cross_validate(self, features, labels, cv_folds=5):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        features_scaled, labels_encoded = self.prepare_data(features, labels)
        # If deep feature extractor was used during training, apply it here as well
        if getattr(self, 'feature_extractor', None) is not None:
            features_scaled = self.feature_extractor.predict(features_scaled, verbose=0)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, features_scaled, labels_encoded, cv=skf, scoring='accuracy')
        return scores

    def evaluate_model(self, y_true, y_pred, y_pred_proba=None):
        y_true_encoded = self.label_encoder.transform(y_true)
        report = classification_report(y_true_encoded, self.label_encoder.transform(y_pred),
                                       target_names=self.classes, output_dict=True)
        cm = confusion_matrix(y_true_encoded, self.label_encoder.transform(y_pred))
        accuracy = accuracy_score(y_true_encoded, self.label_encoder.transform(y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_encoded,
                                                                  self.label_encoder.transform(y_pred),
                                                                  average='weighted')
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm
        }
        if y_pred_proba is not None:
            results['mean_confidence'] = np.mean(np.max(y_pred_proba, axis=1))
        return results

    def plot_confusion_matrix(self, cm, save_path=None):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_model(self, path=config.MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'classes': self.classes
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load_model(self, path=config.MODEL_PATH):
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.classes = model_data['classes']
        print(f"Model loaded from {path}")

    def predict(self, features):
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        # Apply deep feature extractor if available
        if getattr(self, 'feature_extractor', None) is not None:
            features_scaled = self.feature_extractor.predict(features_scaled, verbose=0)
        prediction_encoded = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        return prediction, confidence, probabilities