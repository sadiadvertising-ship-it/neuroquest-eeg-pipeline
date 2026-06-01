#!/usr/bin/env python3
"""
EEG Emotion Classification System
Main entry point with command-line interface
"""

import argparse
import sys
import time
import os
import numpy as np
from data_acquisition import DataAcquisition
from preprocessing import Preprocessor
from model_training import EmotionClassifier
from real_time_classification import run_real_time_classification
from gui_app import run_gui
from utils import Logger
import config

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='EEG Emotion Classification System')

    parser.add_argument('--mode', choices=['train', 'realtime', 'gui'], required=True,
                        help='Operation mode: train, realtime or gui')
    parser.add_argument('--port', default=config.SERIAL_PORT,
                        help=f'Serial port for OpenBCI board (default: {config.SERIAL_PORT})')
    parser.add_argument('--model', default=config.MODEL_PATH,
                        help=f'Path to model file (default: {config.MODEL_PATH})')
    parser.add_argument('--duration', type=int, default=300,
                        help='Training duration in seconds (default: 300)')
    parser.add_argument('--model-type', choices=['svm', 'rf'], default='svm',
                        help='Model type for training (default: svm)')
    parser.add_argument('--csv', help='Path to CSV file with pre-collected training data')
    parser.add_argument('--csv-dir', default='data/sessions',
                        help='Directory containing session CSVs to train from (default: data/sessions)')
    parser.add_argument('--min-csv', type=int, default=20,
                        help='Minimum number of CSV files required in --csv-dir to start training (default: 20)')

    return parser.parse_args()

def collect_training_data(serial_port):
    """Collect training data with labeled emotions using timed protocol."""
    print("Starting training data collection with timed protocol...")
    print(f"Protocol: {config.SAMPLING_DURATION}s sampling + {config.REST_DURATION}s rest per emotion")
    print(f"Rounds per emotion: {config.TRAINING_ROUNDS}")

    training_data = []
    labels = []

    with DataAcquisition(serial_port) as daq:
        daq.start_streaming()

        for round_num in range(config.TRAINING_ROUNDS):
            print(f"\n--- Round {round_num + 1}/{config.TRAINING_ROUNDS} ---")

            for emotion in config.CLASSES:
                # Show stimulus phase
                print(f"🎯 STIMULUS: Show '{emotion}' stimulus now!")
                print(f"⏱️  Sampling for {config.SAMPLING_DURATION} seconds...")

                # Collect data during stimulus
                data = daq.collect_data(config.SAMPLING_DURATION)
                training_data.extend(data)
                labels.extend([emotion] * len(data))

                print(f"✅ Completed sampling for '{emotion}'")

                # Rest phase (except after last emotion in last round)
                if not (round_num == config.TRAINING_ROUNDS - 1 and emotion == config.CLASSES[-1]):
                    print(f"😌 REST: {config.REST_DURATION} seconds rest period...")
                    time.sleep(config.REST_DURATION)
                    print("⏱️  Rest completed. Prepare next stimulus.")

    print("\n🎉 Training data collection completed!")
    print(f"Total data points collected: {len(training_data)}")
    print(f"Labels distribution: {np.unique(labels, return_counts=True)}")

    return np.array(training_data), np.array(labels)

def train_model(training_data, labels, model_type, model_path):
    """Train and evaluate the emotion classification model."""
    print("Preprocessing training data...")

    preprocessor = Preprocessor()
    features_list = []

    # Process each data window
    for timestamp, eeg_data in training_data:
        features = preprocessor.preprocess_window(eeg_data)
        features_list.append(features)

    features = np.array(features_list)

    print("Training model...")

    classifier = EmotionClassifier(model_type)
    X_test, y_test, y_pred, y_pred_proba = classifier.train_model(features, labels)

    # Evaluate model
    results = classifier.evaluate_model(y_test, y_pred, y_pred_proba)

    print("\nModel Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.2f}")
    print(f"Precision: {results['precision']:.2f}")
    print(f"Recall: {results['recall']:.2f}")
    print(f"F1-Score: {results['f1_score']:.2f}")

    # Display confusion matrix
    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print("Predicted →")
    print("Actual ↓")
    print(f"{'':<12} {'  '.join(f'{cls:<8}' for cls in config.CLASSES)}")
    for i, true_cls in enumerate(config.CLASSES):
        print(f"{true_cls:<12} {'  '.join(f'{cm[i,j]:<8}' for j in range(len(config.CLASSES)))}")

    # Plot and save confusion matrix
    classifier.plot_confusion_matrix(cm, save_path='models/confusion_matrix.png')

    # Cross-validation
    cv_scores = classifier.cross_validate(features, labels)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    # Save model
    classifier.save_model(model_path)

    return results

def load_csv_training_data(csv_path):
    """Load training data from CSV file."""
    try:
        import pandas as pd
        print(f"Loading training data from {csv_path}...")

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples from CSV")

        # Extract features and labels
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        features = df[feature_cols].values
        labels = df['label'].values

        # Convert to expected format (list of tuples with dummy timestamps)
        training_data = [(i * 0.1, features[i]) for i in range(len(features))]

        return np.array(training_data), np.array(labels)

    except Exception as e:
        print(f"Failed to load CSV data: {e}")
        return None, None

def handle_training(args):
    """Handle training mode."""
    try:
        # Branch 1: Train from a directory of pre-collected session CSVs
        if hasattr(args, 'csv_dir') and args.csv_dir:
            csv_dir = args.csv_dir
            if not os.path.isdir(csv_dir):
                print(f"CSV dir not found: {csv_dir}")
                return

            csv_files = sorted(
                [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]
            )
            print(f"Found {len(csv_files)} CSV files in {csv_dir}")
            if len(csv_files) < getattr(args, 'min_csv', 20):
                print(f"Need at least {args.min_csv} CSV files to train. Found {len(csv_files)}. Exiting.")
                return

            # Aggregate features and labels
            try:
                import pandas as pd
            except Exception as e:
                print(f"Pandas required to load CSVs: {e}")
                sys.exit(1)

            dfs = []
            for path in csv_files:
                try:
                    df = pd.read_csv(path)
                    dfs.append(df)
                except Exception as e:
                    print(f"Skipping {path}: {e}")

            if not dfs:
                print("No readable CSVs in directory. Exiting.")
                return

            df_all = pd.concat(dfs, ignore_index=True)
            feature_cols = [c for c in df_all.columns if c.startswith('feature_')]
            if not feature_cols or 'label' not in df_all.columns:
                print("CSV files missing feature_ columns or label column.")
                return

            features = df_all[feature_cols].values
            labels = df_all['label'].values

            # Train model from aggregated features
            print("Training model from session CSV features...")
            classifier = EmotionClassifier(args.model_type)
            X_test, y_test, y_pred, y_pred_proba = classifier.train_model(features, labels)

            # Evaluate and report
            results = classifier.evaluate_model(y_test, y_pred, y_pred_proba)
            print("\nModel Evaluation Results (sessions):")
            print(f"Accuracy: {results['accuracy']:.2f}")
            print(f"Precision: {results['precision']:.2f}")
            print(f"Recall: {results['recall']:.2f}")
            print(f"F1-Score: {results['f1_score']:.2f}")

            # Save confusion matrix
            cm_path = 'models/confusion_matrix_sessions.png'
            classifier.plot_confusion_matrix(results['confusion_matrix'], save_path=cm_path)
            print(f"Confusion matrix saved to {cm_path}")

            # Save model
            classifier.save_model(args.model)
            print("Training from sessions completed successfully!")
            return

        # Branch 2: Train from a single CSV if provided
        training_data, labels = None, None
        if hasattr(args, 'csv') and args.csv:
            training_data, labels = load_csv_training_data(args.csv)
        else:
            # Collect training data using timed protocol
            training_data, labels = collect_training_data(args.port)

        if training_data is None or len(training_data) == 0:
            print("No training data available. Exiting.")
            return

        # Train model (online collection path)
        results = train_model(training_data, labels, args.model_type, args.model)

        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

def handle_realtime(args):
    """Handle real-time classification mode."""
    try:
        print("Starting real-time emotion classification...")
        run_real_time_classification(args.model, args.port)
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        print("Please train a model first using --mode train")
        sys.exit(1)
    except Exception as e:
        print(f"Real-time classification failed: {e}")
        sys.exit(1)

def handle_gui(args):
    """Handle GUI mode."""
    try:
        print("Launching GUI for EEG acquisition, FFT, and protocol-driven collection...")
        # run_gui handles QApplication lifecycle and BoardShim session
        run_gui(args.port)
    except Exception as e:
        print(f"GUI failed: {e}")
        sys.exit(1)

def main():
    """Main function."""
    args = parse_arguments()

    print("EEG Emotion Classification System")
    print("=" * 40)

    if args.mode == 'train':
        handle_training(args)
    elif args.mode == 'realtime':
        handle_realtime(args)
    elif args.mode == 'gui':
        handle_gui(args)

if __name__ == "__main__":
    main()