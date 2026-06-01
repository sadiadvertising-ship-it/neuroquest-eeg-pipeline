#!/usr/bin/env python3
"""
Test script to verify all modules can be imported and basic functionality works
"""

def test_imports():
    """Test importing all modules."""
    try:
        import config
        print("✓ config imported successfully")

        import data_acquisition
        print("✓ data_acquisition imported successfully")

        import preprocessing
        print("✓ preprocessing imported successfully")

        import model_training
        print("✓ model_training imported successfully")

        import real_time_classification
        print("✓ real_time_classification imported successfully")

        import utils
        print("✓ utils imported successfully")

        import main
        print("✓ main imported successfully")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key classes."""
    try:
        from preprocessing import Preprocessor
        from model_training import EmotionClassifier
        from utils import Logger

        # Test preprocessor
        prep = Preprocessor()
        print("✓ Preprocessor initialized")

        # Test classifier (without loading model)
        clf = EmotionClassifier()
        print("✓ EmotionClassifier initialized")

        # Test logger
        logger = Logger()
        print("✓ Logger initialized")

        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing EEG Emotion Classification System")
    print("=" * 50)

    print("\n1. Testing imports...")
    imports_ok = test_imports()

    print("\n2. Testing basic functionality...")
    functionality_ok = test_basic_functionality()

    print("\n" + "=" * 50)
    if imports_ok and functionality_ok:
        print("✓ All tests passed! The system is ready for use.")
        print("\nNext steps:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Connect OpenBCI Ganglion board")
        print("- Run training: python main.py --mode train --port /dev/cu.usbmodem11")
        print("- Run real-time: python main.py --mode realtime --model models/emotion_classifier.pkl")
    else:
        print("✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()