import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import config

class DataAcquisition:
    def __init__(self, serial_port=config.SERIAL_PORT, board_id=config.BOARD_ID):
        self.serial_port = serial_port
        self.board_id = board_id
        self.board = None
        self.is_streaming = False

    def initialize_board(self):
        """Initialize the OpenBCI board connection."""
        try:
            params = BrainFlowInputParams()
            params.serial_port = self.serial_port
            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()
            print(f"Board initialized on port {self.serial_port}")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize board: {e}")

    def start_streaming(self):
        """Start data streaming."""
        if self.board is None:
            raise RuntimeError("Board not initialized")
        self.board.start_stream()
        self.is_streaming = True
        print("Data streaming started")

    def stop_streaming(self):
        """Stop data streaming."""
        if self.board and self.is_streaming:
            self.board.stop_stream()
            self.is_streaming = False
            print("Data streaming stopped")

    def collect_data(self, duration_seconds):
        """Collect EEG data for a specified duration."""
        if not self.is_streaming:
            raise RuntimeError("Streaming not started")

        start_time = time.time()
        data_buffer = []

        while time.time() - start_time < duration_seconds:
            if self.board.get_board_data_count() > 0:
                data = self.board.get_board_data()
                # Extract EEG channels (first 4 for Ganglion)
                eeg_data = data[:config.CHANNELS]
                timestamp = time.time()
                data_buffer.append((timestamp, eeg_data))
            time.sleep(0.1)  # Small delay to avoid busy waiting

        return np.array(data_buffer)

    def get_current_data(self, num_samples=1):
        """Get current EEG data samples."""
        if not self.is_streaming:
            return None

        data = self.board.get_current_board_data(num_samples)
        if data.size > 0:
            return data[:config.CHANNELS]  # EEG channels only
        return None

    def close(self):
        """Close the board session."""
        if self.board:
            self.board.release_session()
            print("Board session released")

    def __enter__(self):
        self.initialize_board()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_streaming()
        self.close()