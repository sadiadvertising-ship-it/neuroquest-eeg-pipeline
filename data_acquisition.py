from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
import config


class DataAcquisition:
    """BrainFlow wrapper. EEG always comes out as [samples, channels] in uV."""

    def __init__(self, serial_port=config.SERIAL_PORT, board_id=config.BOARD_ID):
        self.serial_port = serial_port
        self.board_id = board_id
        self.board = None
        self.is_streaming = False
        # Row 0 of a BrainFlow packet is the package counter, the EEG rows come from the board description
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)[:config.CHANNELS]
        self.package_channel = BoardShim.get_package_num_channel(board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.firmware = None  # Ganglion firmware as BrainFlow decodes it (2 or 3), known after connecting
        self.ganglion = board_id in (BoardIds.GANGLION_BOARD.value, BoardIds.GANGLION_NATIVE_BOARD.value)

    def initialize_board(self):
        """Initialize the OpenBCI board connection."""
        try:
            params = BrainFlowInputParams()
            params.serial_port = self.serial_port
            if self.ganglion:
                # BrainFlow 5.12.1 detects the firmware by itself and ignores this; newer releases assume
                # firmware 3 unless told otherwise, which would decode a firmware 2 board wrongly
                params.other_info = 'fw:auto'
            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()
            if self.ganglion:
                try:
                    self.firmware = int(self.board.config_board('get_firmware_version'))
                except Exception:
                    self.firmware = None  # the analysis then guesses it from the packet ids
            print(f"Board initialized on port {self.serial_port}"
                  + (f" (Ganglion firmware {self.firmware})" if self.firmware else ""))
        except Exception as e:
            raise ConnectionError(f"Failed to initialize board: {e}")

    def start_streaming(self):
        """Start data streaming."""
        if self.board is None:
            raise RuntimeError("Board not initialized")
        self.board.start_stream(450000)
        self.is_streaming = True
        print("Data streaming started")

    def stop_streaming(self):
        """Stop data streaming."""
        if self.board and self.is_streaming:
            self.is_streaming = False
            self.board.stop_stream()
            print("Data streaming stopped")

    def read_new(self):
        """
        Samples that arrived since the last call (they are removed from the board buffer):
        (timestamps, package counter, [samples, channels]).
        """
        data = self.board.get_board_data()
        return data[self.timestamp_channel], data[self.package_channel], data[self.eeg_channels].T

    def close(self):
        """Close the board session."""
        if self.board and self.board.is_prepared():
            self.board.release_session()
            print("Board session released")

    def __enter__(self):
        try:
            self.initialize_board()
        except BaseException:
            self.close()  # Ctrl+C or an error halfway through prepare_session
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.stop_streaming()
        finally:
            self.close()
