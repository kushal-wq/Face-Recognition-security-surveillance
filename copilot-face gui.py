import sys
import os
import cv2
import pickle
import numpy as np
import serial
import time
import whisper  # OpenAI Whisper
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QMessageBox, QProgressBar, QTabWidget, QFrame, QGridLayout, QSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import pyaudio
import wave
import uuid  # For generating unique filenames


class SpeechRecognitionThread(QThread):
    """Thread for handling speech recognition in the background using OpenAI Whisper."""
    command_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.speech_enabled = True
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.model = whisper.load_model("base")  # Load the Whisper base model

    def run(self):
        """Continuously listen for speech commands."""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()

            while self.speech_enabled:
                time.sleep(0.1)  # Sleep to avoid busy-waiting

        except Exception as e:
            print(f"Error in speech recognition thread: {e}")
        finally:
            self.cleanup()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        try:
            # Create a unique temporary file name
            temp_audio_file = os.path.join(os.getcwd(), f"temp_audio_{uuid.uuid4()}.wav")
            print(f"Temporary file path: {temp_audio_file}")  # Debugging: Print the file path

            # Save the recorded audio to the temporary file
            with wave.open(temp_audio_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(in_data)

            # Verify the file was created
            if not os.path.exists(temp_audio_file):
                print(f"Error: File {temp_audio_file} was not created.")
                return (in_data, pyaudio.paContinue)

            # Transcribe the audio using Whisper
            result = self.model.transcribe(temp_audio_file)
            text = result["text"].strip().lower()

            if text:
                self.command_signal.emit(text)

            # Clean up the temporary audio file
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
                print(f"Deleted temporary file: {temp_audio_file}")  # Debugging: Confirm deletion
            else:
                print(f"Error: File {temp_audio_file} does not exist for deletion.")

        except Exception as e:
            print(f"Error in audio callback: {e}")

        return (in_data, pyaudio.paContinue)

    def cleanup(self):
        """Clean up resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def stop(self):
        """Stop the speech recognition thread."""
        self.speech_enabled = False
        self.cleanup()
        self.quit()


class CombinedRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Recognition and Tracking System")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize Arduino connection
        self.arduino = None
        self.arduino_connected = False
        self.initialize_arduino()

        # Initialize face recognition and tracking
        self.dataset_folder = 'dataset'
        self.recognizer_file = 'face_recognizer.yml'
        self.names_file = 'names.pkl'
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cap = None
        self.face_timer = QTimer()
        self.face_timer.timeout.connect(self.update_face_frame)
        self.names = {}
        self.tracking_enabled = False
        self.target_name = None
        self.target_face_center = None

        # Initialize speech recognition
        self.speech_thread = SpeechRecognitionThread()
        self.speech_thread.command_signal.connect(self.handle_speech_command)

        # Create dataset folder if it doesn't exist
        os.makedirs(self.dataset_folder, exist_ok=True)

        # Load names if the file exists
        if os.path.exists(self.names_file):
            with open(self.names_file, 'rb') as f:
                self.names = pickle.load(f)

        # Load recognizer if it exists
        if os.path.exists(self.recognizer_file):
            self.recognizer.read(self.recognizer_file)
            print("Recognizer model loaded successfully.")
        else:
            print("Recognizer model not found. Please train the model first.")

        # Apply styles and initialize UI
        self.apply_styles()
        self.initUI()

    def initialize_arduino(self):
        try:
            self.arduino = serial.Serial('COM8', 9600, timeout=1)  # Change COM port as needed
            time.sleep(2)  # Wait for connection to establish
            self.arduino_connected = True
            print("Arduino connected successfully.")
        except Exception as e:
            print(f"Arduino not connected: {e}")
            self.arduino_connected = False

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: Arial;
                font-size: 14px;
                background-color: #f0f0f0;
            }
            QPushButton {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                min-width: 130px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton[class="stop"] {
                background-color: #f44336;
            }
            QPushButton[class="stop"]:hover {
                background-color: #da190b;
            }
            QLabel {
                color: #333333;
                font-size: 16px;
            }
            QProgressBar {
                border: 2px solid #dddddd;
                border-radius: 5px;
                text-align: center;
                font-size: 14px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QTabWidget::pane {
                border: 1px solid #dddddd;
                border-radius: 5px;
            }
            QTabBar::tab {
                padding: 10px 25px;
                margin: 2px;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background: #4CAF50;
                color: white;
            }
        """)

    def initUI(self):
        main_layout = QVBoxLayout()

        # Header
        header = QLabel("<h1 style='color: #4CAF50;'>Advanced Recognition and Tracking System</h1>")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Tabs
        self.tabs = QTabWidget()

        # Create tabs
        self.create_data_collection_tab()
        self.create_face_recognition_tab()

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def create_data_collection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("<h2>Face Data Collection & Training</h2>")
        title.setAlignment(Qt.AlignCenter)

        # Input group
        input_group = QFrame()
        input_group.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        input_layout = QGridLayout()

        name_label = QLabel("Person's Name:")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name for training")
        
        samples_label = QLabel("Number of Samples:")
        self.samples_input = QSpinBox()
        self.samples_input.setRange(20, 100)
        self.samples_input.setValue(30)

        input_layout.addWidget(name_label, 0, 0)
        input_layout.addWidget(self.name_input, 0, 1)
        input_layout.addWidget(samples_label, 1, 0)
        input_layout.addWidget(self.samples_input, 1, 1)
        input_group.setLayout(input_layout)

        # Video display
        self.training_label = QLabel()
        self.training_label.setFixedSize(640, 480)
        self.training_label.setStyleSheet("border: 2px solid #dddddd; border-radius: 5px;")

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Buttons
        button_layout = QHBoxLayout()
        collect_btn = QPushButton("Collect Face Data")
        collect_btn.clicked.connect(self.start_collection)
        
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self.train_model)

        button_layout.addWidget(collect_btn)
        button_layout.addWidget(train_btn)

        layout.addWidget(title)
        layout.addWidget(input_group)
        layout.addWidget(self.training_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.progress_bar)
        layout.addLayout(button_layout)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Data Collection")

    def create_face_recognition_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("<h2>Face Recognition & Tracking</h2>")
        title.setAlignment(Qt.AlignCenter)

        # Tracking controls
        tracking_frame = QFrame()
        tracking_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        tracking_layout = QHBoxLayout()

        self.target_combo = QComboBox()
        self.target_combo.addItem("Select Target")
        self.update_target_combo()  # Populate combo box with dataset names
        tracking_layout.addWidget(QLabel("Track Target:"))
        tracking_layout.addWidget(self.target_combo)

        self.track_button = QPushButton("Start Tracking")
        self.track_button.clicked.connect(self.toggle_tracking)
        tracking_layout.addWidget(self.track_button)

        tracking_frame.setLayout(tracking_layout)

        # Video display
        self.face_video_label = QLabel()
        self.face_video_label.setFixedSize(640, 480)
        self.face_video_label.setStyleSheet("border: 2px solid #dddddd; border-radius: 5px;")

        # Buttons
        button_layout = QHBoxLayout()
        start_btn = QPushButton("Start Recognition")
        start_btn.clicked.connect(self.start_face_recognition)
        
        stop_btn = QPushButton("Stop Recognition")
        stop_btn.setProperty("class", "stop")
        stop_btn.clicked.connect(self.stop_face_recognition)

        # Speech control button
        self.speech_btn = QPushButton("Enable Speech Control")
        self.speech_btn.clicked.connect(self.toggle_speech_recognition)
        button_layout.addWidget(self.speech_btn)

        button_layout.addWidget(start_btn)
        button_layout.addWidget(stop_btn)

        layout.addWidget(title)
        layout.addWidget(tracking_frame)
        layout.addWidget(self.face_video_label, alignment=Qt.AlignCenter)
        layout.addLayout(button_layout)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Face Recognition")

    def send_coordinates_to_arduino(self, x_center=None, command=None):
        if self.arduino_connected:
            try:
                if x_center is not None:
                    # Send X-coordinate for face tracking
                    coordinates = f"POS:{x_center}\n"  # Use newline as a delimiter
                    self.arduino.write(coordinates.encode())
                    print(f"Sent servo position: {x_center}")
                elif command is not None:
                    # Send speech command
                    self.arduino.write(f"{command}\n".encode())
                    print(f"Sent speech command: {command}")
            except Exception as e:
                print(f"Failed to send data to Arduino: {e}")
        else:
            print("Arduino not connected. Unable to send data.")

    def toggle_tracking(self):
        if not self.tracking_enabled:
            selected_target = self.target_combo.currentText()
            if selected_target == "Select Target" or selected_target not in self.names.values():
                QMessageBox.warning(self, "Warning", "Please select a valid target!")
                return
            
            self.tracking_enabled = True
            self.target_name = selected_target
            self.track_button.setText("Stop Tracking")
            self.track_button.setStyleSheet("background-color: #f44336;")
        else:
            self.tracking_enabled = False
            self.target_name = None
            self.track_button.setText("Start Tracking")
            self.track_button.setStyleSheet("")

    def start_face_recognition(self):
        if not os.path.exists(self.recognizer_file):
            QMessageBox.warning(self, "Warning", "Model not trained! Please train the model first.")
            return

        try:
            self.face_cap = cv2.VideoCapture(0)
            if not self.face_cap.isOpened():
                raise Exception("Could not open video device.")
            self.face_timer.start(int(1000 / 30))  # 30 FPS
            self.update_target_combo()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start face recognition: {e}")

    def stop_face_recognition(self):
        self.face_timer.stop()
        if self.face_cap:
            self.face_cap.release()

    def update_face_frame(self):
        if not self.face_cap:
            return

        ret, frame = self.face_cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (100, 100))

            # Check if the recognizer has been trained
            if not os.path.exists(self.recognizer_file):
                cv2.putText(frame, "Model not trained", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                continue

            # Predict the face
            try:
                label_id, confidence = self.recognizer.predict(face_img)
                print(f"Predicted ID: {label_id}, Confidence: {confidence}")
                if confidence < 100:  # Confidence threshold
                    name = self.names.get(label_id, "Unknown")
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Track the target if enabled
                    if self.tracking_enabled and name == self.target_name:
                        face_center_x = x + w // 2
                        self.target_face_center = face_center_x
                        self.send_coordinates_to_arduino(x_center=face_center_x)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            except Exception as e:
                print(f"Error during prediction: {e}")

        # Convert frame to QPixmap
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        bytes_per_line = 3 * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.face_video_label.setPixmap(pixmap)

    def update_target_combo(self):
        """Populate the target combo box with names from the dataset folder."""
        self.target_combo.clear()
        self.target_combo.addItem("Select Target")
        
        if not os.path.exists(self.dataset_folder):
            return
        
        for person_name in os.listdir(self.dataset_folder):
            person_path = os.path.join(self.dataset_folder, person_name)
            if os.path.isdir(person_path):
                self.target_combo.addItem(person_name)

    def start_collection(self):
        """Start collecting face data for training."""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name for the person!")
            return

        num_samples = self.samples_input.value()
        person_folder = os.path.join(self.dataset_folder, name)
        os.makedirs(person_folder, exist_ok=True)

        try:
            self.face_cap = cv2.VideoCapture(0)
            if not self.face_cap.isOpened():
                raise Exception("Could not open video device.")

            self.progress_bar.setMaximum(num_samples)
            self.progress_bar.setValue(0)

            sample_count = 0
            while sample_count < num_samples:
                ret, frame = self.face_cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face_img = gray[y:y + h, x:x + w]
                    face_img = cv2.resize(face_img, (100, 100))

                    # Save the face image
                    face_filename = os.path.join(person_folder, f"{name}_{sample_count}.jpg")
                    cv2.imwrite(face_filename, face_img)

                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Sample: {sample_count + 1}/{num_samples}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    sample_count += 1
                    self.progress_bar.setValue(sample_count)

                # Display the frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = frame.shape
                bytes_per_line = 3 * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.training_label.setPixmap(pixmap)

                # Wait for a short period to avoid capturing too many similar images
                time.sleep(0.5)

            QMessageBox.information(self, "Success", f"Collected {num_samples} samples for {name}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to collect face data: {e}")
        finally:
            if self.face_cap:
                self.face_cap.release()

    def train_model(self):
        """Train the face recognition model using collected data."""
        try:
            faces = []
            labels = []
            label_ids = {}
            current_id = 0

            for root, dirs, files in os.walk(self.dataset_folder):
                for dir_name in dirs:
                    label_ids[dir_name] = current_id
                    current_id += 1

                    person_folder = os.path.join(root, dir_name)
                    for file_name in os.listdir(person_folder):
                        if file_name.endswith(".jpg"):
                            face_img = cv2.imread(os.path.join(person_folder, file_name), cv2.IMREAD_GRAYSCALE)
                            faces.append(face_img)
                            labels.append(label_ids[dir_name])

            if not faces:
                QMessageBox.warning(self, "Warning", "No face data found! Please collect data first.")
                return

            # Train the recognizer
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save(self.recognizer_file)

            # Save the label IDs
            with open(self.names_file, 'wb') as f:
                pickle.dump(label_ids, f)

            QMessageBox.information(self, "Success", "Model trained successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to train model: {e}")

    def toggle_speech_recognition(self):
        """Toggle speech recognition on or off."""
        if self.speech_thread.isRunning():
            self.speech_thread.stop()
            self.speech_btn.setText("Enable Speech Control")
        else:
            self.speech_thread.start()
            self.speech_btn.setText("Disable Speech Control")

    def handle_speech_command(self, command):
        """Handle speech commands."""
        print(f"Received speech command: {command}")
        self.send_coordinates_to_arduino(command=command)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CombinedRecognitionApp()
    window.show()
    sys.exit(app.exec_())