# Driver Drowsiness Detection

A real-time driver drowsiness detection system built with Streamlit and TensorFlow that monitors eye movements through webcam feed to detect drowsiness and alert the driver.

## 🚗 Overview

This application uses computer vision and deep learning to monitor a driver's eyes in real-time and detect signs of drowsiness. When drowsiness is detected, the system triggers visual and audio alerts to help prevent accidents caused by falling asleep while driving.

## ✨ Features

- **Real-time Eye Monitoring**: Uses webcam to continuously monitor driver's eyes
- **Deep Learning Detection**: Employs a trained CNN model to classify eye states (open/closed)
- **Instant Alerts**: Provides immediate visual warnings and audio alarms when drowsiness is detected
- **Web-based Interface**: Built with Streamlit for easy access through any web browser
- **Face & Eye Detection**: Uses Haar cascades for accurate face and eye region detection

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **Computer Vision**: OpenCV
- **Deep Learning**: TensorFlow/Keras
- **Real-time Streaming**: streamlit-webrtc
- **Audio Processing**: pygame
- **Video Processing**: av (PyAV)

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:Dhanush-M555/driver-drowsiness-detection-streamlit-app.git
   cd driver-drowsiness-detection-streamlit-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run Hello.py
   ```

4. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## 📱 How It Works

1. **Camera Access**: The app requests permission to access your webcam
2. **Face Detection**: Uses Haar cascade classifiers to detect faces in the video stream
3. **Eye Region Extraction**: Identifies and extracts left and right eye regions
4. **Drowsiness Classification**: The trained CNN model analyzes eye images to determine if eyes are open or closed
5. **Alert System**: When eyes remain closed for a threshold period (10 consecutive frames), the system:
   - Displays a "Drowsiness Detected" warning on screen
   - Shows a Streamlit warning message
   - Plays an audio alarm

## 🧠 Model Details

- **Model File**: `drowiness_new2.h5` (pre-trained CNN model)
- **Input Size**: 145x145 pixel eye images
- **Classes**: Open/Closed eye classification
- **Cascade Files**: 
  - `haarcascade_frontalface_default.xml` - Face detection
  - `haarcascade_lefteye_2splits.xml` - Left eye detection  
  - `haarcascade_righteye_2splits.xml` - Right eye detection

## 🎯 Use Cases

- **Personal Use**: Monitor your own alertness during long drives
- **Fleet Management**: Integrate into commercial vehicle monitoring systems
- **Research**: Study driver behavior and fatigue patterns
- **Safety Training**: Demonstrate the importance of staying alert while driving

## 🌐 Live Demo

[Driver Drowsiness Detection Website](https://driver-drowsiness-detection.streamlit.app/)


## ⚠️ Important Notes

- Ensure good lighting conditions for optimal face and eye detection
- The system requires clear visibility of both eyes
- Audio alerts may not work in all browser configurations due to autoplay policies
- This is a demonstration system and should not be relied upon as the sole safety measure while driving

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests to improve the system.
