# Hand Gesture Recognition System ✋

A real-time hand gesture recognition system built using Python, OpenCV, and TensorFlow (CNN). The project captures hand gesture images through a webcam, trains a deep learning model on a custom dataset, and performs live gesture prediction.

## Gestures Supported
- Fist
- Palm
- Thumbs Up

## Project Structure
hand-gesture-recognition/
├── dataset/
│   ├── fist/
│   ├── palm/
│   └── thumbs_up/
├── train.py
├── app.py
├── model.h5
├── requirements.txt
└── README.md

## Technologies Used
- Python
- OpenCV
- TensorFlow / Keras
- NumPy

## Working
1. Collect hand gesture images using a webcam.
2. Train a CNN model on the collected dataset.
3. Save the trained model as model.h5.
4. Use the trained model for real-time gesture recognition via webcam.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Train the model:
   python train.py
3. Run real-time detection:
   python app.py

## Future Improvements
- Add more hand gestures
- Improve accuracy with better data collection
- Use MediaPipe hand landmarks for robustness

## Author
Arya
