# Emotion Detector

This Python script utilizes OpenCV and a pre-trained deep-learning model to detect and classify emotions in real time using your webcam.

## Prerequisites

- Python 3.x
- Libraries: `cv2`, `numpy`, `tensorflow`

## Setting Up a Virtual Environment
It's recommended to work within a virtual environment to manage your project dependencies. Here's how you can create and activate a virtual environment:
```bash
python -m venv myenv
```
- On Windows:
```cmd
myenv\Scripts\activate
```
- On macOS and Linux:
```bash
source myenv/bin/activate
```

## Install the libraries
```bash
pip install opencv-python numpy tensorflow
```
  
## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/prashantgarbuja/emotion-detector.git
   cd emotion-detector
   ```
2. **Download Pre-trained Model**:

   You'll need to download the pre-trained emotion detection [model](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5) and pre-trained [XML file](https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) for detecting frontal faces. The files are already downloaded and inserted in this repo. Make sure to place it in the same directory as the Python script.

4. **Run the Script**:

   Execute the Python script:

   ```bash
   python emotion_detector.py
   ```

   This will open your webcam and start detecting emotions in real-time.

   Press `q` to exit the program.

## Description

The script performs the following steps:

1. Loads a pre-trained emotion detection model.
2. Captures video from your webcam.
3. Detects faces in each frame using a Haar Cascade classifier.
4. For each detected face, it:
   - Resizes and preprocesses the region of interest.
   - Passes it through the model for emotion classification.
   - Draws a bounding box around the face with the predicted emotion label.

## Acknowledgements

- [Pre-trained Emotion Detection Model](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5)
- [Pre-trained XML File for detecting frontal faces](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5)

Feel free to contribute, report issues, or suggest improvements!
