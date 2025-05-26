# Webcam-Based Heart Rate Detection Using Deep Learning

## Overview

This project implements a non-contact heart rate detection system using a standard webcam and deep learning techniques. By leveraging facial video data, the model estimates heart rate (BPM) without requiring any physical sensors. The system is trained and evaluated on the UBFC-RPPG dataset, utilizing a CNN-LSTM architecture to process temporal facial features extracted from video frames.

---

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Limitations & Ethical Considerations](#limitations--ethical-considerations)
- [References](#references)
- [License](#license)
- [Contact](#contact)

---

## Features

- Non-contact heart rate estimation using webcam video
- Face detection and region-of-interest extraction
- Feature extraction from facial video (mean green channel intensity)
- Deep learning model (CNN-LSTM) for temporal signal processing
- End-to-end pipeline: video input → face detection → feature extraction → heart rate prediction

---

## Demo


![Demo Screenshot](https://github.com/Kali414/DRDO_PROJECT/blob/main/static/DRDO.webp)


---

## Project Structure


- **Feature/**: Notebooks for feature extraction and exploration.
- **Notebook/**: Model training/testing notebooks and preprocessed data files.
- **static/**, **templates/**: Web application assets (HTML, CSS, JS).
- **app.py**: Main application script (Flask app).
- **bpm_model.keras**, **bpm_model_v2.keras**: Saved Keras models.
- **formula_bpm_measure.py**: Heart rate calculation utilities.
- **haar_cascade_frontal_face.xml**: Face detection model.
- **requirements.txt**: Python dependencies.

---

## Dataset

- **UBFC-RPPG Dataset**: Public dataset for remote photoplethysmography (rPPG) and heart rate estimation.
- Contains facial videos and synchronized ground truth heart rate (from pulse oximeter).
- [Dataset details and terms](https://sites.google.com/view/ybenezeth/ubfcrppg)
- **Note**: For subjects 21 and 27, facial images must not be published or shared.

**Ground Truth Format:**
- *Dataset 1*: `.xmp` files (Timestep, HR, SpO2, PPG)
- *Dataset 2*: `ground_truth.txt` (PPG, HR, Timestep)

 *Download the UBFC-RPPG dataset (see [Dataset](https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk?usp=drive_link)) .*

---

## Methodology

1. **Data Preprocessing**:
   - Load video frames and corresponding ground truth HR.
   - Detect face region using Haar cascade.
   - Extract mean green channel intensity from the face region for each frame.
   - Segment features into windows of 24 frames.

2. **Model Training**:
   - Input: Sequence of mean green values (shape: 24, 1).
   - Output: Corresponding HR values (shape: 24).
   - Model: CNN-LSTM (see below).

3. **Prediction**:
   - For a new webcam video, repeat face detection and feature extraction.
   - Model predicts heart rate for each window.

---

## Model Architecture

- **Input**: Sequence of 24 mean green channel values.
- **CNN Layers**: 1D convolution + max pooling for local feature extraction.
- **LSTM Layers**: Capture temporal dependencies in the signal.
- **Dense Layers**: Regression output for BPM.
- **Regularization**: Dropout and batch normalization.
- **Loss**: Mean Absolute Error (MAE).

---

## Installation

1. **Clone the repository**
```
git clone https://github.com/Kali414/DRDO_PROJECT.git
cd DRDO_PROJECT
```

2. **Install dependencies:**
```
pip install -r requirements.txt
```

3. **Run Web Application:**
```
python app.py
```

---

## Limitations & Ethical Considerations

- Accuracy may vary with lighting, movement, and camera quality.
- Application can only detect HR for 1 people at a time
- Sudden change can cause incorrect HR calculation. In the most case, HR can be correctly detected after 10 seconds being stable infront of the camera
- Dataset restrictions: Do not publish or share facial images from restricted subjects.
---


## References
- Real Time Heart Rate Monitoring From Facial RGB Color Video Using Webcam by H. Rahman, M.U. Ahmed, S. Begum, P. Funk
- Remote Monitoring of Heart Rate using Multispectral Imaging in Group 2, 18-551, Spring 2015 by Michael Kellman Carnegie (Mellon University), Sophia Zikanova (Carnegie Mellon University) and Bryan Phipps (Carnegie Mellon University)
- Non-contact, automated cardiac pulse measurements using video imaging and blind source separation by Ming-Zher Poh, Daniel J. McDuff, and Rosalind W. Picard
- Camera-based Heart Rate Monitoring by Janus Nørtoft Jensen and Morten Hannemose
---

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.


---

## Contact

For questions or contributions, contact: [kalicharansahoo91@gmail.com , sumandalai0509@gmail.com]

---


