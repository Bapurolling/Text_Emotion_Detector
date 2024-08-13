# Text Emotion Detector
This repository contains the end-to-end implementation of a Text Emotion Detector using BERT, Docker, Jenkins, Streamlit, and AWS for deployment.

## Overview
The Text Emotion Detector classifies input text into six predefined emotion categories: anger, fear, joy, love, sadness, and surprise. The project follows modern development practices, incorporating Continuous Integration/Continuous Deployment (CI/CD) pipelines and cloud deployment.

## Features

- **Emotion Detection**: Accurately predicts emotions from text inputs.
- **Streamlit UI**: A user-friendly interface for inputting text and viewing predictions.
- **Dockerized Deployment**: Fully containerized using Docker for easy deployment.
- **CI/CD Pipeline**: Integrated Jenkins pipeline for continuous integration and deployment.
- **AWS Deployment**: Deployed on an AWS EC2 instance for scalable cloud access.

## Dataset

The model was trained on the [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?select=train.txt) available on Kaggle. The dataset includes 16,000 training samples, 2,000 validation samples, and 2,000 test samples, with the following emotion labels:
- Anger
- Fear
- Joy
- Love
- Sadness
- Surprise

### Steps to Run the Project
#### Clone the repository:
```bash
git clone https://github.com/Bapurolling/Text_Emotion_Detector.git
cd Text_Emotion_Detector
```

#### Build and Run the Docker Container:
```bash
docker build -t text_emotion_detector .
docker run -p 8501:8501 text_emotion_detector
```
#### Access the App:
Open a browser and go to http://localhost:8501 to access the Streamlit app.
### Alternate Way: Directly Pull the Docker Image from Docker Hub
You can skip the above steps and directly pull and run the pre-built Docker image from Docker Hub:

#### Pull the Docker Image

```bash
docker pull bapurolling/end-to-end:latest
```
#### Run the Docker Container

```bash

docker run -p 8501:8501 bapurolling/end-to-end:latest
```
#### Access the App
Open your web browser and go to http://localhost:8501 to use the Text Emotion Detector.
## User Interface

Here are some screenshots of the Text Emotion Detector application:

### Main UI
![Main UI](screenshots/ui_main.png)

### Prediction Result
![Prediction Result](screenshots/ui_prediction.png)


## CI/CD Pipeline
The project includes a Jenkins pipeline that automatically pulls the latest Docker image from Docker Hub, stops any running container, and deploys the new one. The pipeline is triggered by any push to the GitHub repository.

## Deployment on AWS
The project is deployed on an AWS EC2 instance to provide scalable and reliable access to the application.

## Conclusion
This project demonstrates an end-to-end machine learning application, including data preparation, model training, Dockerization, and CI/CD deployment. It's a comprehensive solution for detecting emotions from text inputs, with an easy-to-use web interface.
