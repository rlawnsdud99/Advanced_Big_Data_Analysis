# Project Ⅱ

## 1. Project Title
### Audio-based Emotion Recognition
To develop an audio-based emotion recognition framework that can convey realistic expressions for dynamic communication.

## 2. Project Introduction
### ■ Objective
The main goal of this project is to recognize emotions for conveying realistic expressions using audio. <br/>
The process involves preprocessing audio data to make it suitable for neural network training, and then using this training to classify emotions. <br/>
It can convey the user's intention and evolve into future work such as digital avatar animation capable of giving various facial expressions.

### ■ Motivation
Beyond simple communication like text or speech, next-generation communication systems capable of conveying rich prosody, expressiveness and identity are gaining attention. <br/>
Through this, users in next-generation digital environments like VR will be able to share and empathize with various emotions. <br/>
This project can also be utilized for medical purposes. <br/>
It is possible to develop an application that enables the free expression of thoughts for patients who find it challenging to convey their emotions. <br/>
Based on this project, it can be utilized in conjunction with research areas such as Computer Vision (CV) focused on generating images or videos of a user's face based on their emotional state, and Brain-Computer Interface (BCI) that directly decodes user intentions from brain signals. <br/>

## 3. Dataset Description
### ■ Background for dataset utilization
- Audio data consists of the x-axis representing time and the y-axis representing amplitude, with a certain period of time known as the frequency occurring between specific points and the next. This frequency can be expressed as hertz (Hz), indicating how many cycles occur in one second. Frequency is commonly used to represent the high and low pitches of sound, while amplitude denotes the loudness of the sound. As audio data is continuous, it needs to be discretized for training models. <br/>
- To convert audio data into discrete form, it undergoes two main processes: sampling and quantization. Sample rate refers to how many samples will be extracted and used per second during the sampling process. A higher sample rate allows for a representation closer to the original sound, but it comes with the drawback of requiring more memory space. Quantization is used to represent continuous information in a discrete form. Increasing the bit size in quantization results in information that closely resembles the original. <br/>
- Since audio information contains a mixture of various frequencies and holds extensive data, it is crucial to extract features that represent the distinctive characteristics of the audio rather than using the raw data directly. Data preprocessing and feature extraction are essential, as the method used to extract features from the data can have a significant impact on the performance of the model. Representative techniques include Mel-Spectrogram and MFCC (Mel-Frequency Cepstral Coefficients). <br/>
### ■ Information
- Number of sentence classes: 2 ('Kids are talking by the door', 'Dogs are sitting by the door')
- Number of audio data: 1050 (train : val : test = 630 : 210 : 210 = 6 : 2 : 2)
- The types of datasets provided: audio files (.wav)
- Number of emotional classes: 8
  - Neutral
  - Calm
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprised
- '.csv' files containing label information or for storing results have been uploaded together. <br/>

- Reference <br/>
S. R. Livingstone, and F. A. Russo, “The ryerson audio-visual database of emotional speech and song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English,” *PloSone*, Vol. 13, No. 5, 2017, pp. e0196391.

### ※ Contact Information
- Email: jiha_park@korea.ac.kr
