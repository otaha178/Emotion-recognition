# Project Name : Emotion-recognition
# Table of Content :
1.[Description](#p1)

2.[Installations](#p2)

3.[Usage](#p3)

4.[Dataset](#p4)



![](https://github.com/omar178/Emotion-recognition/blob/master/emotions/Happy.PNG)
![](https://github.com/omar178/Emotion-recognition/blob/master/emotions/angry.PNG)
![](https://github.com/omar178/Emotion-recognition/blob/master/emotions/neural.PNG)



<a id="p1"></a> 
# Description:Dataset
However, before we get started, it’s important to understand that as humans, our emotions are in
a constant fluid state. We are never 100% happy or 100% sad. Instead, our emotions mix together.
When experiencing “surprise” we might also be feeling “happiness” (such as a surprise birthday
party) or “fright” (if the surprise is not a welcome one). And even during the “scared” emotion, we
might feel hints of “anger” as well.
When studying emotion recognition, it’s important to not focus on a single class label (as we
sometimes do in other classification problems). Instead, it’s much more advantageous for us to
look at the probability of each emotion and characterize the distribution.
## What does Emotion Recognition mean?

Emotion recognition is a technique used in software that allows a program to "read" the emotions on a human face using advanced image processing. Companies have been experimenting with combining sophisticated algorithms with image processing techniques that have emerged in the past ten years to understand more about what an image or a video of a person's face tells us about how he/she is feeling and not just that but also showing the probabilities of mixed emotions a face could has.

<a id="p2"></a> 
# Installations:
-keras
-imutils
-cv2
-numpy

<a id="p3"></a> 
# Usage:
[ Demo]
-run real_time_video.py
## Train
-run train_emotion_classifier


<a id="p4"></a> 
# Dataset:

I have used [this](https://www.kaggle.com/c/3364/download-all)
Download it and put the csv in fer2013/fer2013/

# Credits
[[This work is inspired from this fascinating [work](https://github.com/oarriaga/face_classification) ]]

