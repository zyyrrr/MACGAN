# MACGAN
Realization of paper: "MACGAN : Bypass Traffic Anomaly Detector and Maintain Attack Characteristics"

## Introduction
The idea is to use a generative adversarial network (GAN) based algorithm to generate adversarial malware examples, which are able to bypass the existing advanced anomaly detection algorithm.

## Dependencies
 ```pytorch 1.0``` ```keras 2.2.2``` 
 
 ## Datasets
 Two data sets, Kitsune and IDS 2017, are used in this paper to evaluate the performance of our scheme.
 
 ## Demo Code
 The MACGAN framework consists of two parts. <br>
 The first part is used to automatically extract the attack signature field. A demo script is provided in example_mirai.py.
 ```
%run example_mirai.py
```
Then, the learning function of GAN in the second part is used to bypass the anomaly detection. A demo script is provided in wgan_mirai.py.
 ```
%run wgan_mirai.py
```
