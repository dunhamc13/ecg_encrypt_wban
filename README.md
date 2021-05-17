# ECG-based Encryption using MFBSG Algorithm to Authenticate WBAN Sensors
This is a collaborative project for:
Christian Dunham, Robin Friedmann, Julius Marx, Niklas Von Zwehl, and Maurice Go ̈rke
from the University of Washington Bothell and Hochschule Mainz

# Abstract
Quantum computing presents a challenge to encryption utilized to secure Wireless Body Area Network (WBAN) sensors. The Multiple Fiducial-Points based Binary Sequence Generation (MFBSG algorithm) has been utilized to generate 128-bit ECG cryptographic keys to provide secure data transmission between (WBAN) sensors. The previous work on ECG biometric encryption created random Binary Sequences (BS) through measuring time between different fiducial points. This research examines if additional feature selection or salt generation can be utilized to prepare ECG-based MFBSG encryption to meet the needs driven by quantum computing in the future. Specifically, this innovative research aims to reduce the vulnerability presented by default passwords by implementing an ECG-based MFBSG encrypted password.
## Runnning the Application
python ECG_BinarySeq.py
   (dependencies: tensorflow -> if on mac using python
                                -> pip install tensorflow
                             -> if on mac using python3
                                -> pip3 install tensorflow
### Runtime: PythonXX and .NET 5.0

### File Structure: 
ECG_BinarySeq.py 
   : Takes a local csv file with 140 columns of ECG data and computers the 
     fiducial points PQRST.
     Creates a file and writes a binary sequence based on fiducial points.
ecg.csv dataset from https://www.kaggle.com/devavratatripathy/ecg-dataset
     
### Configurations
Command Line Arguments:`

### Current Description of Implementation
XYz....

### Current Description of future Implementation
1) Need to implement algorithm to abastract fiducial points
2) Need to determine fiducial vectors and how to create binary sequence
3) Need to write that BS to a file

## Performance Discussions:
TBD...

