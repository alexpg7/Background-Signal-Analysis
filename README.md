# 📉Background-Signal-Analysis
A project made to filter the true signal from the background of a detector.

This project goes through one of the most important topics of experimental physics: distinguishing the noise of a detector from the real detection. With the use of Machine Learning and Montecarlo methods, one can filter the noise to obtain the real data.

## ⚛️The experiment

The premise is simple: we have a detector of dimensions $1\times 1$ (adimensional) that records each point where a particle has collided. We know for sure that some of the detections may not correspond to the particles we really want to detect and they are considered noise. 

The goal is to classify each detection with two labels: **Background** and **Signal**. To do so, we can make use of 2 Montecarlo-generated datasets corresponding to each type of data to train a neural network to do the job.

## 📑Dataset

The 3 files used are simple data:

* [``dataset.txt``]() The detections we want to filter.
* [``mc_bkg.txt``]() The Montecarlo generated background sample.
* [``mc_signal.txt``]() The Montecarlo generated signal sample.

All of them are two columns corresponding to the $X$ and $Y$ coordinates, separated by a ``tab``. Example:
```file
     X             Y       
=======================
0.8110		-0.0370
0.2074		0.7868
0.8212		0.4156
```

So we will have to skip the first two rows when reading.

## ♟️Strategy

To accomplish our task, we will be using the ``scikit-learn`` library, which allows us to use ``MLPRegressor()`` as a simple Neural Network model.

In an ideal world, our Neural Network $N$ would work the following way:

```math
\begin{align}
\begin{pmatrix} & & \\ & N & \\ & & \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} = 1&\text{if (x, y) is Background}\\
\begin{pmatrix} & & \\ & N & \\ & & \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} = 0&\text{if (x, y) is Signal}
\end{align}
```

## 🔰Start the project

The first step is to import all the necessary packages.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy.stats import norm
from statistics import NormalDist
import plotly.express as px
from scipy.stats import gaussian_kde

#machine learning packages
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
```

## 👓Read the data

We simply read de data and store it in different variables. (skipping the first two rows, as we said)

```python
# read the data
data = np.loadtxt('DataSet/dataset.txt', unpack=True, skiprows=2).transpose()

# Read the montecarlo samples
MCb = np.loadtxt('DataSet/mc_bkg.txt', unpack=True, skiprows=2)
MCs = np.loadtxt('DataSet/mc_signal.txt', unpack=True, skiprows=2)
```

We now want to classify the Montecarlo data in two different types: **Data used to train** and **Data used to predict**, separating them with the half of the sample ``l///2``

```python
# define the length of the data
l = len(MCs.transpose())

# first half training
MCs_train = MCs.transpose()[0:l//2]
MCb_train = MCb.transpose()[0:l//2]

# second half predicting
MCs_predct = MCs.transpose()[l//2:]
MCb_predict = MCb.transpose()[l//2:]

# defining noise or signal output
zeros = np.zeros(int(l/2)).transpose()
ones = np.ones(int(l/2)).transpose()

# joint samples train
train_sample = np.concatenate((MCb_train,MC_strain))
train_output = np.concatenate((ones,zeros))

# joint samples predict
Predict = np.concatenate((MCb_predict, MCs_predict))
```

