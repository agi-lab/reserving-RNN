# Recurrent Neural Networks for Individual Loss Reserving with Case Estimates

## Overview

Recent advances in computing capabilities means it is becoming more feasible to estimate reserves using individual claim-level 
data as opposed to traditional aggregate-level data. We propose a recurrent neural network model that is equipped to analyse a 
claim’s entire payment history and case estimate development history, the latter being a data source that is frequently overlooked 
in the existing literature. Given the costs involved in producing case estimates, one of our aims is to determine whether a neural 
network trained solely on the payment history can outperform the raw case estimates.

Analysis is conducted on multiple datasets simulated from the SPLICE R package (Avanzi, Taylor and Wang, 2022).

## How To Use

See [post-uni-modelling](https://github.com/agi-lab/reserving-RNN/tree/post-uni-modelling) for the up-to-date files. 
Both [main](https://github.com/agi-lab/reserving-RNN/tree/main) and [masters-thesis](https://github.com/agi-lab/reserving-RNN/tree/masters-thesis) branches refer to previous work from my Master's 
thesis titled 'Individual Loss Reserving with Recurrent Neural Networks' (Avanzi, Lambrianidis, Taylor and Wong, 2024).

The anaconda environment used for all Python code, as well as a list of R packages and versions used, can be found [here](https://github.com/agi-lab/reserving-RNN/tree/main/Environments). These 
should be installed prior to running any of the files.

The files should be examined and run in the following order:

**1. Generate Dataset.R and Generate Datasets.R**

As the names suggest, these files are responsible for simulating the datasets from [SPLICE](https://github.com/agi-lab/SPLICE) (Avanzi, B., Taylor, G., Wang, M., 2023) 
and [SynthETIC](https://github.com/agi-lab/SynthETIC) (Avanzi B., Taylor G., Wang M., Wong B., 2020).

'Generate Datasets.R' also contains some tests for aggregate models on the simulated datasets. These include the paid and 
incurred chain ladder models, the munich chain ladder (Quarg & Mack, 2004) and the paid-incurred chain (Merz & Wüthrich, 
2010).

**2. Data Manipulation.py**

Contains the main data manipulation, as well as train-test splitting. Prepares the raw data for input into the RNN models.


**3. Model Training.ipynb files**

These notebooks rely on 'Functions.py'. This script contains all the functions and classes to be called from each of the model 
training notebooks. "Model Training WithInc.ipynb" should be examined first, then "... NoInc", then "... WithInc OCL".


## Contact

For any questions or further information, please contact Matthew Lambrianidis (mlambrianidi@student.unimelb.edu.au)
