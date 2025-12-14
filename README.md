# An Individual Loss Reserving Framework for Recurrent and Feed-Forward Neural Networks and Case Estimates

## Overview

The use of neural networks trained on individual claims data has become increasingly popular in the
actuarial reserving literature. We consider how to best input historical payment data in neural network
models. Additionally, case estimates are also available in the format of a time series, and we extend our
analysis to assessing their predictive power.

In this paper, we compare a feed-forward neural network trained on summarised transactions to
a recurrent neural network equipped to analyse a claim’s entire payment history and/or case estimate
development history. We draw conclusions from training and comparing the performance the models
on multiple, comparable highly complex datasets simulated from SPLICE (Avanzi et al., 2023). We find
evidence that case estimates will improve predictions significantly, but that equipping the neural network
with memory only leads to meagre improvements. Although the case estimation process and quality will
vary significantly between insurers, we provide a standardised methodology for assessing their value.

## How To Use

See [main](https://github.com/agi-lab/reserving-RNN/tree/main) for the files submitted as part of the paper. 
The [masters-thesis](https://github.com/agi-lab/reserving-RNN/tree/masters-thesis) branch refers to previous work from my Master's 
thesis titled 'Individual Loss Reserving with Recurrent Neural Networks' (Lambrianidis, 2024).

The anaconda environment used for all Python code, as well as a list of R packages and versions used, can be found [here](https://github.com/agi-lab/reserving-RNN/tree/main/Environments). These 
should be installed prior to running any of the files.

The files should be examined and run in the following order:

**1. Generate Dataset.R**

As the name suggests, this file is responsible for simulating the datasets from [SPLICE](https://github.com/agi-lab/SPLICE) (Avanzi, B., Taylor, G., Wang, M., 2023) 
and [SynthETIC](https://github.com/agi-lab/SynthETIC) (Avanzi B., Taylor G., Wang M., Wong B., 2020).

**2. Data Manipulation.R**

Contains the main data manipulation, as well as train-test splitting. Prepares the raw data for input into the RNN(+) and FNN(+) models.

**3. Model Training.ipynb files**

These notebooks rely on 'Functions.py'. This script contains all the functions and classes to be called from each of the model 
training notebooks.

## Contact

For any questions or further information, please contact Matthew Lambrianidis (mlambrianidi@student.unimelb.edu.au)
