# LSTM Model from Scratch

This is an LSTM model for predicting temperature from scratch -- meaning without Tensorflow, Keras, or Scikit-learn!

* Dataset: https://www.bgc-jena.mpg.de/wetter/
* Used Extra Libraries: pandas, numpy, matplotlib, requests

## Installation and Setup

The main runner code is in `main.py` while the model code is placed in `model/lstm.py`.

To install libraries, create a virtual environment and install the required libraries with:

```bash
# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Install libraries
python3 -m pip install -r requirements.txt
# or
pip install -r requirements.txt
```

The `main.py` file has the main function which runs the model. Note that you can comment out sections to not do those things (ie. comment out the data download section if it is already downloaded).

Run the model and tests with:

```bash
python3 main.py
```

## Example Training Rounds

Example training data (experiments with learning rate) is located at [experiments.log](./experiments.log).

## Experimentation Notebook

A lot of experimentation was done to get the code to the state you see here. All of the experimentation code can be seen in the [Jupyter Notebook](./6375_Project_LSTM_Temperature_Predicition_Model.ipynb).

Some of this includes:
* detailed dataset exploration
* data correlation discovery
* messing around with the dataset size
* full explainations of what is happening

## Research Paper

We also wrote up a research paper detailing the math and methods we used to develop this project. This can be found at [final-paper.pdf](./final-paper.pdf).

