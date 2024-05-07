# LSTM Model from Scratch

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
