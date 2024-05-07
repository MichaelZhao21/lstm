from model.lstm import Model
from model.data import *


def main():
    # Download data; can comment this out if data already exists
    download_data()

    # Load data and create model
    df = load_data()
    lstm = Model(learn_rate=0.001)
    lstm.pre_process(df)
    lstm.setup()

    # Load model from file; uncomment if continuing training
    # lstm.load()
    # lstm.past_epochs=20

    # Train and save the model
    lstm.train(epochs=20)
    lstm.save()

    # Run tests
    lstm.test()


if __name__ == "__main__":
    main()
