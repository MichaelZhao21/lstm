import numpy as np
import numpy.typing as npt
import math
import pandas as pd
import matplotlib.pyplot as plt


class Model:
    def __init__(self, past_size=24, learn_rate=0.01):
        """Initializes the model and defines required functions for backpropogation"""
        # Learning rate
        self.learn_rate = learn_rate

        # Number of time steps to train (if time_steps = 4, it will take the previous 3 days and predict day 4)
        self.past_size = past_size

        # Define the activation functions
        self.sigmoid = lambda x: 1 / (1 + math.e ** (-x))
        self.tanh = lambda x: (math.e ** (x) - math.e ** (-x)) / (
            math.e ** (x) + math.e ** (-x)
        )

        # Derivatives of the activation functions
        self.d_sigmoid = lambda x: x * (1 - x)
        self.d_tanh = lambda x: 1 - x**2

        # Vectorize the activation functions
        self.sigmoid_vec = np.vectorize(self.sigmoid)
        self.tanh_vec = np.vectorize(self.tanh)

        print("Initialized LSTM Model")

    # Slices the input data into n time slices
    def _form_slices(
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Form time slices for LSTM training"""
        n = len(X)
        X_sliced = np.full((n - self.past_size, self.past_size, self.in_dims), 0.0)
        y_sliced = np.full((n - self.past_size, self.past_size), 0.0)
        for i in range(0, n - self.past_size):
            X_sliced[i] = X[i : i + self.past_size]
            y_sliced[i] = y[
                i + 1 : i + self.past_size + 1
            ]  # Offset by 1 from the input
        return X_sliced, y_sliced

    def _get_xy(
        self, df: pd.DataFrame
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Extract the X and y components from a dataset"""
        X = df.to_numpy()
        y = df.iloc[:, df.columns == "T (degC)"].to_numpy().flatten()
        return X, y

    def pre_process(self, df: pd.DataFrame):
        # Dimensionality Reduction and Downsampling
        # Only keep most correlated columns and every 6 rows (corr. to hours)
        self.df_cut = df[
            [
                "T (degC)",
                "VPmax (mbar)",
                "rho (g/m**3)",
                "VPdef (mbar)",
                "H2OC (mmol/mol)",
                "VPact (mbar)",
                "sh (g/kg)",
            ]
        ]
        self.df_cut = self.df_cut[::6].reset_index(drop=True)

        # Perform Z-Score Normalization
        df_normalized = (self.df_cut - self.df_cut.mean()) / self.df_cut.std()

        # Split data into training, validation, and test sets
        # Split is 70% 20% 10%
        size = len(df_normalized)
        train = df_normalized[: math.floor(size * 0.7)]
        validation = df_normalized[math.floor(size * 0.7) : math.floor(size * 0.9)]
        test = df_normalized[math.floor(size * 0.9) :]

        # Define as the number of input dimensions
        self.in_dims = len(train.columns)

        # Extract the X and y components
        self.X_train, self.y_train = self._get_xy(train)
        self.X_validation, self.y_validation = self._get_xy(validation)
        self.X_test, self.y_test = self._get_xy(test)

        # Slice all 3 sets of data
        self.X_train_sliced, self.y_train_sliced = self._form_slices(
            self.X_train, self.y_train
        )
        self.X_validation_sliced, self.y_validation_sliced = self._form_slices(
            self.X_validation, self.y_validation
        )
        self.X_test_sliced, self.y_test_sliced = self._form_slices(
            self.X_test, self.y_test
        )

        print("Finished model pre-processing!")

    def setup(self):
        """
        Set up the required data structures and initialize weights.
        ONLY run this if you want to restart training or training for the 1st time
        """
        # Set up the matrix to store internal node calculations
        # Rows: f = 0, i = 1, a = 2, o = 3, c = 4, h = 5 (fiaoch)
        # Columns: Each time slice to take
        # Z-axis: Each input attribute
        self.out = np.full((6, self.past_size, self.in_dims), 0.0)

        # Rows: f = 0, i = 1, a = 2, o = 3 (fiao)
        # Columns: # W = 0, U = 1, b = 2 (WUb)
        # Z-axis: Each input attribute
        self.weights = np.full((4, 3, self.in_dims), 0.0)

        # Dense layer weights - 1D array, size of input dimensions
        # Bias is just a single number for the output node bias
        self.d_weights = np.full((self.in_dims), 0.0)
        self.d_bias = 0.0

        # Output of the entire network - literally just a single number
        self.pred = 0.0

        # Initialize weights to random numbers using Normal Xavier Initialization
        limit = np.sqrt(
            6 / (self.in_dims + self.past_size)
        )  # Xavier initialization limit
        self.weights = np.random.uniform(
            -limit, limit, size=(4, 3, self.in_dims)
        )  # LSTM gate weights
        self.d_weights = np.random.uniform(
            -limit, limit, size=self.in_dims
        )  # Dense layer weights
        self.d_bias = np.random.uniform(-limit, limit)  # Dense layer bias

        # Keep track of total number of epochs already trained
        self.past_epochs = 0
        self.epoch_test_smes = []
        self.epoch_validation_smes = []

        print("Finished model setup")

    def save(self):
        """Save the weights to a file"""
        np.save("weights.npy", self.weights)
        np.save("d_weights.npy", self.d_weights)
        np.save("d_bias.npy", self.d_bias)
        np.save("epoch_test_smes.npy", self.epoch_test_smes)
        np.save("epoch_validation_smes.npy", self.epoch_validation_smes)

        print("All weights and data saved to files")

    def load(self):
        """Load the weights from the save files to continue training or testing"""
        self.weights = np.load("weights.npy")
        self.d_weights = np.load("d_weights.npy")
        self.d_bias = np.load("d_bias.npy")
        self.epoch_test_smes = np.load("epoch_test_smes.npy").tolist()
        self.epoch_validation_smes = np.load("epoch_validation_smes.npy").tolist()

        print("All weights and data loaded from files")

    def _train_iteration(
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> float:
        """Trains a single iteration, given input/output, and returns the squared error"""
        ### FORWARD PROPOGATION ###
        ## LSTM Layer ##

        # Store previous h_t and c_t for LSTM memory
        prev_ht = 0
        prev_ct = 0

        # Loop through all time stamps from t = 0 to n
        for t in range(self.past_size):
            # Set previous values
            if t != 0:
                prev_ct = self.out[4][t - 1]
                prev_ht = self.out[5][t - 1]

            # Calculate the values at each gate
            self.out[0, t] = self.sigmoid(
                self.weights[0, 0] * X[t]
                + self.weights[0, 1] * prev_ht
                + self.weights[0, 2]
            )
            self.out[1, t] = self.sigmoid(
                self.weights[1, 0] * X[t]
                + self.weights[1, 1] * prev_ht
                + self.weights[1, 2]
            )
            self.out[2, t] = self.tanh(
                self.weights[2, 0] * X[t]
                + self.weights[2, 1] * prev_ht
                + self.weights[2, 2]
            )
            self.out[3, t] = self.sigmoid(
                self.weights[3, 0] * X[t]
                + self.weights[3, 1] * prev_ht
                + self.weights[3, 2]
            )
            self.out[4, t] = self.out[0, t] * prev_ct + self.out[1, t] * self.out[2, t]
            self.out[5, t] = self.out[3, t] * self.tanh(self.out[4, t])

        ## Dense Layer ##
        # Calculate the weight sums
        self.pred = sum(self.out[5, -1] * self.d_weights) + self.d_bias

        ### BACKWARD PROPOGATION ###

        ## Dense Layer ##
        # Delta of Xi (outputs of LSTM) - gradients to pass to dense layer
        dxi = np.full((self.in_dims), 0.0)

        # Loop through each input + store gradient/update weights
        for i in range(self.in_dims):
            dxi[i] = (self.pred - y[-1]) * self.d_weights[i]
            self.d_weights[i] = (
                self.d_weights[i]
                - self.learn_rate * (self.pred - y[-1]) * self.out[5, t, i]
            )

        # Also update output bias
        self.d_bias = self.d_bias - self.learn_rate * (self.pred - y[-1])

        ## LSTM Layer ##

        # Delta h_(t-1) for previous layer calc in LSTM
        prev_delht = 0.0

        # Next d_ct used in backprop
        next_dct = 0.0

        # Store weight delta sums
        # Rows: f, i, a, o
        # Columns: dW, dU, db
        dw_sum = np.full((4, 3, self.in_dims), 0.0)

        # Loop through time backwards from t = end time-1 to 0
        for t in range(self.past_size - 1, -1, -1):
            # Calculate all gates' deltas
            dht = prev_delht if t != self.past_size - 1 else dxi
            dct = (
                dht * self.out[3][t] * (1 - self.tanh_vec(self.out[4][t]) ** 2)
                + next_dct * self.out[0][t]
            )
            dat = dct * self.out[1][t]
            dit = dct * self.out[2][t]
            dft = dct * (0 if t == 0 else self.out[4][t - 1])
            dot = dht * self.tanh_vec(self.out[4][t])
            dgates = np.array([dft, dit, dat, dot])

            # Calculate d(gates_t)/db = dgates*
            dgates_star = np.array(
                [
                    self.d_sigmoid(self.out[0, t]),
                    self.d_sigmoid(self.out[1, t]),
                    self.d_tanh(self.out[2, t]),
                    self.d_sigmoid(self.out[3, t]),
                ]
            )

            # Calculate prev_delht and store next_dct
            prev_delht = np.sum(dgates * dgates_star * self.weights[:, 1], axis=0)
            next_dct = dct

            # Calculate all errors' deltas
            dw_sum[:, 0] += dgates * dgates_star * X[t]
            dw_sum[:, 1] += dgates * dgates_star * (0 if t == 0 else self.out[5, t - 1])
            dw_sum[:, 2] += dgates * dgates_star

        # Final weight updates
        self.weights -= self.learn_rate * dw_sum

        # Return squared error
        return (self.pred - y[-1]) ** 2

    def train(self, epochs=5):
        """Trains the model for a given number of epochs"""
        print("Beginning training...")
        epoch_total = self.past_epochs + epochs

        # WE NEED TO SPLIT THIS DATASET bc training is TOOOO slow!
        # Each epoch will only contain 1/10 of the training set bc this runs too slowly
        # Also each time we will validate on a random 1/10 of the validation set
        # epoch_train_subset = len(self.X_train_sliced) // 10
        epoch_train_subset = len(self.X_train_sliced) // 5
        epoch_validation_subset = len(self.X_validation_sliced) // 5

        # Loop through the epochs
        # NOTE: each epoch takes like 2.5 mins :sob:
        for _ in range(epochs):

            # TRAINING: Loop through each training example
            sse_train = 0.0

            X_train_sub = self.X_train_sliced[:-epoch_train_subset]
            y_train_sub = self.y_train_sliced[:-epoch_train_subset]
            for ex in range(len(X_train_sub)):
                sse_train += self._train_iteration(X_train_sub[ex], y_train_sub[ex])
            self.mse_train = sse_train / len(X_train_sub)

            # VALIDATION: Loop through each validation example
            sse_validation = 0.0

            X_val_sub = self.X_validation_sliced[:-epoch_validation_subset]
            y_val_sub = self.y_validation_sliced[:-epoch_validation_subset]
            for ex in range(len(X_val_sub)):
                sse_validation += self._test_iteration(X_val_sub[ex], y_val_sub[ex])
            self.mse_validation = sse_validation / len(X_val_sub)

            # Save the errors for graphing later
            self.epoch_test_smes.append(self.mse_train)
            self.epoch_validation_smes.append(self.mse_validation)

            # Print out info
            self.past_epochs += 1
            print(
                f"Epoch {self.past_epochs}/{epoch_total}: [train] {self.mse_train} [validation] {self.mse_validation}"
            )

        print("Finished training!")

    def _test_iteration(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        """Runs a forward propogation step and returns the squared error"""
        # Store previous h_t and c_t for LSTM memory
        prev_ht = 0
        prev_ct = 0

        # Loop through all time stamps from t = 0 to n
        for t in range(self.past_size):
            # Set previous values
            if t != 0:
                prev_ct = self.out[4][t - 1]
                prev_ht = self.out[5][t - 1]

            # Calculate the values at each gate
            self.out[0, t] = self.sigmoid(
                self.weights[0, 0] * X[t]
                + self.weights[0, 1] * prev_ht
                + self.weights[0, 2]
            )
            self.out[1, t] = self.sigmoid(
                self.weights[1, 0] * X[t]
                + self.weights[1, 1] * prev_ht
                + self.weights[1, 2]
            )
            self.out[2, t] = self.tanh(
                self.weights[2, 0] * X[t]
                + self.weights[2, 1] * prev_ht
                + self.weights[2, 2]
            )
            self.out[3, t] = self.sigmoid(
                self.weights[3, 0] * X[t]
                + self.weights[3, 1] * prev_ht
                + self.weights[3, 2]
            )
            self.out[4, t] = self.out[0, t] * prev_ct + self.out[1, t] * self.out[2, t]
            self.out[5, t] = self.out[3, t] * self.tanh(self.out[4, t])

        ## Dense Layer ##
        # Calculate the weight sums
        self.pred = sum(self.out[5, -1] * self.d_weights) + self.d_bias

        # Return error
        return (self.pred - y[-1]) ** 2
    
    def test(self) -> tuple[float, list[float]]:
        """
        Runs a full test using the test set. Returns the MSE and the results
        of the predictions. Also saves a few graphs!
        """

        ## Run the test data through the network ##
        print("Running test data through network...")

        sse_test = 0.0
        preds = []

        # Loop through the test data
        for i in range(len(self.X_test_sliced)):
            sse_test += self._test_iteration(self.X_test_sliced[i], self.y_test_sliced[i])
            preds.append(self.pred)
        mse_test = sse_test / len(self.X_test_sliced)

        print("Done! MSE:", mse_test)
        print("Creating graphs...")

        ## Plot the training and validation errors ##

        training_error = self.epoch_test_smes
        validation_error = self.epoch_validation_smes

        # Create epochs label
        epochs = np.arange(1, len(training_error) + 1)

        # Plot training and validation errors
        plt.plot(epochs, training_error, label='Training Error')
        plt.plot(epochs, validation_error, label='Validation Error')

        # Add labels and legend
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Training and Validation Errors Over Epochs')
        plt.legend()
        plt.xticks(epochs)

        # Show the plot
        plt.savefig('training_validation_errors.png')

        ## Plot the actual vs predicted values ##

        # Un-normalize the temperature data for plotting
        std = self.df_cut.std()["T (degC)"]
        mean = self.df_cut.mean()["T (degC)"]
        predicted = np.array(preds) * std + mean
        actual = np.array(self.y_test_sliced[:, -1]) * std + mean

        # Unfortunately we can't plot all datapoints as the graph would be ginormous!
        actual = actual[:100]
        predicted = predicted[:100]

        # Generate time steps (consecutive integers corresponding to indices)
        time_steps = list(range(len(actual)))

        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual data as a line plot
        ax.plot(time_steps, actual, label='Actual', marker='o')

        # Plot predicted data as a scatter plot
        ax.scatter(time_steps, predicted, label='Predicted', color='red')

        # Add labels and title
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Temperature')
        ax.set_title('Actual vs. Predicted Temperature')

        # Add legend
        ax.legend()

        # Show the plot
        plt.savefig('actual_vs_predicted.png')

        print("Graphs saved!")

        return mse_test, preds
