import numpy as np
import numpy.typing as npt
import math
import pandas as pd


class Model:
    def __init__(self, past_size=24, learn_rate=0.1):
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
        df_cut = df[
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
        df_cut = df_cut[::6].reset_index(drop=True)

        # Perform Z-Score Normalization
        df_normalized = (df_cut - df_cut.mean()) / df_cut.std()

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
        rand_num = np.vectorize(np.random.normal)
        self.weights = rand_num(self.weights)
        self.d_weights = rand_num(self.d_weights)
        self.d_bias = np.random.normal()

        # Keep track of total number of epochs already trained
        self.past_epochs = 0

    def load_weights(self, weights, d_weights, d_bias):
        """Load weights from passed in values"""
        self.weights = weights
        self.d_weights = d_weights
        self.d_bias = d_bias

    def save(self):
        """Save the weights to a file"""
        np.save("weights.npy", self.weights)
        np.save("d_weights.npy", self.d_weights)
        np.save("d_bias.npy", self.d_bias)

    def load(self):
        """Load the weights from the save files to continue training or testing"""
        self.weights = np.load("weights.npy")
        self.d_weights = np.load("d_weights.npy")
        self.d_bias = np.load("d_bias.npy")

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
        self.pred = sum(self.out[5, -1] * self.d_weights)

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
        # Loop through the epochs
        # NOTE: each epoch takes like 2.5 mins :sob:
        for e in range(epochs):
            # TRAINING: Loop through each training example
            sse_train = 0.0
            for ex in range(len(self.X_train_sliced)):
                sse_train += self._train_iteration(
                    self.X_train_sliced[ex], self.y_train_sliced[ex]
                )
            self.mse_train = sse_train / len(self.X_train_sliced)

            # VALIDATION: Run _test_err to get the error for this epoch
            sse_validation = 0.0
            for ex in range(len(self.X_validation_sliced)):
                sse_validation += self._train_iteration(
                    self.X_validation_sliced[ex], self.y_validation_sliced[ex]
                )
            self.mse_validation = sse_validation / len(self.X_validation_sliced)

            # Print out info
            self.past_epochs += 1
            print(
                f"Epoch {self.past_epochs}/{epochs}: [train] {self.mse_train} [validation] {self.mse_validation}"
            )

    def test(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
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
        self.pred = sum(self.out[5, -1] * self.d_weights)

        # Return error
        return (self.pred - y[-1]) ** 2

    def test_graph(self):
        """Uses the test set to create a final testing accuracy and graphs"""
