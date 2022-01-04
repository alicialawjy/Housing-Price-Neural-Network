import torch
import pickle
import random
import sys
import numpy as np
import pandas as pd

from torch import nn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold


class TorchNeuralNetwork(nn.Module):

    def __init__(self, n_input_vars, neurons_per_layer, layers_activations):
        super(TorchNeuralNetwork, self).__init__()
        self.layers = list()
        input_dim = n_input_vars
        activation_dict = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "identity": nn.Identity()}
        for current_layer in range(len(neurons_per_layer)):
            output_dim = neurons_per_layer[current_layer]
            self.layers.append(nn.Linear(input_dim, output_dim))
            if layers_activations[current_layer] in activation_dict:
                self.layers.append(activation_dict[layers_activations[current_layer]])
            else:
                self.layers.append(activation_dict['identity'])

            input_dim = output_dim
        self.linear_relu_stack = nn.Sequential(
            *self.layers)

    def forward(self, x):
        output = x
        for current_layer in self.layers:
            output = current_layer.forward(output)
        return output


class Regressor:
    def __init__(self, x, nb_epoch=1000, hidden_neurons=[50, 50, 50], hidden_activations=['relu', 'relu', 'relu'],
                 output_activation="identity", lr=0.00001):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame}:
                Raw input data of shape (batch_size, input_size), 
                used to compute the size of the network.
            - nb_epoch {int}: 
                Number of epoch to train the network.
            - hidden_neurons {list}: 
                Each element in the list represents the number of neurons in each linear layer. 
                The length of the list determines the number of linear layers.
            - hidden_activations {list}:
                List of the activation functions to apply to the output of each linear layer.
            - output_activation {str}:
                The activation function used for the output layer.
            - lr {float}:
                The learning rate.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocess x
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.df_missing_mean = None
        self.df_missing_mode = None
        self.min_max_scaler = None
        self.min_max_scaler_y = None
        self.df_combined_columns = None
        self.df_categorical_dummies_columns = None
        hidden_neurons.append(self.output_size)
        hidden_activations.append(output_activation)
        self.hidden_neurons = hidden_neurons
        self.hidden_activations = hidden_activations
        self.neural_network = TorchNeuralNetwork(self.input_size, hidden_neurons, hidden_activations)
        self.loss = torch.nn.MSELoss()
        self.lr = lr
        self.optimiser = torch.optim.Adam(self.neural_network.parameters(), lr=lr)
        # self.optimiser = torch.optim.Adagrad(self.neural_network.parameters() , lr=lr)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def reset(self, x):
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.df_missing_mean = None
        self.df_missing_mode = None
        self.min_max_scaler = None
        self.min_max_scaler_y = None
        self.df_combined_columns = None
        self.df_categorical_dummies_columns = None
        self.neural_network = TorchNeuralNetwork(self.input_size, self.hidden_neurons, self.hidden_activations)
        self.loss = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.neural_network.parameters(), lr=self.lr)

    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess inputs and outputs of the network.
        In our implementation, we strives to accomodate generic dataset rather than specific
        California housing dataset.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - tensor_x {torch.Tensor} -- Preprocessed input array of
              size (batch_size, input_size).
            - tensor_y {torch.Tensor} -- Preprocessed target array of
              size (batch_size, 1). Returns None if y = None.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Pre-processing for training data
        if training:

            # Separate into dataframes containing only categorical or only numerical data
            df_numerical = x.loc[:, x._get_numeric_data().columns]
            df_categorical = x.drop(x._get_numeric_data().columns, axis=1)

            # (i) Handle missing numerical values, 
            # replacing them with the column mean
            df_mean = df_numerical.mean()
            self.df_missing_mean = df_mean
            df_numerical.fillna(df_mean, inplace=True)

            # (ii) Handle missing textual/categorical data, 
            # replacing them with the column mode
            df_mode = df_categorical.mode()
            self.df_missing_mode = df_mode
            df_categorical.fillna(df_mode, inplace=True)

            # (iii) Transform textual data
            df_categorical_dummies = pd.get_dummies(df_categorical, dtype=float)
            df_categorical_dummies_columns = df_categorical_dummies.columns
            self.df_categorical_dummies_columns = df_categorical_dummies_columns

            # (iv) Combine categorical and numerical datasets
            df_combined = pd.concat([df_numerical, df_categorical_dummies], axis=1)
            self.df_combined_columns = df_combined.columns

            # (v) Normalise x data
            min_max_scaler = preprocessing.MinMaxScaler()
            min_max_scaler.fit(df_combined)
            self.min_max_scaler = min_max_scaler
            x_scaled = min_max_scaler.transform(df_combined)
            df = pd.DataFrame(x_scaled, columns=self.df_combined_columns)

            # (vi) Normalise y data
            if y is not None:
                min_max_scaler = preprocessing.MinMaxScaler()
                min_max_scaler.fit(y)
                self.min_max_scaler_y = min_max_scaler
                y_scaled = min_max_scaler.transform(y)

        # Pre-processing for test data
        else:
            # Separate into dataframes containing only categorical or only numerical data
            df_numerical = x.loc[:, x._get_numeric_data().columns]
            df_categorical = x.drop(x._get_numeric_data().columns, axis=1)

            # (i) Handle missing numerical values
            # replacing them with the column mean obtained from the training set (self.df_missing_mean)
            df_numerical.fillna(self.df_missing_mean, inplace=True)

            # (ii) Handle missing textual/categorical data
            # replacing them with the column mode obtained from the training set (self.df_missing_mode)
            df_categorical.fillna(self.df_missing_mode, inplace=True)

            # (iii) Transform textual data
            df_categorical_dummies = pd.get_dummies(df_categorical, dtype=float)
            train_headings = self.df_categorical_dummies_columns
            test_headings = df_categorical_dummies.columns
            # If the test set have extra categories that the model did not train for, drop these datasets
            extra_headings = test_headings.difference(train_headings)
            df_categorical_dummies.drop(columns=list(extra_headings), inplace=True)
            # If the test sets are missing some categories present in the training data, set these categories to 0
            missing_headings = train_headings.difference(test_headings)
            df_categorical_dummies[list(missing_headings)] = 0

            # (iv) Combine categorical and numerical datasets
            df_combined = pd.concat([df_numerical, df_categorical_dummies], axis=1)

            # (v) Normalise x data with the same self.min_max_scaler used for the training set
            x_scaled = self.min_max_scaler.transform(df_combined)
            df = pd.DataFrame(x_scaled, columns=self.df_combined_columns)

            # (vi) Normalise y data with the same self.min_max_scaler_y used for the training set
            if y is not None:
                y_scaled = self.min_max_scaler_y.transform(y)

        # Convert x (np.dataframe) into a torch.Tensor
        tensor_x = torch.from_numpy(np.array(df, dtype=np.float32)).to(torch.float)

        # Convert y (np.dataframe) into a torch.Tensor
        if y is not None:
            tensor_y = torch.from_numpy(np.array(y_scaled)).to(torch.float)

        return tensor_x, (tensor_y if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y=y, training=True)
        for current_epoch in range(self.nb_epoch):
            # Reset the gradients
            self.optimiser.zero_grad()

            # Compute loss
            predicted_output = self.neural_network.forward(X)
            current_loss = self.loss(predicted_output, Y)

            # Backward pass (compute the gradients)
            current_loss.backward()

            # Update parameters
            self.optimiser.step()
            # print(current_loss.item())

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame}:
                Raw input array of shape (batch_size, input_size).

        Returns:
            Y_pred {numpy.ndarray} 
                Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)
        Y_pred_normalised = self.neural_network.forward(X)
        Y_pred_normalised = Y_pred_normalised.detach().numpy()
        Y_pred = self.min_max_scaler_y.inverse_transform(Y_pred_normalised)
        return Y_pred

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        Y_pred = self.predict(x)  # numpy.ndarray
        score = mean_squared_error(y.to_numpy(), Y_pred, squared=False)
        return score

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score_R2(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        Y_pred = self.predict(x)  # numpy.ndarray
        score = r2_score(y.to_numpy(), Y_pred)
        return score

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


def cross_validation(X, Y, regressor):
    # create folds of data
    kf = KFold(n_splits=5, shuffle=True)
    avg_score = 0
    counter = 0
    for train, val in kf.split(X):
        train_x = X.iloc[train]
        train_y = Y.iloc[train]
        val_x = X.iloc[val]
        val_y = Y.iloc[val]
        # reset the nerual network
        regressor.rest(train_x)
        # Train the model
        regressor.fit(train_x, train_y)
        # Obtain the r2 score for the model
        # (i) Train results (use to check overfitting)
        regressor_score_test = regressor.score(val_x, val_y)
        avg_score += regressor_score_test
        counter += 1
    return avg_score / counter


def regressor_hyper_parameter_search(x, y, iteration_number=5, save_results=False, random_search=False,
                                     excel_file_name="record.xlsx"):
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame}
            Raw input array of shape (batch_size, input_size).
        - y {pd.DataFrame}:
            Raw ouput array of shape (batch_size, 1).
        - iteration_number {int}:
            Number of iterations for random search.
        - save_results {bool}:
            If true, results from all iterations will be saved in a file 
            with the filename specified according to input excel_file_name.
        - random_search {bool}:
            If true, hyperparameter search will be executed with a random search method.
            Otherwise, grid search is used.
        - excel_file_name {str}:
            Filename used to store the results if save_results=True.
        
    Returns:
        - best_model {Regressor object}:
            The model with the best testing score found.
        - best_model_score {float}:
            The r2 score for the best model.
        - best_record {dict}:
            key: hyperparameter/ results
            value: the values/ scores for the corresponding hyperparameter/ result for the best model.
            
    """
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # Hyperparameter values to iterate from
    learning_rate_list = [0.02, 0.03, 0.04]
    learning_rate = [0.03, 0.08]
    hidden_neurons_max = 30
    num_of_layers_max = 8
    drop_neurons_init = 8
    drop_neurons = 10
    activation_available = ['relu']
    # activation_available = ['relu', 'sigmoid', 'tanh']

    # Initialise records
    max_score = float('Inf')
    best_record = {}
    record = {}
    inner_record = {}
    best_model_k = 0

    # Split dataset into train, test and validation sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    k = 0
    # Grid Search
    if not random_search:
        # Iterate across different learning rates
        for lr in learning_rate_list:
            # Iterate across different number of hidden layers
            for max_layer in range(2, num_of_layers_max):
                # Iterate across different drop_neuron values 
                # (how much the neurons can differ from the previous layer)
                for n in range(drop_neurons_init, drop_neurons + 1):
                    # Initialise
                    inner_record = {}
                    # temp_hidden_neurons_max = random.randint(hidden_neurons_max/2, hidden_neurons_max)
                    temp_hidden_neurons_max = hidden_neurons_max
                    hidden_activations = []
                    hidden_neurons = []
                    # Build the hidden_neurons and hidden_activations list for the iteration.
                    for layer in range(max_layer):
                        temp_hidden_neurons_max = random.randint(max(2, temp_hidden_neurons_max - n),
                                                                 temp_hidden_neurons_max)
                        hidden_neurons.append(temp_hidden_neurons_max)
                        hidden_activations.append(random.choice(activation_available))

                    # Initialise the Regressor
                    regressor = Regressor(x, nb_epoch=3000, hidden_neurons=hidden_neurons,
                                          hidden_activations=hidden_activations, output_activation="identity", lr=lr)
                    # Train the model
                    regressor.fit(X_train, y_train)
                    # Obtain the r2 score for the model
                    # (i) Train results (use to check overfitting)
                    regressor_score_train = regressor.score(X_train, y_train)
                    r2_train = regressor.score_R2(X_train, y_train)
                    # (ii) Test results
                    regressor_score_test = regressor.score(X_test, y_test)
                    r2_test = regressor.score_R2(X_test, y_test)
                    # (iii) Validation results
                    regressor_score = regressor.score(X_val, y_val)
                    r2 = regressor.score_R2(X_val, y_val)

                    print("Model " + str(k) + "score: " + str(regressor_score))
                    # Record the results for this iteration
                    inner_record = {"num_of_layers": max_layer, "hidden_activations": hidden_activations,
                                    "hidden_neurons": hidden_neurons, "lr": lr,
                                    "regressor_score_train": regressor_score_train,
                                    "regressor_score_val": regressor_score,
                                    "regressor_score_test": regressor_score_test,
                                    "regressor_r2_score_train": r2_train,
                                    "regressor_r2_score_val": r2,
                                    "regressor_r2_score_test": r2_test,
                                    "best_model": None}

                    # If this iteration performs better than max_score, 
                    # Save the model and replace max_score & best_record
                    if regressor_score < max_score:
                        save_regressor(regressor)
                        best_record = inner_record
                        max_score = regressor_score
                        best_model_k = k
                        inner_record["best_model"] = "yes"

                    record["model" + str(k)] = inner_record
                    k += 1

    # Random Search
    elif random_search:
        for i in range(iteration_number):
            # Initialise 
            inner_record = {}
            hidden_activations = []
            hidden_neurons = []
            # Select parameters from specified range for this iteration
            num_of_layers = random.randint(2, num_of_layers_max)
            lr = random.uniform(learning_rate[0], learning_rate[-1])
            temp_hidden_neurons_max = hidden_neurons_max
            # Build the hidden_neurons and hidden_activations list for the iteration.
            for layer in range(num_of_layers):
                temp_hidden_neurons_max = random.randint(max(2, temp_hidden_neurons_max - 4), temp_hidden_neurons_max)
                hidden_activations.append(random.choice(activation_available))
                hidden_neurons.append(temp_hidden_neurons_max)

            # Initialise the Regressor
            regressor = Regressor(x, nb_epoch=3000, hidden_neurons=hidden_neurons,
                                  hidden_activations=hidden_activations, output_activation="identity", lr=lr)
            # Train the model
            regressor.fit(X_train, y_train)
            # Obtain the r2 score for the model
            # (i) Train results (use to check overfitting)
            regressor_score_train = regressor.score(X_train, y_train)
            r2_train = regressor.score_R2(X_train, y_train)
            # (ii) Test results
            regressor_score_test = regressor.score(X_test, y_test)
            r2_test = regressor.score_R2(X_test, y_test)
            # (iii) Validation results
            regressor_score = regressor.score(X_val, y_val)
            r2 = regressor.score_R2(X_val, y_val)
            print("Model " + str(i) + "score: " + str(regressor_score))
            # Record the results for this iteration 
            inner_record = {"num_of_layers": num_of_layers, "hidden_activations": hidden_activations,
                            "hidden_neurons": hidden_neurons, "lr": lr, "regressor_score_train": regressor_score_train,
                            "regressor_score_val": regressor_score, "regressor_score_test": regressor_score_test,
                            "regressor_r2_score_train": r2_train,
                            "regressor_r2_score_val": r2,
                            "regressor_r2_score_test": r2_test,
                            "best_model": None}

            # If this iteration performs better than max_score, 
            # Save the model and replace max_score & best_record
            if regressor_score < max_score:
                save_regressor(regressor)
                best_record = inner_record
                max_score = regressor_score
                best_model_k = k
                inner_record["best_model"] = "yes"
            record["model" + str(k)] = inner_record
            k += 1

    # Test the best model
    best_model = load_regressor()
    best_model_score = best_model.score(X_test, y_test)

    # Save results to excel file if specified.
    if save_results:
        df = pd.DataFrame.from_dict(record, orient='index')
        df.to_excel(excel_file_name)

    return best_model, best_model_score, best_record, record

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    """
    Used to run cross validation on the saved Regressor model (part2_model.pickle)
    """
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = load_regressor()
    # regressor.fit(x_train, y_train)
    # save_regressor(regressor)

    # Error
    error = regressor.score(x, y)
    print("\nRegressor error: {}\n".format(error))


def example_main2(iteration_number=10, random_search=True, excel_file_name="record.xlsx"):
    """
    Used to run hyperparameter tuning.
    """
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting

    # Hyperparameter Tuning *added section
    best_model, best_model_score, best_record, record = regressor_hyper_parameter_search(x_train, y_train,
                                                                                         iteration_number=int(
                                                                                             iteration_number),
                                                                                         save_results=True,
                                                                                         random_search=random_search,
                                                                                         excel_file_name=excel_file_name)
    print("Best model score: " + str(best_model_score))
    print(best_record)


def get_graphs_table(excel_file_name="lr.xlsx"):
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = load_regressor()
    lr_list = np.arange(0.01, 0.2, 0.01)
    print(lr_list)
    record = {}
    for lr in lr_list:
        inner_record = {}
        regressor.reset(X_train)
        regressor.lr = lr
        regressor.fit(X_train, y_train)
        rmse_train = regressor.score(X_train, y_train)
        r2_train = regressor.score_R2(X_train, y_train)
        rmse_val = regressor.score(X_val, y_val)
        r2_val = regressor.score_R2(X_val, y_val)
        rmse_test = regressor.score(X_test, y_test)
        r2_test = regressor.score_R2(X_test, y_test)
        inner_record = {"lr": lr, "regressor_score_train": rmse_train,
                        "regressor_score_val": rmse_val, "regressor_score_test": rmse_test,
                        "regressor_r2_score_train": r2_train,
                        "regressor_r2_score_val": r2_val,
                        "regressor_r2_score_test": r2_test}
        record[str(lr)] = inner_record

    df = pd.DataFrame.from_dict(record, orient='index')
    df.to_excel(excel_file_name)


def draw_model():
    data = pd.read_csv("housing.csv")
    output_label = "median_house_value"
    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    with open('part2_model.pickle', 'rb') as target:
        model = pickle.load(target)
    print(model)


def regen_best_model(filename='part2_model2.pickle'):
    """
    Used to run cross validation on the saved Regressor model (part2_model.pickle)
    """
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(X_train, nb_epoch=3000, hidden_neurons=[27, 21, 19, 19], hidden_activations=['relu', 'relu', 'relu', 'relu'],
                 output_activation="identity", lr=0.02)
                 
    regressor.fit(X_train, y_train)
    error1 = regressor.score(X_train, y_train)
    error2 = regressor.score(X_val, y_val)
    error3 = regressor.score(X_test, y_test)

    print("\nRegressor validation error: {}\n".format(error1))
    print("\nRegressor validation error: {}\n".format(error2))
    print("\nRegressor validation error: {}\n".format(error3))

    with open(filename, 'wb') as target:
        pickle.dump(regressor, target)

    print("Done!")

    # Error
    

if __name__ == "__main__":

    # If no random search
    if sys.argv[1] == "grid":
        print("grid search for the hyperprameter...")
        example_main2(iteration_number=sys.argv[2], random_search=False, excel_file_name=sys.argv[3])
    elif sys.argv[1] == "evaluate":
        # print("cross validation on the loaded regressor part2_model.pickle")
        example_main()
    # If random search
    elif sys.argv[1] == "draw":
        from torchviz import make_dot

        print("Drawing")
        draw_model()
    elif sys.argv[1] == "lr":
        get_graphs_table()
    elif sys.argv[1] == "regen":
        regen_best_model(filename=sys.argv[2])
    else:
        print("random search for hyperparameter")
        example_main2(iteration_number=sys.argv[2], random_search=True, excel_file_name=sys.argv[3])
