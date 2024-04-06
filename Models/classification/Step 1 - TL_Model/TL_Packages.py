# import common libraries and set fixed values
import importlib.util
import pdb as pdb
from pathlib import Path

import numpy as np
import pandas as pd

# Tensorflow libraries
import tensorflow as tf

# import PyTorch libraries
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split

# SciPy libraries
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense, Input, ReLU
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model, Sequential
from tqdm import tqdm

# Set the execution path and root directory
execution_path = "../../../"
ROOT_DIR = Path(f"{execution_path}/Common_Utils/BalancingAlgorithms/")

# Load balance method module from specified file
spec_file = ROOT_DIR / "Balance.py"
spec = importlib.util.spec_from_file_location("balance", spec_file)
balance_method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(balance_method)


class network_tl:
    """
    A class for transfer learning neural network.
    """

    def __init__(self, X, Y, n_hidden=4, learning_rate=1e-2, device="cpu", epochs=500):
        """
        Initialize the network_tl class.

        Parameters:
        - X (numpy.ndarray): Input data.
        - Y (numpy.ndarray): Target labels.
        - n_hidden (int): Number of neurons in the hidden layer.
        - learning_rate (float): Learning rate for optimization.
        - device (str): Device to run the computation ('cpu' or 'cuda').
        - epochs (int): Number of epochs for training.
        """
        self.device = device
        self.X = X
        self.Y = Y.reshape(-1, 1)
        self.Y_t = K.variable(self.Y)  # TensorFlow tensor
        self.Y_tind = None  # Initialized to None
        self.n_input_dim = X.shape[1]
        self.n_output = 1
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.epochs = epochs

        if self.device == "cuda":
            self.cuda()

    def set_Y_tind(self, Y_tind):
        """
        Set the Y_tind attribute.

        Parameters:
        - Y_tind (object): Y_tind value to set.
        """
        self.Y_tind = Y_tind

    def make_pop(self, Architecture_GROUP):
        """
        Create and compile a neural network model based on the specified architecture group.

        Parameters:
        - Architecture_GROUP (str): Architecture group identifier.
        """
        input_layer = Input(shape=(self.n_input_dim,))

        if Architecture_GROUP == "TL_All_Unfrozen":
            # Define the architecture with all layers trainable
            X = Dense(self.n_hidden, activation="relu", name="dense1")(input_layer)
            X = Dense(20, activation="relu", name="dense2")(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_1_Frozen":
            # Freeze the first layer
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2")(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_2_Frozen":
            # Freeze the first two layers
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_3_Frozen":
            # Freeze the first three layers
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3", trainable=False)(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_All_Frozen":
            # Freeze all layers
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3", trainable=False)(X)
            X = Dense(
                self.n_output, activation="sigmoid", name="dense4", trainable=False
            )(X)

        # Create and compile the model
        self.nn_pop = Model(inputs=input_layer, outputs=X, name="nn_pop")
        self.nn_pop.compile(
            optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def train(self, n_iters=None):
        """
        Trains the neural network.
        Args:
            n_iters (int, optional): Number of iterations for training. Defaults to the value of `self.epochs`.
        """
        if n_iters is None:
            n_iters = self.epochs
        # Fit the neural network model to the training data
        self.nn_pop.fit(
            x=self.X, y=self.Y, epochs=n_iters, batch_size=128, verbose=False
        )
        # By assigning self.nn_pop.history.history['loss'] to self.training_loss, it stores the training loss values in the instance variable self.training_loss. This allows the loss values to be accessed later, such as when visualizing the training curve or evaluating the performance of the model.
        self.training_loss = self.nn_pop.history.history["loss"]
        # By assigning self.nn_pop.history.history['accuracy'] to self.training_accuracy, it stores the training accuracy values in the instance variable self.training_accuracy. This allows the accuracy values to be accessed later, such as when visualizing the training curve or evaluating the performance of the model.
        self.training_accuracy = self.nn_pop.history.history[
            "accuracy"
        ]  # Record training accuracy values

    def make_ind(self, Architecture_GROUP, pretrained_model):
        """
        Creates an individual neural network model tailored for transfer learning by integrating
        a predefined architecture with a pretrained model. This function allows for selective
        freezing of the newly added dense layers based on the specified architecture group,
        facilitating various degrees of fine-tuning and feature extraction capabilities.
        Args:
            Architecture_GROUP (str): Specifies the architecture group for transfer learning,
                determining which layers are to be frozen or unfrozen in the transfer learning process.
            pretrained_model (tf.keras.models.Model): The pretrained Keras model to be utilized
                for transfer learning, providing a foundation of learned features.
        The method updates the class instance with a new Keras model (`nn_ind`) designed according
        to the specified `Architecture_GROUP`. It also compiles the model with the Adam optimizer
        and binary crossentropy loss function, suitable for binary classification tasks.
        """
        # Initialize the input layer with shape based on the model's expected input dimension.
        input_layer = Input(shape=(self.n_input_dim,))
        # Architecture group "TL_All_Unfrozen": No layers are frozen, allowing all to learn during training.
        if Architecture_GROUP == "TL_All_Unfrozen":
            # Sequentially add dense layers with ReLU activation; the final layer uses sigmoid for binary output.
            X = Dense(self.n_hidden, activation="relu", name="dense1")(input_layer)
            X = Dense(20, activation="relu", name="dense2")(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        # Architecture group "TL_1_Frozen": The first added dense layer is frozen.
        elif Architecture_GROUP == "TL_1_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2")(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        # Architecture group "TL_2_Frozen": The first two added dense layers are frozen.
        elif Architecture_GROUP == "TL_2_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        # Architecture group "TL_3_Frozen": The first three added dense layers are frozen.
        elif Architecture_GROUP == "TL_3_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3", trainable=False)(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        # Architecture group "TL_All_Frozen": All added dense layers are frozen.
        elif Architecture_GROUP == "TL_All_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3", trainable=False)(X)
            X = Dense(
                self.n_output, activation="sigmoid", name="dense4", trainable=False
            )(X)
        # Define the model with the specified input and output layers, and compile it.
        self.nn_ind = Model(inputs=input_layer, outputs=X, name="nn_ind")
        self.nn_ind.compile(
            optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        # Integrate weights from the pretrained model into the corresponding layers of the new model.
        for layernum in range(len(pretrained_model.layers) - 1):
            self.nn_ind.layers[layernum].set_weights(
                pretrained_model.layers[layernum].get_weights()
            )

    def retrain(self, Xind, Yind, n_iters=None):
        """
        Retrains the individual neural network model (nn_ind) using the provided dataset.

        Args:
            Xind (np.array): Input features for retraining the model.
            Yind (np.array): Target outputs for the training dataset.
            n_iters (int, optional): Number of training iterations (epochs). Defaults to self.epochs if not provided.

        This method updates the model by fitting it on a new dataset, adjusting weights to better predict the provided targets.
        """
        self.Xind = Xind
        # Reshape target outputs to ensure compatibility with Keras expectations.
        self.Yind = Yind.reshape(-1, 1)
        # Convert target outputs to Keras tensor for loss calculations.
        self.Y_tind = K.variable(self.Yind)
        if n_iters is None:
            n_iters = (
                self.epochs
            )  # Default to predefined number of epochs if not specified.

        # Fit the model on the new data with a specified batch size, suppressing output with verbose=False.
        self.nn_ind.fit(
            x=self.Xind, y=self.Yind, epochs=n_iters, batch_size=128, verbose=False
        )
        # Retrieve training loss and accuracy from the model's history.
        self.training_loss = self.nn_ind.history.history["loss"]
        self.training_accuracy = self.nn_ind.history.history["accuracy"]

    def predict(self, X):
        """
        Makes predictions using the individual neural network model on provided input data.

        Args:
            X (np.array): Input features for making predictions.

        Returns:
            np.array: The predicted outputs generated by the model.
        """
        return self.nn_ind.predict(X)

    def pop_predict(self, X):
        """
        Makes predictions using the population neural network model on provided input data.

        Args:
            X (np.array): Input features for making predictions.

        Returns:
            np.array: The predicted outputs generated by the population model.
        """
        return self.nn_pop.predict(X)

    def calculate_loss(self, y_hatin):
        """
        Calculates the binary cross-entropy loss between the predicted values and the actual targets.

        Args:
            y_hatin (np.array): Predicted outputs.

        Returns:
            float: The calculated binary cross-entropy loss.
        """
        y_hat = K.variable(y_hatin)
        return K.eval(binary_crossentropy(self.Y_tind, y_hat))

    def save_model(self, path):
        """
        Saves the population neural network model to a specified file path.

        Args:
            path (str): The file path where the model should be saved.
        """
        self.nn_pop.save(path)

    def calculate_accuracy(self, y_hat_class, Y):
        """
        Calculates the accuracy of the model based on the provided target outputs and the model's predictions.

        Args:
            y_hat_class (np.array): The predicted class labels (as integers).
            Y (np.array): The actual target outputs.

        Returns:
            float: The accuracy of the model as a decimal fraction.
        """
        return np.sum(Y.reshape(-1, 1) == y_hat_class) / len(Y)

    def test(self, X_test, Y_test):
        """
        Tests the individual model's performance on a given test dataset.

        Args:
            X_test (np.array): The input features of the test dataset.
            Y_test (np.array): The target outputs of the test dataset.

        Returns:
            Various metrics or outcomes to evaluate the model's performance, based on the implementation of `evaluate`.
        """
        y_hat = self.predict(X_test)
        y_hat_class = (y_hat + 0.5).astype(int)
        return self.evaluate(y_hat_class, Y_test)

    def pop_test(self, X_test, Y_test):
        """
        Tests the population model's performance on a given test dataset.

        Args:
            X_test (np.array): The input features of the test dataset.
            Y_test (np.array): The target outputs of the test dataset.

        Returns:
            Various metrics or outcomes to evaluate the model's performance, based on the implementation of `evaluate`.
        """
        y_hat = self.pop_predict(X_test)
        y_hat_class = (y_hat + 0.5).astype(int)
        return self.evaluate(y_hat_class, Y_test)

    def evaluate(self, y_hat_class, Y):
        """
        Calculates various performance metrics of the model predictions against the true labels.

        Args:
            y_hat_class (numpy.ndarray): Predicted labels by the model.
            Y (numpy.ndarray): True labels.

        Returns:
            tuple: A tuple containing an array of calculated metrics and the confusion matrix.

        The function computes a confusion matrix and various performance metrics such as weighted accuracy,
        sensitivity, specificity, precision, F1 score, Jaccard index, and AUC-ROC score to evaluate
        the model's prediction accuracy.
        """
        cm = np.zeros((2, 2), dtype=int)  # Initialize a 2x2 confusion matrix with zeros
        y_hat_class = y_hat_class.reshape(-1)  # Flatten y_hat_class to ensure it is 1D

        # Populate confusion matrix with True Negatives, False Positives, False Negatives, and True Positives
        for y_hat, y in zip(y_hat_class, Y):
            if y == 0:
                if y_hat == 0:
                    cm[0, 0] += 1  # True Negative
                else:
                    cm[0, 1] += 1  # False Positive
            else:
                if y_hat == 1:
                    cm[1, 1] += 1  # True Positive
                else:
                    cm[1, 0] += 1  # False Negative

        # Extract performance metrics from the confusion matrix
        tn, fp, fn, tp = cm.ravel()
        assert (tp + tn + fp + fn) != 0.0, "Division by zero error"

        # Calculation of various performance metrics
        wa0 = (tn / (2 * (tn + fp))) if (tn + fp) != 0.0 else 0.0
        wa1 = (tp / (2 * (tp + fn))) if (tp + fn) != 0.0 else 0.0
        wa = wa0 + wa1  # Weighted Accuracy
        s = tp / (tp + fn) if (tp + fn) != 0.0 else 0.0  # Specificity
        p0 = tn / (tn + fn) if (tn + fn) != 0.0 else 0.0  # Precision for class 0
        p1 = tp / (tp + fp) if (tp + fp) != 0.0 else 0.0  # Precision for class 1
        r = tn / (tn + fp) if (tn + fp) != 0.0 else 0.0  # Sensitivity/Recall
        fscore0 = (
            2 * p0 * r / (p0 + r) if (p0 + r) != 0.0 else 0.0
        )  # F1 Score for class 0
        fscore1 = (
            2 * p1 * s / (p1 + s) if (p1 + s) != 0.0 else 0.0
        )  # F1 Score for class 1
        j = tn / (tn + fp + fn) if (tn + fp + fn) != 0.0 else 0.0  # Jaccard Index
        tpr = tp / (fn + tp) if (fn + tp) != 0 else 0.0  # True Positive Rate
        fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0  # True Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0  # False Negative Rate
        fdr = fp / (tp + fp) if (tp + fp) != 0 else 0.0  # False Discovery Rate
        fo_rate = fn / (fn + tn) if (fn + tn) != 0 else 0.0  # False Omission Rate
        auc_roc = (1 / 2) - (fpr / 2) + (tpr / 2)  # AUC-ROC score approximation
        pavg = (p0 + p1) / 2.0  # Average Precision
        f1avg = (fscore0 + fscore1) / 2.0  # Average F1 Score

        # Return the performance metrics and the confusion matrix
        return (
            np.array(
                [
                    wa,
                    r,
                    s,
                    p0,
                    p1,
                    pavg,
                    fscore0,
                    fscore1,
                    f1avg,
                    auc_roc,
                    fpr,
                    fdr,
                    fnr,
                    fo_rate,
                    j,
                ]
            ),
            cm,
        )

    def show_training_curve(self):
        """
        Displays the training loss and accuracy curves over iterations.
        Utilizes matplotlib to plot the training progress with a clear and intuitive graph.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.training_loss, "-r", label="Training Loss")
        ax.plot(self.training_accuracy, "-b", label="Training Accuracy")
        ax.set_title("Training Loss & Accuracy Rate")
        ax.set_xlabel("Iteration", fontsize="xx-large")
        ax.set_ylabel("Percentage", fontsize="xx-large")
        ax.xaxis.set_tick_params(labelsize="xx-large")
        ax.yaxis.set_tick_params(labelsize="xx-large")
        ax.legend(frameon=False, loc="lower center", ncol=2, fontsize="medium")
        plt.grid(True)
        plt.show()
        plt.close()

    def save_training_curve(self, filename):
        """
        Saves the training loss and accuracy curves to an image file.

        Args:
            filename (str): The name of the file to save the plot to, without an extension.

        Utilizes matplotlib to plot and save the training progress as an image file for future reference.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.training_loss, "-r", label="Training Loss")
        ax.plot(self.training_accuracy, "-b", label="Training Accuracy")
        ax.set_title("Training Loss & Accuracy Rate")
        ax.set_xlabel("Iteration", fontsize="xx-large")
        ax.set_ylabel("Percentage", fontsize="xx-large")
        ax.xaxis.set_tick_params(labelsize="xx-large")
        ax.yaxis.set_tick_params(labelsize="xx-large")
        ax.legend(frameon=False, loc="lower center", ncol=2, fontsize="xx-large")
        plt.grid(True)
        plt.savefig(f"{filename}.png", bbox_inches="tight")
        plt.close()

    def show_evaluation(self, results):
        """
        Display the evaluation results including various performance metrics.
        Args:
            results (array-like): An array containing evaluation metrics.
        The function prints out evaluation metrics such as weighted accuracy, sensitivity/recall, precision,
        F1 score, AUC-ROC score, and more to provide insights into the model's performance.
        """
        print("\n[Evaluation Result]")
        print("{:<25}: {:.4}".format("Weighted Accuracy", results[0]))
        print("{:<25}: {:.4}".format("Sensitivity/Recall", results[1]))
        print("{:<25}: {:.4}".format("Specificity", results[2]))
        print("{:<25}: {:.4}".format("Precision_class0", results[3]))
        print("{:<25}: {:.4}".format("Precision_class1", results[4]))
        print("{:<25}: {:.4}".format("Precision_avg", results[5]))
        print("{:<25}: {:.4}".format("F1_class0", results[6]))
        print("{:<25}: {:.4}".format("F1_class1", results[7]))
        print("{:<25}: {:.4}".format("F1_avg", results[8]))
        print("{:<25}: {:.4}".format("auc_roc_score", results[9]))
        print("{:<25}: {:.4}".format("False_Discovery_Rate", results[10]))
        print("{:<25}: {:.4}".format("False_Negative_Rate", results[11]))
        print("{:<25}: {:.4}".format("False_Omission_Rate", results[12]))
        print("{:<25}: {:.4}".format("False_Positive_Rate", results[13]))
        print("{:<25}: {:.4}".format("Jaccard", results[14]))

    def save_evaluation(self, results, filename, target):
        """
        Save the evaluation results to a CSV file.
        Args:
            results (array-like): An array containing evaluation metrics.
            filename (str): The name of the file to save the results to.
            target (str): The target column name in the CSV file.
        The function creates a DataFrame containing evaluation metrics and saves it to a CSV file
        for further analysis or documentation purposes.
        """
        df_eval = pd.DataFrame(
            index=[
                "Weighted Accuracy",
                "Sensitivity/Recall",
                "Precision_avg",
                "F1_avg",
                "Accuracy",
                "Specificity",
                "Precision",
                "Precision_class1",
                "F1",
                "F1_class1",
                "Dice",
                "Jaccard",
            ]
        )
        df_eval[target] = np.array(results)
        df_eval.to_csv(filename)

    def show_confusion_matrix(self, cm):
        """
        Display the confusion matrix.
        Args:
            cm (numpy.ndarray): Confusion matrix.
        The function plots and displays the confusion matrix for visualizing the model's performance.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.matshow(cm, cmap="binary", interpolation="nearest")
        ax.set_title("Confusion Matrix", fontsize="xx-large")
        ax.set_xlabel("Predicted", fontsize="xx-large")
        ax.set_ylabel("True", fontsize="xx-large")
        ax.set_xticklabels([""] + ["0", "1"], fontsize="xx-large")
        ax.set_yticklabels([""] + ["0", "1"], fontsize="xx-large")
        ax.xaxis.set_ticks_position("bottom")
        for (ii, jj), z in np.ndenumerate(cm):
            ax.text(
                jj,
                ii,
                "{:0.1f}".format(z),
                bbox=dict(facecolor="white", edgecolor="0.3"),
                ha="center",
                va="center",
                fontsize="xx-large",
            )
        plt.show()
        plt.close()

    def save_confusion_matrix(self, cms, filename):
        """
        Save the confusion matrices to an image file.
        Args:
            cms (list): List of confusion matrices.
            filename (str): The name of the file to save the plot to, without an extension.
        The function plots and saves the confusion matrices to an image file for further analysis or documentation.
        """
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        for i in range(len(cms)):
            cm = cms[i]
            cax = axs[i].matshow(cm, cmap="binary", interpolation="nearest")
            axs[i].set_title(f"Confusion Matrix Round #{i + 1}", fontsize="xx-large")
            axs[i].set_xlabel("Predicted", fontsize="xx-large")
            axs[i].set_ylabel("True", fontsize="xx-large")
            axs[i].set_xticks([0, 1])
            axs[i].set_xticklabels(["0", "1"], fontsize="xx-large")
            axs[i].set_yticks([0, 1])
            axs[i].set_yticklabels(["0", "1"], fontsize="xx-large")
            axs[i].xaxis.set_ticks_position("bottom")
            for (ii, jj), z in np.ndenumerate(cm):
                axs[i].text(
                    jj,
                    ii,
                    "{:0.1f}".format(z),
                    bbox=dict(facecolor="white", edgecolor="0.3"),
                    ha="center",
                    va="center",
                    fontsize="xx-large",
                )
        plt.savefig(filename + ".png", bbox_inches="tight")
        plt.close()
