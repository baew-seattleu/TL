# import common libraries and set fixed values
import numpy as np
import pandas as pd
import pdb as pdb
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

# SciPy libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

# Tensorflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, ReLU
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

# import PyTorch libraries
import torch
import torch.nn as nn
import importlib.util

execution_path = "../../../"
ROOT_DIR = Path(f"{execution_path}/Common_Utils/BalancingAlgorithms/")

# spec_file = ROOT_DIR / "ADKNN_Balance.py"
spec_file = ROOT_DIR / "Balance.py"


spec = importlib.util.spec_from_file_location("balance", spec_file)
balance_method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(balance_method)


class network_tl:
    def __init__(self, X, Y, n_hidden=4, learning_rate=1e-2, device="cpu", epochs=500):
        self.device = device
        self.X = X
        self.Y = Y.reshape(-1, 1)
        self.Y_t = K.variable(self.Y)
        self.Y_tind = None
        # self.Y_t = torch.FloatTensor(self.Y).to(device=self.device)
        self.n_input_dim = X.shape[1]
        self.n_output = 1
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.epochs = epochs

        # if self.device == "cuda":
        #     self.cuda()

    def set_Y_tind(self, Y_tind):
        self.Y_tind = Y_tind

    def make_pop(self, Architecture_GROUP):
        input_layer = Input(shape=(self.n_input_dim,))

        if Architecture_GROUP == "TL_All_Unfrozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1")(input_layer)
            X = Dense(20, activation="relu", name="dense2")(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_1_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2")(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_2_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_3_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3", trainable=False)(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_All_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3", trainable=False)(X)
            X = Dense(
                self.n_output, activation="sigmoid", name="dense4", trainable=False
            )(X)

        self.nn_pop = Model(inputs=input_layer, outputs=X, name="nn_pop")
        self.nn_pop.compile(
            optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def train(self, n_iters=None):
        if n_iters is None:
            n_iters = self.epochs
        self.nn_pop.fit(
            # x=self.X, y=self.Y, epochs=n_iters, batch_size=128, verbose=False
            x=self.X,
            y=self.Y,
            epochs=n_iters,
            batch_size=256,
            verbose=False,
        )
        ##By assigning self.nn_pop.history.history['loss'] to self.training_loss, it stores the training loss values in the instance variable self.training_loss. This allows the loss values to be accessed later, such as when visualizing the training curve or evaluating the performance of the model.
        self.training_loss = self.nn_pop.history.history["loss"]
        ##By assigning self.nn_pop.history.history['accuracy'] to self.training_accuracy, it stores the training accuracy values in the instance variable self.training_accuracy. This allows the accuracy values to be accessed later, such as when visualizing the training curve or evaluating the performance of the model.
        self.training_accuracy = self.nn_pop.history.history["accuracy"]

    def make_ind(self, Architecture_GROUP, pretrained_model):
        input_layer = Input(shape=(self.n_input_dim,))

        if Architecture_GROUP == "TL_All_Unfrozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1")(input_layer)
            X = Dense(20, activation="relu", name="dense2")(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_1_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2")(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_2_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3")(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_3_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3", trainable=False)(X)
            X = Dense(self.n_output, activation="sigmoid", name="dense4")(X)
        elif Architecture_GROUP == "TL_All_Frozen":
            X = Dense(self.n_hidden, activation="relu", name="dense1", trainable=False)(
                input_layer
            )
            X = Dense(20, activation="relu", name="dense2", trainable=False)(X)
            X = Dense(20, activation="relu", name="dense3", trainable=False)(X)
            X = Dense(
                self.n_output, activation="sigmoid", name="dense4", trainable=False
            )(X)

        self.nn_ind = Model(inputs=input_layer, outputs=X, name="nn_ind")

        self.nn_ind.compile(
            optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        for layernum in range(len(pretrained_model.layers) - 1):
            self.nn_ind.layers[layernum].set_weights(
                pretrained_model.layers[layernum].get_weights()
            )

    ##This method retrains the individual model (nn_ind) on new data (Xind and Yind) for a specified number of iterations (n_iters).
    def retrain(self, Xind, Yind, n_iters=None):
        self.Xind = Xind
        self.Yind = Yind.reshape(-1, 1)
        self.Y_tind = K.variable(self.Yind)
        if n_iters is None:
            n_iters = self.epochs
        ##change the batch-size and check
        self.nn_ind.fit(
            # x=self.Xind, y=self.Yind, epochs=n_iters, batch_size=128, verbose=False
            x=self.Xind,
            y=self.Yind,
            epochs=n_iters,
            batch_size=256,
            verbose=False,
        )
        self.training_loss = self.nn_ind.history.history["loss"]
        # self.training_accuracy = self.nn_ind.history.history['categorical_accuracy']
        self.training_loss = self.nn_ind.history.history["loss"]
        self.training_accuracy = self.nn_ind.history.history["accuracy"]

    def predict(self, X):
        return self.nn_ind.predict(X)

    def pop_predict(self, X):
        return self.nn_pop.predict(X)

    def calculate_loss(self, y_hatin):
        y_hat = K.variable(y_hatin)
        return K.eval(binary_crossentropy(self.Y_tind, y_hat))

    def save_model(self, path):
        # Save the Keras model using the save method
        self.nn_pop.save(path)

    def calculate_accuracy(self, y_hat_class, Y):
        # pdb.set_trace()
        return np.sum(Y.reshape(-1, 1) == y_hat_class) / len(Y)

    def test(self, X_test, Y_test):
        # print("X_test", self.X_test)
        # print("Y_test", self.Y_test)
        y_hat = self.predict(X_test)
        y_hat_class = (y_hat + 0.5).astype(int)
        # y_hat_class = np.where(y_hat.cpu()<0.5, 0, 1)
        # print("y_hat", self.y_hat)
        # print("y_hat_class", self.y_hat_class)
        # print("Y_test", self.Y_test)
        return self.evaluate(y_hat_class, Y_test)

    def pop_test(self, X_test, Y_test):
        # print("X_test", self.X_test)
        # print("Y_test", self.Y_test)
        y_hat = self.pop_predict(X_test)
        y_hat_class = (y_hat + 0.5).astype(int)
        # y_hat_class = np.where(y_hat.cpu()<0.5, 0, 1)
        # print("y_hat", self.y_hat)
        # print("y_hat_class", self.y_hat_class)
        # print("Y_test", self.Y_test)
        return self.evaluate(y_hat_class, Y_test)

    def evaluate(self, y_hat_class, Y):  # Y = real value, Y_hat = expected
        # cm = np.array([[0, 0], [0, 0]])
        # if all(y_hat_class == Y.reshape(-1,1)):
        # cm[0][0] += sum(y_hat_class == 0)
        # cm[1][1] += sum(y_hat_class == 1)
        # else:
        # cm = confusion_matrix(Y.reshape(-1, 1), y_hat_class)
        cm = np.zeros(
            (2, 2), dtype=int
        )  # Initialize the confusion matrix as a 2x2 matrix of zeros
        y_hat_class = y_hat_class.reshape(-1)  # Ensure y_hat_class is a 1D array
        # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
        for y_hat, y in zip(y_hat_class, Y):
            if y == 0:
                if y_hat == 0:
                    cm[0, 0] += 1  # True Negative (TN)
                else:
                    cm[0, 1] += 1  # False Positive (FP)
            else:
                if y_hat == 1:
                    cm[1, 1] += 1  # True Positive (TP)
                else:
                    cm[1, 0] += 1  # False Negative (FN)
        # print("cm",cm)
        # print("cm.ravel()",cm.ravel())
        tn, fp, fn, tp = cm.ravel()
        assert (tp + tn + fp + fn) != 0.0
        # a = (tp + tn) / (tp + tn + fp + fn)                                       ## Removed - Accuracy
        wa0 = (
            (tn / (2 * (tn + fp))) if (tn + fp) != 0.0 else 0.0
        )  ## Local Var - Weighted_Accuracy_class0
        wa1 = (
            (tp / (2 * (tp + fn))) if (tp + fn) != 0.0 else 0.0
        )  ## Local Var - Weighted_Accuracy_class1
        wa = wa0 + wa1  ## Weighted Accuracy
        s = tp / (tp + fn) if (tp + fn) != 0.0 else 0.0  ## Specificity
        p0 = tn / (tn + fn) if (tn + fn) != 0.0 else 0.0  ## Precision_class0
        p1 = tp / (tp + fp) if (tp + fp) != 0.0 else 0.0  ## Precision_class1
        r = tn / (tn + fp) if (tn + fp) != 0.0 else 0.0  ## Sensitivity/Recall
        fscore0 = 2 * p0 * r / (p0 + r) if p0 + r != 0.0 else 0.0  ## F1_class0
        fscore1 = 2 * p1 * s / (p1 + s) if p1 + s != 0.0 else 0.0  ## F1_class1
        # d = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) != 0.0 else 0.0     ## Removed - Accuracy
        j = tn / (tn + fp + fn) if (tn + fp + fn) != 0.0 else 0.0  ## Jaccard
        tpr = (
            tp / (fn + tp) if (fn + tp) != 0 else 0.0
        )  ## Local Var - True Positive Rate
        fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0  ## False Positive Rate
        tnr = (
            tn / (tn + fp) if (tn + fp) != 0 else 0.0
        )  ## Local Var - True Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0  ## False Negative Rate
        fdr = fp / (tp + fp) if (tp + fp) != 0 else 0.0  ## False Discovery Rate
        fo_rate = fn / (fn + tn) if (fn + tn) != 0 else 0.0  ## False Omission Rate
        # Formula for AUC_Roc score without using function - https://stackoverflow.com/questions/50848163/manually-calculate-auc
        auc_roc = (1 / 2) - (fpr / 2) + (tpr / 2)  ## auc_roc score
        pavg = (p0 + p1) / 2.0  ## Precision_avg
        f1avg = (fscore0 + fscore1) / 2.0  ## F1_avg
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
        # print('{:<25}: {:.4}'.format("logarithmic_loss", results[10]))
        print("{:<25}: {:.4}".format("False_Discovery_Rate", results[10]))
        print("{:<25}: {:.4}".format("False_Negative_Rate", results[11]))
        print("{:<25}: {:.4}".format("False_Omission_Rate", results[12]))
        print("{:<25}: {:.4}".format("False_Positive_Rate", results[13]))
        print("{:<25}: {:.4}".format("Jaccard", results[14]))

    def save_evaluation(self, results, filename, target):
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
        # plt.savefig(filename+'.pdf')
        plt.close()
