import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from more_itertools import windowed

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    concatenate,
)
from tensorflow.keras.models import Model, model_from_json
from dvclive.keras import DVCLiveCallback

from sherlock.deploy import helpers


class SherlockModel:
    def __init__(self):
        self.lamb = 0.0001
        self.do = 0.35
        self.lr = 0.0001

        self.model_files_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'model_files'))

    def fit(
        self, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val, model_id: str
    ):
        return fit_many([X_train], y_train, X_val, y_val, model_id)

    def fit(
        self, X_train, y_train, X_val, y_val, model_id
    ):
        if model_id == "sherlock":
            raise ValueError(
                "`model_id` cannot be `sherlock` to avoid overwriting the original model weights."
            )
        num_classes = len(set(y_train))

        encoder = LabelEncoder()
        encoder.fit(y_train)

        feature_cols = helpers.categorize_features()

        X_val_char = X_val[feature_cols["char"]]
        X_val_word = X_val[feature_cols["word"]]
        X_val_par = X_val[feature_cols["par"]]
        # X_val_regex = X_val[feature_cols["regex"]]
        X_val_rest = X_val[feature_cols["rest"]]

        y_train_int = encoder.transform(y_train)
        y_val_int = encoder.transform(y_val)
        y_train_cat = tf.keras.utils.to_categorical(y_train_int)
        y_val_cat = tf.keras.utils.to_categorical(y_val_int)

        callbacks = [EarlyStopping(monitor="val_loss", patience=5), DVCLiveCallback()]

        char_model_input, char_model = self._build_char_submodel(len(feature_cols["char"]))
        word_model_input, word_model = self._build_word_submodel(len(feature_cols["word"]))
        par_model_input, par_model = self._build_par_submodel(len(feature_cols["par"]))
        # regex_model_input, regex_model = self._build_regex_submodel(len(feature_cols["regex"]))
        rest_model_input, rest_model = self._build_rest_submodel(len(feature_cols["rest"]))

        # Merge submodels and build main network
        # merged_model1 = concatenate([char_model, word_model, par_model, regex_model, rest_model])
        merged_model1 = concatenate([char_model, word_model, par_model, rest_model])

        merged_model_output = self._add_main_layers(merged_model1, num_classes)

        model = Model(
            # [char_model_input, word_model_input, par_model_input, regex_model_input, rest_model_input],
            [char_model_input, word_model_input, par_model_input, rest_model_input],
            merged_model_output,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

        sample_start = 0
        #for (batch, (start_idx, end_idx)) in enumerate(windowed(range(0, len(X_train) + 1, 1000), 2)):
        #    sys.stderr.write(f'Batch {batch + 1}...\n')
        #    X = X_train[start_idx:end_idx]
        X = X_train
        X_train_char = X[feature_cols["char"]]
        X_train_word = X[feature_cols["word"]]
        X_train_par = X[feature_cols["par"]]
        # X_train_regex = X[feature_cols["regex"]]
        X_train_rest = X[feature_cols["rest"]]

        model.fit(
            x=[
                X_train_char.values,
                X_train_word.values,
                X_train_par.values,
                # X_train_regex.values,
                X_train_rest.values,
            ],
            y=y_train_cat,
            #y=y_train_cat[start_idx:end_idx],
            #validation_data=X_val,
            validation_steps=5,
            validation_data=(
                [
                    X_val_char.values,
                    X_val_word.values,
                    X_val_par.values,
                    #X_val_regex.values,
                    X_val_rest.values,
                ],
                y_val_cat,
            ),
            callbacks=callbacks,
            epochs=100,
        )

        self.model = model

        _ = helpers._get_categorical_label_encodings(y_train, y_val, model_id)

    def predict(self, X: pd.DataFrame, model_id: str = "sherlock") -> np.array:
        """Use sherlock model to generate predictions for X.

        Parameters
        ----------
        X
            Featurized dataframe to generate predictions for.
        model_id
            ID of the model used for generating predictions.

        Returns
        -------
        Array with predictions for X.
        """
        y_pred = self.predict_proba(X, model_id)
        y_pred_classes = helpers._proba_to_classes(y_pred, model_id)

        return y_pred_classes

    def predict_proba(self, X: pd.DataFrame, model_id: str = "sherlock") -> np.array:
        """Use sherlock model to generate predictions for X.

        Parameters
        ----------
        X
            Featurized data set to generate predictions for.
        model_id
            Identifier of a trained model to use for generating predictions.

        Returns
        -------
        Array with predictions for X.
        """
        feature_cols_dict = helpers.categorize_features()

        y_pred = self.model.predict(
            [
                X[feature_cols_dict["char"]].values,
                X[feature_cols_dict["word"]].values,
                X[feature_cols_dict["par"]].values,
                X[feature_cols_dict["rest"]].values,
            ]
        )

        return y_pred

    def initialize_model_from_json(
        self, with_weights: bool, model_id: str = "sherlock"
    ):
        """Load model architecture and populate with pretrained weights.

        Parameters
        ----------
        with_weights
            Whether to populate the model with trained weights.
        model_id
            The ID of the model file to build, defaults to `sherlock` for using the
            sherlock model with the original weights.
        """
        # callbacks = [EarlyStopping(monitor="val_loss", patience=5)]

        model_filename = os.path.join(
            self.model_files_directory, f"{model_id}_model.json"
        )
        if not os.path.exists(model_filename):
            raise ValueError(
                f"""
                No model file associated with this ID: {model_id}, was found.
                The desired model should be specified and stored first before it can be used.
                """
            )

        file = open(model_filename, "r")
        model = model_from_json(file.read())
        file.close()

        if with_weights:
            weights_filename = os.path.join(
                self.model_files_directory, f"{model_id}_weights.h5"
            )
            if not os.path.exists(weights_filename):
                raise ValueError(
                    f"""
                    There are no weights associated with this model ID: {model_id}.
                    The desired model should be trained first before it can be initialized.
                    """
                )
            model.load_weights(weights_filename)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

        self.model = model

    def store_weights(self, model_id: str):
        if model_id == "sherlock":
            raise ValueError(
                "Cannot save model weights with `sherlock` model ID. Choose an alternative."
            )

        weights_filename = os.path.join(
            self.model_files_directory, f"{model_id}_weights.h5"
        )

        os.makedirs(self.model_files_directory, exist_ok=True)
        self.model.save_weights(weights_filename)

    def _build_char_submodel(self, char_shape):
        n_weights = 300

        char_model_input = Input(shape=(char_shape,))
        char_model1 = BatchNormalization(axis=1)(char_model_input)
        char_model2 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(char_model1)
        char_model3 = Dropout(self.do)(char_model2)
        char_model4 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(char_model3)

        return char_model_input, char_model4

    def _build_word_submodel(self, word_shape):
        n_weights = 200

        word_model_input = Input(shape=(word_shape,))
        word_model1 = BatchNormalization(axis=1)(word_model_input)
        word_model2 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(word_model1)
        word_model3 = Dropout(self.do)(word_model2)
        word_model4 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(word_model3)

        return word_model_input, word_model4

    def _build_par_submodel(self, par_shape):
        n_weights = 400

        par_model_input = Input(shape=(par_shape,))
        par_model1 = BatchNormalization(axis=1)(par_model_input)
        par_model2 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(par_model1)
        par_model3 = Dropout(self.do)(par_model2)
        par_model4 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(par_model3)

        return par_model_input, par_model4

    def _build_regex_submodel(self, regex_shape):
        n_weights = 1000

        regex_model_input = Input(shape=(regex_shape,))
        regex_model1 = BatchNormalization(axis=1)(regex_model_input)
        regex_model2 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(regex_model1)
        regex_model3 = Dropout(self.do)(regex_model2)
        regex_model4 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(regex_model3)

        return regex_model_input, regex_model4

    def _build_rest_submodel(self, rest_shape):

        # Build submodel for remaining features
        rest_model_input = Input(shape=(rest_shape,))
        rest_model1 = BatchNormalization(axis=1)(rest_model_input)

        return rest_model_input, rest_model1

    def _add_main_layers(self, merged_model1, num_classes):
        n_weights = 500

        merged_model2 = BatchNormalization(axis=1)(merged_model1)
        merged_model3 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(merged_model2)
        merged_model4 = Dropout(self.do)(merged_model3)
        merged_model5 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(merged_model4)
        merged_model_output = Dense(
            num_classes,
            activation=tf.nn.softmax,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(merged_model5)

        return merged_model_output
