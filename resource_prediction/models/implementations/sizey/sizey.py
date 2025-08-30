"""Sizing tasks for the Sizey model."""

# Code from jonathanbader
# Source: https://github.com/dos-group/sizey

import logging
import statistics
import sys
from typing import Tuple
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from resource_prediction.models.implementations.sizey import (
    KNNPredictor,
    LinearPredictor,
    NeuralNetworkPredictor,
    PredictionModel,
    RandomForestPredictor,
)
from resource_prediction.models.implementations.sizey.experiment_constants import (
    OffsetStrategy,
    UnderPredictionStrategy,
)

simplefilter("ignore", category=ConvergenceWarning)


class Sizey:
    """Sizey prediction ensemble.

    Ensemble comprises of:
        - Linear Regression
        - Neural Network
        - Random Forest
        - KNN

    Sizey uses Resource Allocation Quality (RAQ)
    to select the best predictor or ensemble predictions.
    """

    # list to store pred errors
    pred_err_lin = []
    pred_err_nn = []
    pred_err_rf = []
    pred_err_knn = []

    lin_counter: int = 0
    nn_counter: int = 0
    rf_counter: int = 0
    knn_counter: int = 0
    max_counter: int = 0
    softmax_counter: int = 0

    # Initialize Predictors
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        alpha: float,
        beta: float,
        offset_strategy: OffsetStrategy,
        default_offset: float,
        error_strategy: UnderPredictionStrategy,
        use_softmax: bool,
        error_metric: str,
        random_state: int,
    ):
        # Create models
        self.linear_predictor = LinearPredictor(
            workflow_name="Test",
            task_name="Test",
            err_metr=error_metric,
            random_state=random_state,
        )
        self.neural_network_predictor = NeuralNetworkPredictor(
            workflow_name="Test",
            task_name="Test",
            err_metr=error_metric,
            random_state=random_state,
        )
        self.random_forest_predictor = RandomForestPredictor(
            workflow_name="Test",
            task_name="Test",
            err_metr=error_metric,
            random_state=random_state,
        )
        self.knn_predictor = KNNPredictor(
            workflow_name="Test",
            task_name="Test",
            err_metr=error_metric,
            random_state=random_state,
        )

        self.alpha = alpha
        self.beta = beta
        self.offset_strategy = offset_strategy
        self.default_offset = default_offset
        self.error_strategy = error_strategy
        self.use_softmax = use_softmax
        self.error_metric = error_metric

        self.max_memory_observed = max(y_train)[0]
        self.min_memory_observed = min(y_train)[0]

        # Historical values
        self.x_full = x_train
        self.y_full = y_train

        # Keep track of last prediction, to calculate error
        self.prediction_lin = None
        self.prediction_nn = None
        self.prediction_rf = None
        self.prediction_knn = None

        # Train all models initially
        self._initial_model_training(x_train, y_train)

    # Initial Training
    def _initial_model_training(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.linear_predictor.initial_model_training(x_train, y_train)
        logging.debug("Linear predictor trained")
        self.neural_network_predictor.initial_model_training(x_train, y_train)
        logging.debug("Neural network predictor trained")
        self.random_forest_predictor.initial_model_training(x_train, y_train)
        logging.debug("Random forest predictor trained")
        self.knn_predictor.initial_model_training(x_train, y_train)
        logging.debug("KNN predictor trained")

    # Predict value, scored by RAQ
    def predict(self, x_test: np.ndarray) -> Tuple[float, float]:
        """Predict the target value.
        Args:
            x_test (np.ndarray): The input features for prediction.

        Returns:
            Tuple[float, float]: The offsetted prediction value and the raw prediction.
        """
        # X_test is 1D but X_full is multi-feature, reshape X_test appropriately
        x_test = x_test.reshape(1, -1)

        self.x_full = np.concatenate((self.x_full, x_test))

        # Step 1.
        # Each model makes a prediction
        self.prediction_lin = self.linear_predictor.predict_task(x_test).flatten()[0]
        # self.pred_err_lin.append(((y_test - prediction_lin) / y_test))
        logging.debug("Linear prediction (raw):         %s", self.prediction_lin)

        self.prediction_nn = self.neural_network_predictor.predict_task(
            x_test
        ).flatten()[0]
        # self.pred_err_nn.append(((y_test - prediction_nn) / y_test))
        logging.debug("Neural network prediction (raw): %s", self.prediction_nn)

        self.prediction_rf = self.random_forest_predictor.predict_task(
            x_test
        ).flatten()[0]
        # self.pred_err_rf.append(((y_test - prediction_rf) / y_test))
        logging.debug("Random forest prediction (raw):  %s", self.prediction_rf)

        self.prediction_knn = self.knn_predictor.predict_task(x_test).flatten()[0]
        # self.pred_err_knn.append(((y_test - prediction_knn) / y_test))
        logging.debug("KNN prediction (raw):            %s", self.prediction_knn)

        # Step 2.
        # RAQ scoring
        max_prediction = max(
            [
                self.prediction_lin,
                self.prediction_nn,
                self.prediction_rf,
                self.prediction_knn,
            ]
        )

        # Step 2.1
        # Get accuracy
        accuracy_lin = self.linear_predictor.model_error
        accuracy_nn = self.neural_network_predictor.model_error
        accuracy_rf = self.random_forest_predictor.model_error
        accuracy_knn = self.knn_predictor.model_error

        # Step 2.2
        # Get efficiency
        efficiency_lin = 1 - self.prediction_lin / max_prediction
        efficiency_nn = 1 - self.prediction_nn / max_prediction
        efficiency_rf = 1 - self.prediction_rf / max_prediction
        efficiency_knn = 1 - self.prediction_knn / max_prediction

        # Step 2.3
        # Calculate RAQ score

        # Alpha near 0 prioritizes accurate models
        # Alpha near 1 prioritizes more memory-efficient predictions
        raq_lin = (1 - self.alpha) * accuracy_lin + (self.alpha * efficiency_lin)
        raq_nn = (1 - self.alpha) * accuracy_nn + (self.alpha * efficiency_nn)
        raq_rf = (1 - self.alpha) * accuracy_rf + (self.alpha * efficiency_rf)
        raq_knn = (1 - self.alpha) * accuracy_knn + (self.alpha * efficiency_knn)

        # Create dict mapping model names to RAQ scores
        raq_scores = {
            "LR": raq_lin,
            "NN": raq_nn,
            "RF": raq_rf,
            "KNN": raq_knn,
        }

        logging.debug("RAQ scores:")
        for model, score in raq_scores.items():
            logging.debug("%s: %s", model, score)

        # Step 3 Model selection strategy
        # If softmax is true, we interpolate with hyperparameter beta
        # If softmax is false, we select the model with the highest RAQ score

        if not self.use_softmax:
            logging.debug("Using argmax strategy to weigh predictors")
            max_raq_strategy = max(raq_scores.values())
            selected_model = next(
                model for model, raq in raq_scores.items() if raq == max_raq_strategy
            )
            logging.debug("%s chosen with RAQ   %s", selected_model, max_raq_strategy)

            prediction = None
            offset = None

            if selected_model == "LR":
                self.lin_counter += 1
                prediction = self.prediction_lin
                offset = self._calculate_offset(
                    self.offset_strategy,
                    self.default_offset,
                    self.pred_err_lin,
                    self.linear_predictor,
                )
                logging.debug("LR accuracy:         %s", accuracy_lin)
                logging.debug("LR efficiency:       %s", efficiency_lin)
                logging.debug("LR offset:           %s", offset)
            elif selected_model == "NN":
                self.nn_counter += 1
                prediction = self.prediction_nn
                offset = self._calculate_offset(
                    self.offset_strategy,
                    self.default_offset,
                    self.pred_err_nn,
                    self.neural_network_predictor,
                )
                logging.debug("NN accuracy:         %s", accuracy_nn)
                logging.debug("NN efficiency:       %s", efficiency_nn)
                logging.debug("NN offset:           %s", offset)
            elif selected_model == "RF":
                self.rf_counter += 1
                prediction = self.prediction_rf
                offset = self._calculate_offset(
                    self.offset_strategy,
                    self.default_offset,
                    self.pred_err_rf,
                    self.random_forest_predictor,
                )
                logging.debug("RF accuracy:         %s", accuracy_rf)
                logging.debug("RF efficiency:       %s", efficiency_rf)
                logging.debug("RF offset:           %s", offset)
            elif selected_model == "KNN":
                self.knn_counter += 1
                prediction = self.prediction_knn
                offset = self._calculate_offset(
                    self.offset_strategy,
                    self.default_offset,
                    self.pred_err_knn,
                    self.knn_predictor,
                )
                logging.debug("KNN accuracy:         %s", accuracy_knn)
                logging.debug("KNN efficiency:       %s", efficiency_knn)
                logging.debug("KNN offset:           %s", offset)

            logging.debug("\n")

            if prediction is not None and offset is not None:
                offset_prediction = prediction + offset * prediction
                return offset_prediction, prediction

            raise ValueError("No valid model selected")

        # Else it is softmax strategy
        logging.debug("Using softmax strategy to weigh predictors")
        offset_lin = self._calculate_offset(
            self.offset_strategy,
            self.default_offset,
            self.pred_err_lin,
            self.linear_predictor,
        )
        logging.debug("Linear offset: %s", offset_lin)

        offset_nn = self._calculate_offset(
            self.offset_strategy,
            self.default_offset,
            self.pred_err_nn,
            self.neural_network_predictor,
        )
        logging.debug("Neural network offset: %s", offset_nn)

        offset_rf = self._calculate_offset(
            self.offset_strategy,
            self.default_offset,
            self.pred_err_rf,
            self.random_forest_predictor,
        )
        logging.debug("Random forest offset: %s", offset_rf)

        # start_time = time.time()
        offset_knn = self._calculate_offset(
            self.offset_strategy,
            self.default_offset,
            self.pred_err_knn,
            self.knn_predictor,
        )
        # end_time = time.time()
        # print(f"KNN offset calculation time: {end_time - start_time:.6f} seconds")
        logging.debug("KNN offset: %s", offset_knn)

        sum_raq_softmax = (
            np.exp(self.beta * raq_lin)
            + np.exp(self.beta * raq_nn)
            + np.exp(self.beta * raq_rf)
            + np.exp(self.beta * raq_knn)
        )
        y_pred_softmax = (
            self.prediction_lin * (np.exp(self.beta * raq_lin) / sum_raq_softmax)
            + self.prediction_nn * (np.exp(self.beta * raq_nn) / sum_raq_softmax)
            + self.prediction_rf * (np.exp(self.beta * raq_rf) / sum_raq_softmax)
            + self.prediction_knn * (np.exp(self.beta * raq_knn) / sum_raq_softmax)
        )

        y_pred_softmax_offset = (
            offset_lin * (np.exp(self.beta * raq_lin) / sum_raq_softmax)
            + offset_nn * (np.exp(self.beta * raq_nn) / sum_raq_softmax)
            + offset_rf * (np.exp(self.beta * raq_rf) / sum_raq_softmax)
            + offset_knn * (np.exp(self.beta * raq_knn) / sum_raq_softmax)
        )

        self.softmax_counter += 1
        offset_prediction = y_pred_softmax + y_pred_softmax * y_pred_softmax_offset

        if offset_prediction < 0:
            return (
                self.min_memory_observed * self.default_offset,
                self.min_memory_observed * self.default_offset,
            )

        return offset_prediction, y_pred_softmax

    def calculate_error(self, y_true: float) -> None:
        """Calculate error for all models based on true value."""
        self.pred_err_lin.append(((y_true - self.prediction_lin) / y_true))
        self.pred_err_nn.append(((y_true - self.prediction_nn) / y_true))
        self.pred_err_rf.append(((y_true - self.prediction_rf) / y_true))
        self.pred_err_knn.append(((y_true - self.prediction_knn) / y_true))

        self.y_full = np.append(self.y_full, y_true)

    def _check_gt0(self, prediction) -> bool:
        return prediction > 0

    def _calculate_offset(
        self,
        offset_strategy: OffsetStrategy,
        default_offset: float,
        prediction_error_list: list,
        predictor: PredictionModel,
    ) -> float:
        """Calculate offset based on strategy and predictor error."""
        if len(prediction_error_list) < 1:
            return default_offset

        if offset_strategy == OffsetStrategy.MED_ALL:
            return self._calculate_med_all_offset(prediction_error_list)

        if offset_strategy == OffsetStrategy.MED_UNDER:
            return self._calculate_med_under_offset(prediction_error_list)

        if offset_strategy == OffsetStrategy.STD_ALL:
            return self._calculate_std_all_offset(prediction_error_list)

        if offset_strategy == OffsetStrategy.STD_UNDER:
            return self._calculate_std_under_offset(prediction_error_list)

        # Dynamic strategy
        if offset_strategy == OffsetStrategy.DYNAMIC:
            return self._calculate_dynamic_offset(prediction_error_list, predictor)
            # return self._select_dynamic_offset_wastage(prediction_error_list, predictor)

        raise NotImplementedError(
            "Offset strategy " + str(offset_strategy) + " not found."
        )

    def _calculate_med_all_offset(self, prediction_error_list: list) -> float:
        """Calculate median offset for all predictions."""
        prediction_error_list = np.absolute(prediction_error_list)
        return statistics.median(list(filter(self._check_gt0, prediction_error_list)))

    def _calculate_med_under_offset(self, prediction_error_list: list) -> float:
        """Calculate median offset for under-predictions."""
        if len(list(filter(self._check_gt0, prediction_error_list))) > 0:
            return statistics.median(
                list(filter(self._check_gt0, prediction_error_list))
            )
        return self.default_offset

    def _calculate_std_all_offset(self, prediction_error_list: list) -> float:
        """Calculate standard deviation offset for all predictions."""
        absolute_errors = np.absolute(prediction_error_list)
        q_1 = np.percentile(absolute_errors, 10)
        q_3 = np.percentile(absolute_errors, 90)
        iqr = q_3 - q_1
        lower_bound = q_1 - 1.5 * iqr
        upper_bound = q_3 + 1.5 * iqr
        # outliers = absolute_errors[
        #     (absolute_errors < lower_bound) | (absolute_errors > upper_bound)
        # ]
        cleaned_errors = absolute_errors[
            (absolute_errors >= lower_bound) & (absolute_errors <= upper_bound)
        ]
        cleaned_std_dev = np.std(cleaned_errors)
        return cleaned_std_dev

    def _calculate_std_under_offset(self, prediction_error_list: list) -> float:
        """Calculate standard deviation offset for under-predictions."""
        if len(list(filter(self._check_gt0, prediction_error_list))) > 0:
            return np.std(list(filter(self._check_gt0, prediction_error_list)))
        return self.default_offset

    def _calculate_dynamic_offset(
        self, prediction_error_list: list, predictor
    ) -> float:
        """Calculate dynamic offset based on prediction errors."""
        # Predict over whole dataset
        y_hat_base = predictor.predict_tasks(self.x_full)
        y_hat_base = y_hat_base.ravel()

        # Get offsets for all strategies
        std_all_offset = self._calculate_std_all_offset(prediction_error_list)
        std_under_offset = self._calculate_std_under_offset(prediction_error_list)
        median_all_offset = self._calculate_med_all_offset(prediction_error_list)
        median_under_offset = self._calculate_med_under_offset(prediction_error_list)

        min_offset_value = None
        min_wastage = sys.maxsize

        # Define offset values to test
        offset_values = [
            std_all_offset,
            std_under_offset,
            median_all_offset,
            median_under_offset,
        ]

        for offset_value in offset_values:
            # Use copy of base predictions for each iteration
            y_hat = y_hat_base.copy()

            y_hat = y_hat + offset_value * y_hat

            y_hat_wo_below_zero = y_hat[y_hat > 0]
            y_hat = np.where(y_hat > 0, y_hat, np.min(y_hat_wo_below_zero.min()))

            diff_arr = y_hat - self.y_full

            vectorized_function = np.vectorize(self._next_or_same_power_of_two)

            wastage_over = np.where(diff_arr > 0, diff_arr, 0).sum()
            wastage_under = (
                (
                    vectorized_function(
                        np.where(diff_arr < 0, 1, 0) * self.y_full / y_hat
                    )
                )
                / 2
                * y_hat
                + (
                    vectorized_function(
                        np.where(diff_arr < 0, 1, 0) * self.y_full / y_hat
                    )
                    * y_hat
                )
                - (np.where(diff_arr < 0, 1, 0) * self.y_full)
            ).sum()

            wastage = wastage_over + wastage_under

            if wastage < min_wastage:
                min_wastage = wastage
                min_offset_value = offset_value

        return min_offset_value if min_offset_value is not None else self.default_offset

    def _next_or_same_power_of_two(self, n):
        """Returns the next power of two greater than or equal to n."""
        n = int(np.ceil(n))

        if n < 0:
            raise ValueError("n must be greater than 0")

        if n == 0:
            return n

        power = 1
        while power < n:
            power <<= 1
        return power

    def handle_underprediction(
        self,
        predicted: float,
    ) -> float:
        """Handle underprediction."""

        # Double strategy
        if self.error_strategy == UnderPredictionStrategy.DOUBLE:
            return predicted * 2
        return self.max_memory_observed

    def update_model(self, x_train: pd.Series, y_train: float) -> None:
        """Update the underlying models."""
        if y_train > self.max_memory_observed:
            self.max_memory_observed = y_train
        if y_train < self.min_memory_observed:
            self.min_memory_observed = y_train

        self.linear_predictor.update_model(x_train, y_train)
        self.neural_network_predictor.update_model(x_train, y_train)
        self.random_forest_predictor.update_model(x_train, y_train)
        self.neural_network_predictor.update_model(x_train, y_train)
