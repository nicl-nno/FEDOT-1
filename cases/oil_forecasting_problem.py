import os
from copy import copy

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData, OutputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from core.utils import project_root


def get_composite_lstm_chain():
    chain = Chain()
    node_trend = PrimaryNode('trend_data_model')
    node_lstm_trend = SecondaryNode('lstm', nodes_from=[node_trend])

    node_residual = PrimaryNode('residual_data_model')
    node_ridge_residual = SecondaryNode('ridge', nodes_from=[node_residual])

    node_final = SecondaryNode('additive_data_model',
                               nodes_from=[node_ridge_residual, node_lstm_trend])
    chain.add_node(node_final)
    return chain


def calculate_validation_metric(pred: OutputData, valid: InputData,
                                name: str, is_visualise=False) -> float:
    forecast_length = valid.task.task_params.forecast_length

    # skip initial part of time series
    predicted = pred.predict[:, pred.predict.shape[1] - 1]
    real = valid.target[len(valid.target) - len(predicted):]

    # plot results
    if is_visualise:
        compare_plot(predicted, real,
                     forecast_length=forecast_length,
                     model_name=name)

    # the quality assessment for the simulation results
    rmse = mse(y_true=real, y_pred=predicted, squared=False)

    return rmse


def compare_plot(predicted, real, forecast_length, model_name):
    plt.clf()
    _, ax = plt.subplots()
    plt.plot(real, linewidth=1, label="Observed", alpha=0.4)
    plt.plot(predicted, linewidth=1, label="Predicted", alpha=0.6)
    ax.legend()
    plt.xlabel('Time, h')
    plt.ylabel('Oil volume')
    plt.title(f'BORE_OIL_VOL for {forecast_length} hours with {model_name}')
    plt.show()


def run_metocean_forecasting_problem(train_file_path, test_file_path,
                                     forecast_length=50, max_window_size=10,
                                     is_visualise=False):
    # specify the task to solve
    task_to_solve = Task(TaskTypesEnum.ts_forecasting,
                         TsForecastingParams(forecast_length=forecast_length,
                                             max_window_size=max_window_size))

    full_path_train = os.path.join(str(project_root()), train_file_path)
    dataset_to_train = InputData.from_csv(
        full_path_train, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=';')

    # a dataset for a final validation of the composed model
    full_path_test = os.path.join(str(project_root()), test_file_path)
    dataset_to_validate = InputData.from_csv(
        full_path_test, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=';')

    for forecasting_step in range(4):
        start = 400 + 100 * forecasting_step
        end = 400 + 100 * (forecasting_step + 1)
        dataset_to_train_local = copy(dataset_to_train)
        dataset_to_train_local.idx = dataset_to_train_local.idx[start:end]

        dataset_to_train_local.target = dataset_to_train_local.target[start:end]
        dataset_to_train_local.features = dataset_to_train_local.features[start:end, :]

        chain_simple = Chain()
        node_single = PrimaryNode('ridge')
        chain_simple.add_node(node_single)

        chain_simple.fit(input_data=dataset_to_train_local, verbose=False)
        prediction = chain_simple.predict(dataset_to_validate)

    rmse_on_valid_simple = calculate_validation_metric(
        prediction, dataset_to_validate,
        f'full-simple_{forecast_length}',
        is_visualise)

    print(f'RMSE simple: {rmse_on_valid_simple}')

    return rmse_on_valid_simple


if __name__ == '__main__':
    # the dataset was obtained from Volve dataset of oil field

    # a dataset that will be used as a train and test set during composition
    file_path_train = 'cases/data/oil/volve_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/oil/volve_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)

    run_metocean_forecasting_problem(full_path_train, full_path_test,
                                     forecast_length=90, is_visualise=True)
