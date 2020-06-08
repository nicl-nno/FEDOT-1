import datetime
import os
import random
from typing import Optional

from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters
from core.composer.visualisation import ComposerVisualiser
from core.models.model import *
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.utils import project_root
from nas.composer.gp_cnn_composer import GPNNComposer, GPNNComposerRequirements
from nas.layer import LayerTypesIdsEnum

random.seed(1)
np.random.seed(1)


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_iceberg_classification_problem(file_path,
                                       max_lead_time: datetime.timedelta = datetime.timedelta(minutes=20),
                                       gp_optimiser_params: Optional[GPChainOptimiserParameters] = None, ):
    dataset_to_compose, dataset_to_validate = InputData.from_json(file_path)
    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    cnn_secondary = [LayerTypesIdsEnum.serial_connection, LayerTypesIdsEnum.dropout]
    cnn_primary = [LayerTypesIdsEnum.conv2d]
    nn_primary = [LayerTypesIdsEnum.dense]
    nn_secondary = [LayerTypesIdsEnum.serial_connection, LayerTypesIdsEnum.dropout]

    # the choice of the metric for the chain quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    composer_requirements = GPNNComposerRequirements(
        cnn_primary=cnn_primary, cnn_secondary=cnn_secondary,
        primary=nn_primary, secondary=nn_secondary, min_arity=2, max_arity=2,
        max_depth=4, pop_size=10, num_of_generations=10,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=max_lead_time, image_size=[75, 75])

    # Create GP-based composer
    composer = GPNNComposer()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function,
                                                is_visualise=False)
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True, input_shape=(75, 75, 3), min_filters=64,
                           max_filters=128, epochs=25)

    ComposerVisualiser.visualise(chain_evo_composed)

    # the quality assessment for the obtained composite models
    roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')

    return roc_on_valid_evo_composed, chain_evo_composed


if __name__ == '__main__':
    # the dataset was obtained from https://www.kaggle.com/c/GiveMeSomeCredit

    # a dataset that will be used as a train and test set during composition

    file_path = 'cases/data/iceberg/iceberg_train.json'
    full_path = os.path.join(str(project_root()), file_path)

    run_iceberg_classification_problem(full_path)
