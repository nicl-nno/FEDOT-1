from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Optional,
    Tuple,
    List
)

from core.layers.layer import LayerTypesIdsEnum
from core.composer.chain import NNChain
from core.composer.composer import Composer
from core.composer.gp_composer.gp_composer import GPComposerRequirements
from core.composer.nn_node import NNNodeGenerator
from core.composer.optimisers.gp_optimiser import GPChainOptimiser, GPChainOptimiserParameters
from core.composer.optimisers.gp_operators import permissible_kernel_parameters_correct
from core.composer.visualisation import ComposerVisualiser
from core.composer.write_history import write_composer_history_to_csv
from core.models.data import InputData
from core.models.data import train_test_data_setup


@dataclass
class GPNNComposerRequirements(GPComposerRequirements):
    conv_kernel_size: Tuple[int, int] = (3, 3)
    conv_strides: Tuple[int, int] = (1, 1)
    pool_kernel_size: Tuple[int, int] = (2, 2)
    pool_strides: Tuple[int, int] = (2, 2)
    min_num_of_neurons: int = 100
    max_num_of_neurons: int = 200
    min_filters = 64
    max_filters = 128
    channels_num = 3
    max_drop_size: int = 1
    image_size: List[int] = None
    cnn_secondary: List[LayerTypesIdsEnum] = None
    cnn_primary: List[LayerTypesIdsEnum] = None
    train_epochs_num: int = 10
    batch_size: int = 24
    num_of_classes = 2

    def __post_init__(self):
        if not self.cnn_secondary:
            self.cnn_secondary = [LayerTypesIdsEnum.serial_connection, LayerTypesIdsEnum.maxpool2d,
                                  LayerTypesIdsEnum.dropout]
        if not self.cnn_primary:
            self.cnn_primary = [LayerTypesIdsEnum.conv2d]
        if not self.primary:
            self.primary = [LayerTypesIdsEnum.dense]
        if not self.secondary:
            self.secondary = [LayerTypesIdsEnum.serial_connection, LayerTypesIdsEnum.dropout]
        if self.max_drop_size > 1:
            self.max_drop_size = 1
        if not all([side_size > 3 for side_size in self.image_size]):
            raise ValueError(f'Specified image size is unacceptable')
        self.conv_kernel_size, self.conv_strides = permissible_kernel_parameters_correct(self.image_size,
                                                                                         self.conv_kernel_size,
                                                                                         self.conv_strides, False)
        self.pool_kernel_size, self.pool_strides = permissible_kernel_parameters_correct(self.image_size,
                                                                                         self.pool_kernel_size,
                                                                                         self.pool_strides, True)
        if self.min_num_of_neurons < 1:
            raise ValueError(f'min_num_of_neurons value is unacceptable')
        if self.max_num_of_neurons < 1:
            raise ValueError(f'max_num_of_neurons value is unacceptable')
        if self.max_drop_size < 1:
            raise ValueError(f'max_drop_size value is unacceptable')
        if self.channels_num > 3 or self.channels_num < 1:
            raise ValueError(f'channels_num value must be anywhere from 1 to 3')
        if self.train_epochs_num < 1:
            raise ValueError(f'epochs number less than 1')
        if self.batch_size < 1:
            raise ValueError(f'batch size less than 1')
        if self.min_filters < 2:
            raise ValueError(f'min_filters value is unacceptable')
        if self.max_filters < 2:
            raise ValueError(f'max_filters value is unacceptable')


class GPNNComposer(Composer):
    def __init__(self):
        super().__init__()

    def compose_chain(self, data: InputData, initial_chain: Optional[NNChain],
                      composer_requirements: Optional[GPNNComposerRequirements],
                      metrics: Optional[Callable], optimiser_parameters: GPChainOptimiserParameters = None,
                      is_visualise: bool = False) -> NNChain:
        train_data, test_data = train_test_data_setup(data, 0.8)

        input_shape = [size for size in composer_requirements.image_size]
        input_shape.append(composer_requirements.channels_num)
        input_shape = tuple(input_shape)
        metric_function_for_nodes = partial(self.metric_for_nodes,
                                            metrics, train_data, test_data, input_shape,
                                            composer_requirements.min_filters, composer_requirements.max_filters,
                                            composer_requirements.num_of_classes, composer_requirements.batch_size,
                                            composer_requirements.train_epochs_num)

        optimiser = GPChainOptimiser(initial_chain=initial_chain,
                                     requirements=composer_requirements,
                                     primary_node_func=NNNodeGenerator.primary_node,
                                     secondary_node_func=NNNodeGenerator.secondary_node, chain_class=NNChain,
                                     parameters=optimiser_parameters)

        best_chain, self.history = optimiser.optimise(metric_function_for_nodes)

        historical_chains, historical_fitness = [list(hist_tuple) for hist_tuple in list(zip(*self.history))]

        if is_visualise:
            ComposerVisualiser.visualise_history(historical_chains, historical_fitness)

        write_composer_history_to_csv(historical_fitness=historical_fitness, historical_chains=historical_chains,
                                      pop_size=composer_requirements.pop_size)

        print("GP composition finished")
        return best_chain

    def metric_for_nodes(self, metric_function, train_data: InputData,
                         test_data: InputData, input_shape, min_filters, max_filters, classes, batch_size, epochs,
                         chain: NNChain) -> float:
        chain.fit(train_data, True, input_shape, min_filters, max_filters, classes, batch_size, epochs)
        return metric_function(chain, test_data)
