from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ActivationTypesIdsEnum(Enum):
    softmax = 'softmax'
    elu = 'elu'
    selu = 'selu'
    softplus = 'softplus'
    relu = 'relu'
    softsign = 'softsign'
    tanh = 'tanh'
    hard_sigmoid = 'hard_sigmoid'
    sigmoid = 'sigmoid'
    linear = 'linear'

activation_types = [type_ for type_ in ActivationTypesIdsEnum]

class LayerTypesIdsEnum(Enum):
    conv2d = 'conv2d'
    flatten = 'flatten'
    dense = 'dense'
    dropout = 'dropout'
    maxpool2d = 'maxpool2d'
    serial_connection = 'serial_connection'

@dataclass
class LayerParams:
    layer_type: LayerTypesIdsEnum
    neurons: int = None
    activation: str = None
    drop: float = None
    pool_size: Tuple[int, int] = None
    kernel_size: Tuple[int, int] = None
    conv_strides: Tuple[int, int] = None
    pool_strides: Tuple[int, int] = None
