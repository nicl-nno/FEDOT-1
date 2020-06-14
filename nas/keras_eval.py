from typing import Any

from keras import layers
from keras import models
from keras import optimizers

from core.models.data import InputData, OutputData
from nas.layer import LayerTypesIdsEnum


def keras_model_fit(model, input_data: InputData, verbose: bool = False, batch_size: int = 24,
                    epochs: int = 15):
    model.fit(input_data.features, input_data.target,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose)
    return keras_model_predict(model, input_data)


def keras_model_predict(model, input_data: InputData):
    evaluation_result = model.predict(input_data.features)
    return OutputData(idx=input_data.idx,
                      features=input_data.features,
                      predict=evaluation_result,
                      task_type=input_data.task_type)


def generate_structure(node: Any):
    if node.nodes_from:
        struct = []
        if len(node.nodes_from) == 1:
            struct.append(node)
            struct += generate_structure(node.nodes_from[0])
            return struct
        elif len(node.nodes_from) == 2:
            struct += generate_structure(node.nodes_from[0])
            struct.append(node)
            struct += generate_structure(node.nodes_from[1])
            return struct
    else:
        return [node]


def create_nn_model(chain: Any, input_shape: tuple, classes: int = 2):
    structure = generate_structure(chain.root_node)
    model = models.Sequential()
    for i, layer in enumerate(structure):
        type = layer.layer_params.layer_type
        if type == LayerTypesIdsEnum.conv2d:
            activation = layer.layer_params.activation.value
            kernel_size = layer.layer_params.kernel_size
            conv_strides = layer.layer_params.conv_strides
            filters_num = layer.layer_params.num_of_filters
            if i == 0:
                model.add(
                    layers.Conv2D(filters_num, kernel_size=kernel_size, activation=activation, input_shape=input_shape,
                                  strides=conv_strides))
            else:
                model.add(
                    layers.Conv2D(filters_num, kernel_size=kernel_size, activation=activation, strides=conv_strides))
            if layer.layer_params.pool_size:
                pool_size = layer.layer_params.pool_size
                pool_strides = layer.layer_params.pool_strides
                model.add(layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides))
        elif type == LayerTypesIdsEnum.flatten:
            model.add(layers.Flatten())
        elif type == LayerTypesIdsEnum.dropout:
            drop = layer.layer_params.drop
            model.add(layers.Dropout(drop))
        elif type == LayerTypesIdsEnum.dense:
            activation = layer.layer_params.activation.value
            neurons_num = layer.layer_params.neurons
            model.add(layers.Dense(neurons_num, activation=activation))
    # Output
    output_shape = 1 if classes == 2 else classes
    model.add(layers.Dense(output_shape, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    model.summary()
    return model
