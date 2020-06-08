from random import choice, randint
from typing import (Tuple, List, Any, Callable)

from nas.layer import LayerTypesIdsEnum, LayerParams


def output_dimension(input_dimension: float, kernel_size: int, stride: int) -> float:
    return ((input_dimension - kernel_size) / stride) + 1


def one_side_parameters_correction(input_dimension: float, kernel_size: int, stride: int) -> \
        Tuple[int, int]:
    output_dim = output_dimension(input_dimension, kernel_size, stride)
    if not float(output_dim).is_integer():
        if kernel_size + 1 < input_dimension:
            kernel_size = kernel_size + 1
        while kernel_size > input_dimension:
            kernel_size = kernel_size - 1
        while not float(
                output_dimension(input_dimension, kernel_size, stride)).is_integer() or stride > input_dimension:
            stride = stride - 1
    return kernel_size, stride


def permissible_kernel_parameters_correct(image_size: List[float], kernel_size: Tuple[int, int],
                                          strides: Tuple[int, int],
                                          pooling: bool) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    is_strides_permissible = all([strides[i] < kernel_size[i] for i in range(len(strides))])
    is_kernel_size_permissible = all([kernel_size[i] < image_size[i] for i in range(len(strides))])
    if not is_strides_permissible:
        if pooling:
            strides = (2, 2)
        else:
            strides = (1, 1)
    if not is_kernel_size_permissible:
        kernel_size = (2, 2)
    return kernel_size, strides


def kernel_parameters_correction(input_image_size: List[float], kernel_size: Tuple[int, int],
                                 strides: Tuple[int, int], pooling: bool) -> Tuple[
    Tuple[int, int], Tuple[int, int]]:
    kernel_size, strides = permissible_kernel_parameters_correct(input_image_size, kernel_size, strides, pooling)
    if len(set(input_image_size)) == 1:
        new_kernel_size, new_strides = one_side_parameters_correction(input_image_size[0], kernel_size[0],
                                                                      strides[0])
        if new_kernel_size != kernel_size:
            kernel_size = tuple([new_kernel_size for i in range(len(input_image_size))])
        if new_strides != strides:
            strides = tuple([new_strides for i in range(len(input_image_size))])
    else:
        new_kernel_size = []
        new_strides = []
        for i in range(len(input_image_size)):
            params = one_side_parameters_correction(input_image_size[i], kernel_size[i], strides[i])
            new_kernel_size.append(params[0])
            new_strides.append(params[1])
        kernel_size = tuple(new_kernel_size) if kernel_size != tuple(new_kernel_size) else kernel_size
        strides = tuple(new_strides) if strides != tuple(new_strides) else strides
    return kernel_size, strides


# TODO is it possible to do without it ?
class StaticStorage:
    current_image_size = None


def is_image_has_permissible_size(image_size, min_size: int, ):
    return all([side_size > min_size for side_size in image_size])


def random_cnn_chain(chain_class: Any, secondary_node_func: Callable, primary_node_func: Callable,
                     requirements, max_depth=None) -> Any:
    max_depth = max_depth if max_depth else requirements.max_depth

    def branch_growth(chain: Any, node_parent: Any, is_conv_branch: bool, offspring_size: int = None,
                      node_parent_height: int = None):
        for offspring_node in range(offspring_size):
            height = node_parent_height + 1
            is_max_depth_exceeded = height >= max_depth - 1
            is_primary_node_selected = height < max_depth - 1 and randint(0, 1)
            primary = is_max_depth_exceeded or is_primary_node_selected
            if primary:
                rand_node_type = choice(requirements.cnn_primary) if is_conv_branch else choice(requirements.primary)
            else:
                rand_node_type = choice(requirements.cnn_secondary) if is_conv_branch else choice(
                    requirements.secondary)

            if rand_node_type == LayerTypesIdsEnum.conv2d:
                activation = choice(requirements.activation_types)
                kernel_size = requirements.conv_kernel_size
                conv_strides = requirements.conv_strides
                pool_size = requirements.pool_kernel_size
                pool_strides = requirements.pool_strides
                if is_image_has_permissible_size(StaticStorage.current_image_size, 4):
                    kernel_size, conv_strides = kernel_parameters_correction(StaticStorage.current_image_size,
                                                                             kernel_size,
                                                                             conv_strides, pooling=False)
                    StaticStorage.current_image_size = [
                        output_dimension(StaticStorage.current_image_size[i], kernel_size[i], conv_strides[i]) for i
                        in
                        range(len(StaticStorage.current_image_size))]

                    if is_image_has_permissible_size(StaticStorage.current_image_size, 4):
                        pool_size, pool_strides = kernel_parameters_correction(StaticStorage.current_image_size,
                                                                               pool_size,
                                                                               pool_strides, pooling=True)

                        StaticStorage.current_image_size = [
                            output_dimension(StaticStorage.current_image_size[i], pool_size[i], pool_strides[i]) for
                            i in range(len(StaticStorage.current_image_size))]
                    else:
                        pool_size, pool_strides = None, None
                else:
                    kernel_size, conv_strides = (1, 1), (1, 1)
                    pool_size, pool_strides = None, None

                layer_params = LayerParams(layer_type=rand_node_type, activation=activation,
                                           kernel_size=kernel_size, conv_strides=conv_strides,
                                           pool_size=pool_size, pool_strides=pool_strides)
            else:
                layer_params = get_random_layer_params(rand_node_type, requirements)

            if primary:
                new_node = primary_node_func(layer_params=layer_params)
            else:
                new_node = secondary_node_func(layer_params=layer_params)
                offspring_size = randint(requirements.min_arity, requirements.max_arity)
                branch_growth(chain, new_node, is_conv_branch, offspring_size, height)
            chain.add_node(new_node)
            node_parent.nodes_from.append(new_node)

    chain = chain_class()
    layer_params = LayerParams(layer_type=LayerTypesIdsEnum.flatten)
    chain_root = secondary_node_func(layer_params=layer_params)
    chain.add_node(chain_root)
    StaticStorage.current_image_size = requirements.image_size
    # left branch of tree generation (cnn part)
    height = 0
    branch_growth(chain, chain_root, True, 1, height)
    # Right branch of tree generation (fully connected nn)
    branch_growth(chain, chain_root, False, 1, height)
    # chain.sort_nodes()
    # ComposerVisualiser.visualise(chain)
    return chain


def get_random_layer_params(type, requirements) -> LayerParams:
    layer_params = None
    if type == LayerTypesIdsEnum.serial_connection:
        layer_params = LayerParams(layer_type=type)
    elif type == LayerTypesIdsEnum.dropout:
        drop = randint(1, (requirements.max_drop_size * 10)) / 10
        layer_params = LayerParams(layer_type=type, drop=drop)
    elif type == LayerTypesIdsEnum.dense or LayerTypesIdsEnum.conv2d:
        activation = choice(requirements.activation_types)
        if LayerTypesIdsEnum.dense:
            neurons = randint(requirements.min_num_of_neurons, requirements.max_num_of_neurons)
            layer_params = LayerParams(layer_type=type, neurons=neurons, activation=activation)
        elif type == LayerTypesIdsEnum.conv2d:
            layer_params = LayerParams(layer_type=type, activation=activation,
                                       kernel_size=requirements.kernel_size, conv_strides=requirements.conv_strides,
                                       pool_size=requirements.pool_size, pool_strides=requirements.pool_strides)
    return layer_params


def check_cnn_branch(root_node: Any, image_size: List[int]):
    def node_check(node: Any):
        result = []
        is_node_correct = True
        type = node.layer_params.layer_type
        if type == LayerTypesIdsEnum.conv2d:
            kernel_size = node.layer_params.kernel_size
            conv_strides = node.layer_params.conv_strides
            StaticStorage.current_image_size = [
                output_dimension(StaticStorage.current_image_size[i], kernel_size[i], conv_strides[i]) for i in
                range(len(StaticStorage.current_image_size))]
            if is_image_has_permissible_size(StaticStorage.current_image_size, 2):
                if not all([float(side_size).is_integer() for side_size in StaticStorage.current_image_size]):
                    is_node_correct = False
                else:
                    if node.layer_params.pool_size:
                        pool_size = node.layer_params.pool_size
                        pool_strides = node.layer_params.pool_strides
                        StaticStorage.current_image_size = [
                            output_dimension(StaticStorage.current_image_size[i], pool_size[i], pool_strides[i]) for i
                            in range(len(StaticStorage.current_image_size))]
                        if not is_image_has_permissible_size(StaticStorage.current_image_size, 2):
                            is_node_correct = False
            else:
                is_node_correct = False
        result.append(is_node_correct)
        if is_node_correct:
            if node.nodes_from:
                for node_from in node.nodes_from:
                    result += node_check(node_from)
        return result

    StaticStorage.current_image_size = image_size
    return all(node_check(root_node))
