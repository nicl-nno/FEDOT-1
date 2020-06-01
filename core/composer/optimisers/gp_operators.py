from random import randint, choice
from typing import (Any, List, Tuple, Callable)
from core.layers.layer import LayerTypesIdsEnum, LayerParams, activation_types


def node_height(chain: Any, node: Any) -> int:
    def recursive_child_height(parent_node: Any) -> int:
        node_child = chain.node_childs(parent_node)
        if node_child:
            height = recursive_child_height(node_child[0]) + 1
            return height
        else:
            return 0

    height = recursive_child_height(node)
    return height


def node_depth(node: Any) -> int:
    if not node.nodes_from:
        return 0
    else:
        return 1 + max([node_depth(next_node) for next_node in node.nodes_from])


def nodes_from_height(chain: Any, selected_height: int) -> List[Any]:
    def get_nodes(node: Any, current_height):
        nodes = []
        if current_height == selected_height:
            nodes.append(node)
        else:
            if node.nodes_from:
                for child in node.nodes_from:
                    nodes += get_nodes(child, current_height + 1)
        return nodes

    nodes = get_nodes(chain.root_node, current_height=0)
    return nodes


def random_ml_chain(chain_class: Any, secondary_node_func: Callable, primary_node_func: Callable,
                    requirements, max_depth=None) -> Any:
    max_depth = max_depth if max_depth else requirements.max_depth

    def chain_growth(chain: Any, node_parent: Any):
        offspring_size = randint(requirements.min_arity, requirements.max_arity)
        for offspring_node in range(offspring_size):
            height = node_height(chain, node_parent)
            is_max_depth_exceeded = height >= max_depth - 1
            is_primary_node_selected = height < max_depth - 1 and randint(0, 1)
            if is_max_depth_exceeded or is_primary_node_selected:
                primary_node = primary_node_func(model_type=choice(requirements.primary))
                node_parent.nodes_from.append(primary_node)
                chain.add_node(primary_node)
            else:
                secondary_node = secondary_node_func(model_type=choice(requirements.secondary))
                chain.add_node(secondary_node)
                node_parent.nodes_from.append(secondary_node)
                chain_growth(chain, secondary_node)

    chain = chain_class()
    chain_root = secondary_node_func(model_type=choice(requirements.secondary))
    chain.add_node(chain_root)
    chain_growth(chain, chain_root)
    return chain


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
    is_strides_permissible = all(
        [strides[i] < kernel_size[i] for i in range(len(strides))])
    is_kernel_size_permissible = all(
        [kernel_size[i] < image_size[i] for i in range(len(strides))])
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


def is_image_has_permissible_size(image_size, min_size: int):
    return all([side_size > min_size for side_size in image_size])


def random_cnn_chain(chain_class: Any, secondary_node_func: Callable, primary_node_func: Callable,
                     requirements, max_depth=None) -> Any:
    max_depth = max_depth if max_depth else requirements.max_depth

    def branch_growth(chain: Any, node_parent: Any, left: bool, offspring_size: int = None,
                      node_parent_height: int = None):
        for offspring_node in range(offspring_size):
            height = node_parent_height + 1
            is_max_depth_exceeded = height >= max_depth - 1
            is_primary_node_selected = height < max_depth - 1 and randint(0, 1)
            primary = is_max_depth_exceeded or is_primary_node_selected
            if primary:
                rand_node = choice(requirements.cnn_primary) if left else choice(requirements.primary)
            else:
                rand_node = choice(requirements.cnn_secondary) if left else choice(requirements.secondary)
            activation = choice(activation_types)
            layer_params = None

            is_conv2d_selected = rand_node == LayerTypesIdsEnum.conv2d
            is_pooling_selected = rand_node == LayerTypesIdsEnum.maxpool2d

            if rand_node == LayerTypesIdsEnum.dense:
                neurons = randint(requirements.min_num_of_neurons, requirements.max_num_of_neurons)
                layer_params = LayerParams(layer_type=rand_node, neurons=neurons, activation=activation)
            elif is_conv2d_selected or is_pooling_selected:
                if is_conv2d_selected:
                    kernel_size = requirements.conv_kernel_size
                    strides = requirements.conv_strides
                    pooling = False
                else:
                    kernel_size = requirements.pool_kernel_size
                    strides = requirements.pool_strides
                    pooling = True
                if is_image_has_permissible_size(StaticStorage.current_image_size, 4):
                    kernel_size, strides = kernel_parameters_correction(StaticStorage.current_image_size,
                                                                        kernel_size,
                                                                        strides, pooling)
                    StaticStorage.current_image_size = [
                        output_dimension(StaticStorage.current_image_size[i], kernel_size[i], strides[i]) for i in
                        range(len(StaticStorage.current_image_size))]
                    if is_image_has_permissible_size(StaticStorage.current_image_size, 2):
                        if is_conv2d_selected:
                            layer_params = LayerParams(layer_type=rand_node, activation=activation,
                                                       kernel_size=kernel_size, strides=strides)
                        else:
                            layer_params = LayerParams(layer_type=rand_node, pool_size=kernel_size, strides=strides)
            elif rand_node == LayerTypesIdsEnum.serial_connection:
                layer_params = LayerParams(layer_type=rand_node)
            elif rand_node == LayerTypesIdsEnum.dropout:
                drop = randint(1, (requirements.max_drop_size * 10)) / 10
                layer_params = LayerParams(layer_type=rand_node, drop=drop)

            if layer_params:
                transform_to_primary = False
                if not primary:
                    secondary_node = secondary_node_func(layer_params=layer_params)
                    offspring_size = randint(requirements.min_arity, requirements.max_arity)
                    branch_growth(chain, secondary_node, left, offspring_size, height)
                    if not secondary_node.nodes_from:
                        if secondary_node.layer_params.layer_type == LayerTypesIdsEnum.maxpool2d:
                            transform_to_primary = True
                    else:
                        chain.add_node(secondary_node)
                        node_parent.nodes_from.append(secondary_node)
                if primary or transform_to_primary:
                    primary_node = primary_node_func(layer_params=layer_params)
                    node_parent.nodes_from.append(primary_node)
                    chain.add_node(primary_node)

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
    return chain

def equivalent_subtree(chain_first: Any, chain_second: Any) -> List[Tuple[Any, Any]]:
    """returns the nodes set of the structurally equivalent subtree as: list of pairs [node_from_tree1, node_from_tree2]
    where: node_from_tree1 and node_from_tree2 are equivalent nodes from tree1 and tree2 respectively"""

    def structural_equivalent_nodes(node_first, node_second):
        nodes = []
        is_same_type = type(node_first) == type(node_second)
        node_first_childs = node_first.nodes_from
        node_second_childs = node_second.nodes_from
        if is_same_type and ((not node_first.nodes_from) or len(node_first_childs) == len(node_second_childs)):
            nodes.append((node_first, node_second))
            if node_first.nodes_from:
                for node1_child, node2_child in zip(node_first.nodes_from, node_second.nodes_from):
                    nodes_set = structural_equivalent_nodes(node1_child, node2_child)
                    if nodes_set:
                        nodes += nodes_set
        return nodes

    pairs_set = structural_equivalent_nodes(chain_first.root_node, chain_second.root_node)
    assert isinstance(pairs_set, list)
    return pairs_set
