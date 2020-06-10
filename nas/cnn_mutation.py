from random import random, choice
from typing import Any
from random import randint
from functools import partial
from copy import deepcopy
from core.composer.optimisers.mutation import MutationTypesEnum, MutationParams, get_mutation_prob
from core.composer.optimisers.gp_operators import nodes_from_height, node_height, node_depth
from nas.cnn_gp_operators import get_random_layer_params, random_branch, check_cnn_branch, branch_output_shape
from nas.layer import LayerTypesIdsEnum, LayerParams


def cnn_simple_mutation(chain: Any, parameters: MutationParams) -> Any:
    node_mutation_probability = get_mutation_prob(mut_id=parameters.requirements.mutation_strength.value,
                                                  root_node=chain.root_node)

    def replace_node_to_random_recursive(node: Any, is_fully_connected_branch: bool) -> Any:
        if not is_fully_connected_branch:
            secondary_nodes = parameters.requirements.cnn_secondary
            primary_nodes = parameters.requirements.cnn_primary
        else:
            secondary_nodes = parameters.requirements.secondary
            primary_nodes = parameters.requirements.primary
        if node.nodes_from:
            if random() < node_mutation_probability:
                new_node_type = choice(secondary_nodes)
                old_node_type = node.layer_params.layer_type
                if not new_node_type == old_node_type and new_node_type == LayerTypesIdsEnum.serial_connection:
                    new_layer_params = get_random_layer_params(new_node_type, parameters.requirements)
                    new_node = parameters.secondary_node_func(layer_params=new_layer_params)
                    chain.update_node(node, new_node)
                    for child in node.nodes_from:
                        replace_node_to_random_recursive(child, is_fully_connected_branch)
        else:
            new_node_type = choice(primary_nodes)
            if new_node_type == LayerTypesIdsEnum.conv2d:
                activation = choice(parameters.requirements.activation_types)
                new_layer_params = LayerParams(layer_type=new_node_type, activation=activation,
                                               kernel_size=node.layer_params.kernel_size,
                                               conv_strides=node.layer_params.conv_strides,
                                               pool_size=node.layer_params.pool_size,
                                               pool_strides=node.layer_params.pool_strides)
            else:
                new_layer_params = get_random_layer_params(new_node_type, parameters.requirements)
            new_node = parameters.primary_node_func(layer_params=new_layer_params)
            chain.update_node(node, new_node)

    for is_fully_connected_branch in (False, True):
        replace_node_to_random_recursive(chain.root_node.nodes_from[int(is_fully_connected_branch)],
                                         is_fully_connected_branch)

    return chain


def cnn_growth_mutation(chain: Any, parameters: MutationParams, local_growth=True) -> Any:
    is_fully_connected_branch = randint(0, 1)
    if is_fully_connected_branch:
        primary_nodes = parameters.requirements.primary
        secondary_nodes = parameters.requirements.secondary
    else:
        primary_nodes = parameters.requirements.cnn_primary
        secondary_nodes = parameters.requirements.cnn_secondary
    random_layer_in_chain = randint(1, node_depth(chain.root_node.nodes_from[is_fully_connected_branch]) + 1)
    node_from_chain = choice(
        nodes_from_height(chain.root_node.nodes_from[is_fully_connected_branch], random_layer_in_chain - 1))
    if local_growth:
        is_primary_node_selected = (not node_from_chain.nodes_from) or (
                node_from_chain.nodes_from and node_from_chain != chain.root_node and randint(0, 1))
    else:
        is_primary_node_selected = randint(0, 1) and not node_height(chain, node_from_chain) \
                                                         < parameters.requirements.max_depth
    if is_primary_node_selected:
        new_node_type = choice(primary_nodes)
        if new_node_type == LayerTypesIdsEnum.conv2d:
            activation = choice(parameters.requirements.activation_types)
            new_layer_params = LayerParams(layer_type=new_node_type, activation=activation,
                                           kernel_size=node_from_chain.layer_params.kernel_size,
                                           conv_strides=node_from_chain.layer_params.conv_strides,
                                           pool_size=node_from_chain.layer_params.pool_size,
                                           pool_strides=node_from_chain.layer_params.pool_strides)
        else:
            new_layer_params = get_random_layer_params(new_node_type, parameters.requirements)
        new_subtree = parameters.secondary_node_func(layer_params=new_layer_params)
        chain.replace_node_with_parents(node_from_chain, new_subtree)
    else:
        if local_growth:
            max_depth = node_depth(node_from_chain)
        else:
            max_depth = parameters.requirements.max_depth - random_layer_in_chain
        new_node_type = choice(secondary_nodes)
        new_layer_params = get_random_layer_params(new_node_type, parameters.requirements)
        new_subtree = parameters.secondary_node_func(layer_params=new_layer_params)
        is_conv_branch = True if not is_fully_connected_branch else False
        if not is_fully_connected_branch:
            current_image_size = parameters.requirements.image_size
            if random_layer_in_chain != 1:
                current_image_size = branch_output_shape(root=chain.root_node.nodes_from[0], image_size=current_image_size,
                                                     subtree_to_delete=node_from_chain)
        else:
            current_image_size = None
        random_branch(secondary_node_func=parameters.secondary_node_func,
                      primary_node_func=parameters.primary_node_func,
                      requirements=parameters.requirements, is_conv_branch=is_conv_branch, height=random_layer_in_chain,
                      node_parent=new_subtree, max_depth=max_depth, image_size=current_image_size)
        if not is_fully_connected_branch:
            chain_copy = deepcopy(chain)  # reserve
            chain.replace_node_with_parents(node_from_chain, new_subtree)
            is_new_conv_branch_permissible = check_cnn_branch(chain.root_node.nodes_from[0],
                                                              parameters.requirements.image_size)
            if not is_new_conv_branch_permissible:
                chain = deepcopy(chain_copy)
    return chain


mutation_by_type = {
    MutationTypesEnum.simple: cnn_simple_mutation,
    MutationTypesEnum.growth: partial(cnn_growth_mutation, local_growth=False),
    MutationTypesEnum.local_growth: partial(cnn_growth_mutation, local_growth=True),
}
