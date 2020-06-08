from random import random, choice
from typing import Any

from core.composer.optimisers.mutation import MutationTypesEnum, MutationParams, get_mutation_prob
from nas.cnn_gp_operators import get_random_layer_params
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


mutation_by_type = {
    MutationTypesEnum.simple: cnn_simple_mutation
}
