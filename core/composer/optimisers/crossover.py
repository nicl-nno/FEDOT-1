from copy import deepcopy
from enum import Enum
from random import randint, choice
from random import random
from typing import Any, List
from core.composer.optimisers.gp_operators import check_cnn_branch
from core.composer.optimisers.gp_operators import nodes_from_height, node_depth, equivalent_subtree
from core.layers.layer import LayerTypesIdsEnum
from core.composer.visualisation import ComposerVisualiser


class CrossoverTypesEnum(Enum):
    subtree = 'subtree'
    onepoint = "onepoint"
    none = 'none'
    cnn_subtree = 'cnn_subtree'


def crossover(types: List[CrossoverTypesEnum], chain_first: Any, chain_second: Any, requirements) -> Any:
    if chain_first is chain_second or random() > requirements.crossover_prob:
        return deepcopy(chain_first)
    type = choice(types)
    chain_first_copy = deepcopy(chain_first)
    if type == CrossoverTypesEnum.none:
        return chain_first_copy
    if type in crossover_by_type.keys():
        return crossover_by_type[type](chain_first_copy, chain_second, requirements)
    else:
        raise ValueError(f'Required crossover not found: {type}')


def cnn_subtree_crossover(chain_first: Any, chain_second: Any, requirements) -> Any:
    chain_first.sort_nodes()
    chain_second.sort_nodes()
    ComposerVisualiser.visualise(chain_first)
    ComposerVisualiser.visualise(chain_second)
    max_depth = requirements.max_depth
    side = randint(0, 1)
    random_layer_in_chain_first = randint(1, node_depth(chain_first.root_node.nodes_from[side]) + 1)
    random_layer_in_chain_second = randint(1, node_depth(chain_second.root_node.nodes_from[side]) + 1)
    node_from_chain_first = choice(
        nodes_from_height(chain_first.root_node.nodes_from[side], random_layer_in_chain_first, 1))
    node_from_chain_second = choice(
        nodes_from_height(chain_second.root_node.nodes_from[side], random_layer_in_chain_second, 1))

    summary_depth = random_layer_in_chain_first + node_depth(node_from_chain_second)
    is_sum_depth_permissible = summary_depth <= max_depth and summary_depth != 0
    all_branch_transfer = random_layer_in_chain_first == 1 and random_layer_in_chain_second == 1
    if all_branch_transfer or (not all_branch_transfer and side and is_sum_depth_permissible):
        chain_first.replace_node_with_parents(node_from_chain_first, node_from_chain_second)
    elif not side and is_sum_depth_permissible and not all_branch_transfer:
        max_pool_primary_to_conv_branch_root = node_from_chain_second.layer_params.layer_type == \
                                               LayerTypesIdsEnum.maxpool2d and not node_from_chain_second.nodes_from \
                                               and random_layer_in_chain_first == 1
        if not max_pool_primary_to_conv_branch_root:
            chain_first.replace_node_with_parents(node_from_chain_first, node_from_chain_second)
            is_new_conv_branch_permissible = check_cnn_branch(chain_first.root_node.nodes_from[side],
                                                              requirements.image_size)
            if not is_new_conv_branch_permissible:
                chain_first = deepcopy(chain_second)
    chain_first.sort_nodes()
    ComposerVisualiser.visualise(chain_first)
    return chain_first


def subtree_crossover(chain_first: Any, chain_second: Any, requirements) -> Any:
    max_depth = requirements.max_depth
    random_layer_in_chain_first = randint(0, chain_first.depth - 1)
    random_layer_in_chain_second = randint(0, chain_second.depth - 1)
    if random_layer_in_chain_first == 0 and random_layer_in_chain_second == 0:
        if randint(0, 1):
            random_layer_in_chain_first = randint(1, chain_first.depth - 1)
        else:
            random_layer_in_chain_second = randint(1, chain_second.depth - 1)

    node_from_chain_first = choice(nodes_from_height(chain_first.root_node, random_layer_in_chain_first))
    node_from_chain_second = choice(nodes_from_height(chain_second.root_node, random_layer_in_chain_second))

    summary_depth = random_layer_in_chain_first + node_depth(node_from_chain_second)
    if summary_depth <= max_depth and summary_depth != 0:
        chain_first.replace_node_with_parents(node_from_chain_first, node_from_chain_second)

    return chain_first


def onepoint_crossover(chain_first: Any, chain_second: Any, max_depth: int) -> Any:
    pairs_of_nodes = equivalent_subtree(chain_first, chain_second)
    if pairs_of_nodes:
        node_from_chain_first, node_from_chain_second = choice(pairs_of_nodes)
        summary_depth = node_depth(chain_first.root_node) - node_depth(node_from_chain_first) + node_depth(
            node_from_chain_second)
        if summary_depth <= max_depth and summary_depth != 0:
            chain_first.replace_node_with_parents(node_from_chain_first, node_from_chain_second)
    return chain_first


crossover_by_type = {
    CrossoverTypesEnum.subtree: subtree_crossover,
    CrossoverTypesEnum.onepoint: onepoint_crossover,
    CrossoverTypesEnum.cnn_subtree: cnn_subtree_crossover
}
