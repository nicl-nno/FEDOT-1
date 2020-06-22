from copy import deepcopy
from random import randint, choice
from typing import Any

from core.composer.optimisers.crossover import CrossoverTypesEnum
from core.composer.optimisers.gp_operators import nodes_from_height, node_depth
from nas.cnn_gp_operators import check_cnn_branch


def cnn_subtree_crossover(chain_first: Any, chain_second: Any, requirements) -> Any:
    max_depth = requirements.max_depth
    side = randint(0, 1)
    random_layer_in_chain_first = randint(1, node_depth(chain_first.root_node.nodes_from[side]) + 1)
    random_layer_in_chain_second = randint(1, node_depth(chain_second.root_node.nodes_from[side]) + 1)
    node_from_chain_first = choice(
        nodes_from_height(chain_first.root_node.nodes_from[side], random_layer_in_chain_first, 1))
    node_from_chain_second = choice(
        nodes_from_height(chain_second.root_node.nodes_from[side], random_layer_in_chain_second, 1))

    summary_depth = random_layer_in_chain_first + node_depth(node_from_chain_second)
    is_summary_depth_permissible = summary_depth <= max_depth and summary_depth != 0
    all_branch_transfer = random_layer_in_chain_first == 1 and random_layer_in_chain_second == 1
    if all_branch_transfer or (side and is_summary_depth_permissible):
        chain_first.replace_node_with_parents(node_from_chain_first, node_from_chain_second)
    elif not side and is_summary_depth_permissible:
        chain_first.replace_node_with_parents(node_from_chain_first, node_from_chain_second)
        is_new_conv_branch_permissible = check_cnn_branch(chain_first.root_node.nodes_from[side],
                                                          requirements.image_size)
        if not is_new_conv_branch_permissible:
            chain_first = deepcopy(chain_second)
    chain_first.sort_nodes()
    return chain_first


crossover_by_type = {
    CrossoverTypesEnum.subtree: cnn_subtree_crossover
}
