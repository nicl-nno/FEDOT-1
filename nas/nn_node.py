from typing import (List, Optional)

from nas.layer import LayerParams, LayerTypesIdsEnum


class NNNode():
    def __init__(self, nodes_from: Optional[List['NNNode']], layer_params: LayerParams):
        self.nodes_from = nodes_from
        self.layer_params = layer_params

    def __str__(self):
        if self.layer_params.layer_type == LayerTypesIdsEnum.conv2d:
            if self.layer_params.pool_size:
                return f'{self.layer_params.layer_type.value} with maxpool2d'
        return self.layer_params.layer_type.value

    @property
    def ordered_subnodes_hierarchy(self) -> List['NNNode']:
        nodes = [self]
        if self.nodes_from:
            for parent in self.nodes_from:
                nodes += parent.ordered_subnodes_hierarchy
        return nodes

class NNNodeGenerator:
    @staticmethod
    def primary_node(layer_params: LayerParams) -> NNNode:
        return PrimaryNode(layer_params=layer_params)

    @staticmethod
    def secondary_node(layer_params: LayerParams = None,
                       nodes_from: Optional[List['NNNode']] = None) -> NNNode:
        return SecondaryNode(nodes_from=nodes_from, layer_params=layer_params)


class PrimaryNode(NNNode):
    def __init__(self, layer_params:LayerParams):
        super().__init__(nodes_from=None, layer_params=layer_params)


class SecondaryNode(NNNode):
    def __init__(self, nodes_from: Optional[List['NNNode']],
                 layer_params: LayerParams):
        nodes_from = [] if nodes_from is None else nodes_from
        super().__init__(nodes_from=nodes_from, layer_params=layer_params)


