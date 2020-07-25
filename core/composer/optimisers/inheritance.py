from copy import deepcopy
from enum import Enum
from typing import (Any, List)

from core.composer.optimisers.selection import SelectionTypesEnum, individuals_selection


class GeneticSchemeTypesEnum(Enum):
    steady_state = 'steady_state'
    generational = 'generational'
    parameter_free = 'parameter_free'


def inheritance(type: GeneticSchemeTypesEnum, selection_types: List[SelectionTypesEnum],
                prev_population: List[Any], new_population: List[Any], max_size: int) -> List[Any]:

    genetic_scheme_by_type = dict()
    genetic_scheme_by_type[GeneticSchemeTypesEnum.generational] = direct_inheritance(new_population, max_size)
    for scheme_type in [GeneticSchemeTypesEnum.steady_state, GeneticSchemeTypesEnum.parameter_free]:
        genetic_scheme_by_type[scheme_type] = steady_state_inheritance(selection_types,
                                                                      prev_population,
                                                                      new_population,
                                                                      max_size)
    return genetic_scheme_by_type[type]


def steady_state_inheritance(selection_types: List[SelectionTypesEnum],
                             prev_population: List[Any],
                             new_population: List[Any], max_size: int):
    return individuals_selection(types=selection_types,
                                 individuals=prev_population + new_population,
                                 pop_size=max_size)


def direct_inheritance(new_population: List[Any], max_size: int):
    return deepcopy(new_population[:max_size])



