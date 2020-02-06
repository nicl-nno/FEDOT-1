from core.evaluation import XGBoost, LogRegression
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum

from core.repository.model_types_repository import (
    ModelsGroup,
    ModelType,
    ModelMetaInfo,
    ModelMetaInfoTemplate,
    ModelTypesRepository,
    ModelGroupsIdsEnum,
    ModelTypesIdsEnum
)
from core.repository.task_types import MachineLearningTasksEnum


# TODO re-imlement as mock
def _initialise_tree_for_test():
    root = ModelsGroup(ModelGroupsIdsEnum.all)

    ml = ModelsGroup(ModelGroupsIdsEnum.ml, parent=root)

    xgboost_meta = ModelMetaInfo(input_type=[NumericalDataTypesEnum.table, CategoricalDataTypesEnum.table],
                                 output_type=[NumericalDataTypesEnum.vector, CategoricalDataTypesEnum.vector],
                                 task_type=[MachineLearningTasksEnum.classification,
                                            MachineLearningTasksEnum.regression])

    ModelType(ModelTypesIdsEnum.xgboost, xgboost_meta, parent=ml)

    knn_meta = ModelMetaInfo(input_type=[NumericalDataTypesEnum.table],
                             output_type=[CategoricalDataTypesEnum.vector],
                             task_type=[MachineLearningTasksEnum.classification])

    ModelType(ModelTypesIdsEnum.knn, knn_meta, parent=ml)

    logit_meta = ModelMetaInfo(input_type=[NumericalDataTypesEnum.table, CategoricalDataTypesEnum.table],
                               output_type=[CategoricalDataTypesEnum.vector],
                               task_type=[MachineLearningTasksEnum.classification])

    ModelType(ModelTypesIdsEnum.logit, logit_meta, parent=ml)

    return root


def test_search_in_repository_by_id_and_metainfo():
    repo = ModelTypesRepository()
    repo._root = _initialise_tree_for_test()

    model_names = repo.search_model_types_by_attributes(
        desired_ids=[ModelGroupsIdsEnum.ml],
        desired_metainfo=ModelMetaInfoTemplate(task_type=MachineLearningTasksEnum.regression))

    impl = repo.obtain_model_implementation(model_names[0])

    assert model_names[0] is ModelTypesIdsEnum.xgboost
    assert len(model_names) == 1
    assert isinstance(impl, XGBoost)


def test_search_in_repository_by_model_id():
    repo = ModelTypesRepository()

    model_names = repo.search_model_types_by_attributes(desired_ids=[ModelGroupsIdsEnum.all])
    impl = repo.obtain_model_implementation(model_names[0])

    assert model_names[0] is ModelTypesIdsEnum.xgboost
    assert len(model_names) == 3
    assert isinstance(impl, XGBoost)


def test_search_in_repository_by_metainfo():
    repo = ModelTypesRepository()

    model_names = repo.search_model_types_by_attributes(desired_metainfo=ModelMetaInfoTemplate(
        input_type=NumericalDataTypesEnum.table,
        output_type=CategoricalDataTypesEnum.vector,
        task_type=MachineLearningTasksEnum.classification))
    impl = repo.obtain_model_implementation(model_names[2])

    assert model_names[2] is ModelTypesIdsEnum.logit
    assert len(model_names) == 3
    assert isinstance(impl, LogRegression)


def test_direct_model_query():
    repo = ModelTypesRepository()

    impl = repo.obtain_model_implementation(ModelTypesIdsEnum.logit)

    assert isinstance(impl, LogRegression)