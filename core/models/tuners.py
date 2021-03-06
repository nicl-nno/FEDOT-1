from datetime import timedelta
from typing import Callable, Tuple, Union

from numpy.random import choice as nprand_choice, randint
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from skopt import BayesSearchCV

from core.composer.timer import TunerTimer
import operator
from core.models.data import InputData, train_test_data_setup


class Tuner:
    def __init__(self, trained_model, tune_data: InputData,
                 params_range: dict,
                 cross_val_fold_num: int,
                 scorer: Union[str, callable],
                 time_limit,
                 iterations: int):
        self.time_limit: timedelta \
            = time_limit
        self.trained_model = trained_model
        self.tune_data = tune_data
        self.params_range = params_range
        self.cross_val_fold_num = cross_val_fold_num
        self.scorer = scorer
        self.max_iterations = iterations

    def tune(self) -> Union[Tuple[dict, object], Tuple[None, None]]:
        raise NotImplementedError()

    def _is_score_better(self, previous, current):
        __compare = {
            'classification': operator.gt,
            'regression': operator.lt
        }
        comparison = __compare.get(self.tune_data.task.task_type.name)
        try:
            return comparison(current, previous)
        except ValueError as ex:
            print(f'Score comparison can not be held because {ex}')


class SklearnTuner(Tuner):
    def __init__(self, trained_model, tune_data: InputData,
                 params_range: dict,
                 cross_val_fold_num: int,
                 scorer: Union[str, callable],
                 time_limit, iterations):
        super().__init__(trained_model=trained_model,
                         tune_data=tune_data,
                         params_range=params_range,
                         cross_val_fold_num=cross_val_fold_num,
                         scorer=scorer,
                         time_limit=time_limit,
                         iterations=iterations)
        self.search_strategy = None

    def tune(self) -> Union[Tuple[dict, object], Tuple[None, None]]:
        raise NotImplementedError()

    def _sklearn_tune(self, tune_data: InputData):
        try:
            search = self.search_strategy.fit(tune_data.features, tune_data.target.ravel())
            return search.best_params_, search.best_estimator_
        except ValueError as ex:
            print(f'Unsuccessful fit because of {ex}')
            return None, None


class SklearnRandomTuner(SklearnTuner):
    def tune(self) -> Union[Tuple[dict, object], Tuple[None, None]]:
        self.search_strategy = RandomizedSearchCV(estimator=self.trained_model,
                                                  param_distributions=self.params_range,
                                                  n_iter=self.max_iterations,
                                                  cv=self.cross_val_fold_num,
                                                  scoring=self.scorer)
        return self._sklearn_tune(tune_data=self.tune_data)


class SklearnGridSearchTuner(SklearnTuner):
    def tune(self) -> Union[Tuple[dict, object], Tuple[None, None]]:
        self.search_strategy = GridSearchCV(estimator=self.trained_model,
                                            param_grid=self.params_range,
                                            cv=self.cross_val_fold_num,
                                            scoring=self.scorer)
        return self._sklearn_tune(self.tune_data)


class SklearnBayesSearchCV(SklearnTuner):
    def tune(self) -> Union[Tuple[dict, object], Tuple[None, None]]:
        self.search_strategy = BayesSearchCV(estimator=self.trained_model,
                                             search_spaces=self.params_range,
                                             n_iter=self.max_iterations,
                                             cv=self.cross_val_fold_num,
                                             scoring=self.scorer)
        return self._sklearn_tune(self.tune_data)


class SklearnCustomRandomTuner(Tuner):
    def tune(self) -> Union[Tuple[dict, object], Tuple[None, None]]:
        try:
            with TunerTimer() as timer:
                best_score = cross_val_score(self.trained_model, self.tune_data.features,
                                             self.tune_data.target, scoring=self.scorer,
                                             cv=self.cross_val_fold_num).mean()
                best_model = self.trained_model
                best_params = None
                for iteration in range(self.max_iterations):
                    params = {k: nprand_choice(v) for k, v in self.params_range.items()}
                    for param in params:
                        setattr(self.trained_model, param, params[param])
                    score = cross_val_score(self.trained_model, self.tune_data.features,
                                            self.tune_data.target, scoring=self.scorer,
                                            cv=self.cross_val_fold_num).mean()
                    if self._is_score_better(previous=best_score, current=score):
                        best_params = params
                        best_model = self.trained_model
                        best_score = score

                    if timer.is_time_limit_reached(self.time_limit):
                        break
                return best_params, best_model
        except ValueError as ex:
            print(f'Unsuccessful fit because of {ex}')
            return None, None


class ForecastingCustomRandomTuner:
    # TODO discuss
    def tune(self,
             fit: Callable,
             predict: Callable,
             tune_data: InputData, params_range: dict,
             default_params: dict, iterations: int) -> dict:

        tune_train_data, tune_test_data = train_test_data_setup(tune_data, 0.5)

        trained_model_default = fit(tune_test_data, default_params)
        prediction_default = predict(trained_model_default, tune_test_data)
        best_quality_metric = _regression_prediction_quality(prediction=prediction_default,
                                                             real=tune_test_data.target)
        best_params = default_params

        for _ in range(iterations):
            random_params = get_random_params(params_range)
            try:
                trained_model_candidate = fit(tune_train_data, random_params)
                prediction_candidate = predict(trained_model_candidate,
                                               tune_test_data)
                quality_metric = _regression_prediction_quality(prediction=prediction_candidate,
                                                                real=tune_test_data.target)
                if quality_metric < best_quality_metric:
                    best_params = random_params
            except ValueError:
                pass
        return best_params


def get_random_params(params_range):
    candidate_params = {}
    for param in params_range:
        param_range = params_range[param]
        param_range_left, param_range_right = param_range[0], param_range[1]
        if isinstance(param_range_left, tuple):
            # set-based params with constant length
            candidate_param = get_constant_length_range(param_range_left, param_range_right)
        elif isinstance(param_range_left, list):
            # set-based params with varied length
            candidate_param = get_varied_length_range(param_range_left, param_range_right)
        else:
            raise ValueError(f'Un-supported params range type {type(param_range_left)}')
        candidate_params[param] = candidate_param
    return candidate_params


def get_constant_length_range(left_range, right_range):
    candidate_param = []
    for sub_param_ind in range(len(left_range)):
        new_sub_param = randint(left_range[sub_param_ind],
                                right_range[sub_param_ind] + 1)
        candidate_param.append(new_sub_param)
    return tuple(candidate_param)


def get_varied_length_range(left_range, right_range):
    candidate_param = []
    subparams_num = randint(1, len(right_range))
    for sub_param_ind in range(subparams_num):
        new_sub_param = randint(left_range[sub_param_ind],
                                right_range[sub_param_ind] + 1)
        candidate_param.append(new_sub_param)
    return candidate_param


def _regression_prediction_quality(prediction, real):
    return mse(y_true=real, y_pred=prediction, squared=False)
