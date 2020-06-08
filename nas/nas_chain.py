from core.composer.chain import Chain
from core.models.data import InputData
from nas.keras_eval import create_nn_model, keras_model_fit, keras_model_predict


class NASChain(Chain):
    def __init__(self, nodes=None, fitted_model=None):
        super().__init__(nodes)
        self.model = fitted_model

    def __eq__(self, other) -> bool:
        return self is other

    def fit(self, input_data: InputData, verbose=False, input_shape: tuple = None,
            min_filters: int = None, max_filters: int = None, classes: int = 2, batch_size=24, epochs=15):
        if not self.model:
            self.model = create_nn_model(self, input_shape, min_filters, max_filters, classes)
        train_predicted = keras_model_fit(self.model, input_data, verbose=True, batch_size=batch_size, epochs=epochs)
        return train_predicted

    def predict(self, input_data: InputData):
        evaluation_result = keras_model_predict(self.model, input_data)
        return evaluation_result
