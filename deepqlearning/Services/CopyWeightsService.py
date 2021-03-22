from tensorflow.keras import Model


class CopyWeightsService:
    @staticmethod
    def execute(model_to_copy: Model, model_to_past: Model, tau: float = 1) -> None:
        variables_from_model_to_copy = model_to_copy.trainable_variables
        variables_from_model_to_past = model_to_past.trainable_variables

        for (variable_from_model_to_copy, variable_from_model_to_past) in zip(
            variables_from_model_to_copy, variables_from_model_to_past
        ):
            variable_from_model_to_past_numpy = variable_from_model_to_past.numpy()
            variable_from_model_to_copy_numpy = variable_from_model_to_copy.numpy()
            variable_from_model_to_past.assign(
                tau * variable_from_model_to_copy_numpy
                + (1 - tau) * variable_from_model_to_past_numpy
            )
