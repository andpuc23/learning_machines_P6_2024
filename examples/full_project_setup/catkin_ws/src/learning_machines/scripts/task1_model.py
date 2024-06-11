class Model:
    def __init__(train_mode:bool,
                 action_space:list):
        pass

    def predict(self, observation) -> tuple(int, float):
        # predicts the action and outputs (action, probability of action)
        pass


    def save_checkpoint(self):
        pass