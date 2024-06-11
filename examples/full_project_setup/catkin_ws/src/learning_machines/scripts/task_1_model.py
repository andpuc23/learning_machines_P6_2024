#!/usr/bin/env python3
class Model:
    def __init__(self, 
                 train_mode:bool,
                 action_space:list):
        pass

    def predict(self, observation) -> tuple:
        # predicts the action and outputs (action, probability of action)
        pass


    def save_checkpoint(self):
        pass