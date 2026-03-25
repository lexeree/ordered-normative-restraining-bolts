from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np


class SimpleMerchantRBWrapper(Wrapper):
    def __init__(self, env, dfa_list=None):
        super().__init__(env)
        self.dfas = dfa_list if not None else []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        inpt = self.env.get_labels()
        for dfa in self.dfa_list:
            dfa.state = 0
            state = dfa.transition(inpt)
            dfa.state = state
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        inpt = self.env.get_labels()
        rb = []
        for dfa in self.dfa_list:
            state = dfa.transition(inpt)
            if state in dfa.final:
                rb.append(dfa.reward)
            else:
                dfa.state = state
                rb.append(0.0)
            dfa.state = dfa.reset(dfa.state, state)
        for r in rb:
            reward += r
        return observation, reward, terminated, truncated, info