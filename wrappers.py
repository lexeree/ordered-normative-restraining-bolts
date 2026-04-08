from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np


class SimpleMerchantRBWrapper(Wrapper):
    def __init__(self, env, dfa_list=None):
        super().__init__(env)
        self.dfa_list = dfa_list if not None else []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for dfa in self.dfa_list:
            dfa.state = 0
        return obs, info

    def step(self, action):
        inpt = self.env.get_labels()
        observation, reward, terminated, truncated, info = self.env.step(action)
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
    
class MOMerchantRBWrapper(SimpleMerchantRBWrapper):
    def __init__(self, env, dfa_list=None):
        super().__init__(env, dfa_list)
        
    def step(self, action):
        inpt = self.env.get_labels()
        observation, reward, terminated, truncated, info = self.env.step(action)
        rb = []
        for dfa in self.dfa_list:
            state = dfa.transition(inpt)
            if state in dfa.final:
                rb.append(dfa.reward)
            else:
                dfa.state = state
                rb.append(0.0)
            dfa.state = dfa.reset(dfa.state, state)
        reward = np.array([reward] + rb)
        return observation, reward, terminated, truncated, info