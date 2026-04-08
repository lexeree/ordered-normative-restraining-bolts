import random
import log
from collections import defaultdict
from gymnasium.wrappers import TimeLimit
import numpy as np
import pickle
import log
from wrappers import SimpleMerchantRBWrapper, MOMerchantRBWrapper


obj_map = {
    'T': 'tree',
    'O': 'ore',
    'D': 'danger',
    'W': 'wood',
    'R': 'rock',
    'H': 'home',
    'M': 'market',
    'E': 'extracted'
}


class Agent:
    def __init__(self, env, dist=False, filter=None, occ=False, export=''):
        self.act_map = {
    0: 'Fight',
    1: 'North',
    2: 'South',
    3: 'East',
    4: 'West',
    5: 'Extract',
    7: 'Unload'
}
        self.actions = list(self.act_map.values())
        self.env = env
        self.name = 'Default Agent'
        self.filter = filter
        self.supervise = occ
        self.dist = dist
        self.logger = log.Log(self.name, self.env.map)

    def get_labels(self, state):
        labels = []
        cellN = cellS = cellE = cellW = ''

        cell = self.env.map[(state.x, state.y)]

        if state.y > 0:
            cellN = self.env.map[(state.x, state.y-1)]

        if state.y < self.env.y - 1:
            cellS = self.env.map[(state.x, state.y+1)]

        if state.x < self.env.x - 1:
            cellE = self.env.map[(state.x+1, state.y)]

        if state.x > 0:
            cellW = self.env.map[(state.x-1, state.y)]

        if cell in obj_map.keys():
            labels.append('at_'+obj_map[cell])
        if cellN in obj_map.keys():
            labels.append('North_'+obj_map[cellN])
        if cellS in obj_map.keys():
            labels.append('South_'+obj_map[cellS])
        if cellE in obj_map.keys():
            labels.append('East_'+obj_map[cellE])
        if cellW in obj_map.keys():
            labels.append('West_'+obj_map[cellW])
        if state.attack:
            labels.append('attacked')
        if 'W' in state.inventory:
            labels.append('has_'+obj_map['W'])
        if 'O' in state.inventory:
            labels.append('has_'+obj_map['O'])
        if state.last_move:
            labels.append('from_'+state.last_move)
        if state.sundown:
            labels.append('sundown')
        return labels


    def getPossibleActions(self, state):
        actions = self.actions.copy()
        if state.final:
            return actions
        cell = state.label
        dist = self.env.distFromDest(state.x, state.y)
        if self.dist:
            if self.env.map[(state.x, state.y-1)] == 'X' or self.env.distFromDest(state.x, state.y - 1) > dist:
                actions.remove('North')
            if self.env.map[(state.x, state.y+1)] == 'X' or self.env.distFromDest(state.x, state.y + 1) > dist:
                actions.remove('South')
            if self.env.map[(state.x+1, state.y)] == 'X' or self.env.distFromDest(state.x + 1, state.y) > dist:
                 actions.remove('East')
            if self.env.map[(state.x-1, state.y)] == 'X' or self.env.distFromDest(state.x - 1, state.y) > dist:
                 actions.remove('West')
        else:
            if self.env.map[(state.x, state.y - 1)] == 'X' or state.last_move == 'South':
                actions.remove('North')
            if self.env.map[(state.x, state.y + 1)] == 'X' or state.last_move == 'North':
                actions.remove('South')
            if self.env.map[(state.x + 1, state.y)] == 'X' or state.last_move == 'West':
                actions.remove('East')
            if self.env.map[(state.x - 1, state.y)] == 'X' or state.last_move == 'East':
                actions.remove('West')
        if cell not in ['T', 'R'] or len(state.inventory) >= self.env.capacity:
            actions.remove('Extract')
        if not state.inventory: #or (not state.attack and cell != 'M'):
            actions.remove('Unload')
        if not state.attack:
            actions.remove('Fight')
        return actions

    def getLegalActions(self, state, train=False):
        if not train and self.supervise and self.filter is not None:
            rec = self.filter.filter(state, self.get_labels(state), self.getPossibleActions(state))
            actions = rec
        else:
            actions = self.getPossibleActions(state)
        return actions

    def act(self, state):
        return 'North'

class RandomAgent(Agent):
    def __init__(self, env, filter=None, occ=False):
        Agent.__init__(self, env, filter, occ)
        self.name = 'RandomAgent'
        self.logger = log.Log(self.name, self.env.map)

    def act(self, state):
        possible = self.getLegalActions(state)
        action = random.choice(possible)
        return action
    

class QLearner():
    def __init__(self, env, eval_env=None, ntrain=10000, gamma=0.9, alpha=0.2, epsilon=0.15):
        self.env = TimeLimit(env, 35)
        self.eval_env = eval_env
        self.ntrain = ntrain
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.logger = log.Log("QLearner")
        self.qvalues = defaultdict(lambda: np.zeros(self.env.action_space.n))     


    def update(self, state, action, nextState, reward, terminated):
        future_q_value = (not terminated) * np.max(self.qvalues[nextState])
        temporal_difference = reward + self.gamma * future_q_value - self.qvalues[state][action]
        self.qvalues[state][action] += self.alpha * temporal_difference

    def policy(self, state, egreedy=False):
        acts = self.env.unwrapped.exclActions()
        if egreedy and random.random() < self.epsilon:
            action = random.choice([a for a in range(self.env.action_space.n) if a not in acts])
        else:
            opts = self.qvalues[state]
            filtered = np.array([-1*np.inf if a in acts else opts[a] for a in range(self.env.action_space.n)])
            action = int(np.argmax(filtered))
        return action

    def train(self, save=None):
        fname = save if not None else 'qvals'
        for i in range(self.ntrain):
            observation, info = self.env.reset()
            episode_over = False
            while not episode_over:
                action = self.policy(observation, egreedy=True)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                self.update(observation, action, next_observation, reward, terminated)
                episode_over = terminated or truncated
                observation = next_observation
            i += 1
            if i % 1000 == 0:
                print(i, "episodes complete") 
        if save is not None:
            with open(fname+'.p', 'bw') as f:
                qvals = dict(self.qvalues.copy())
                pickle.dump(qvals, f)   

    def evaluate(self, runs=1, record=True):
        evaluation = []
        for i in range(runs):
            observation, info = self.env.reset()
            episode_over = False
            while not episode_over:
                action = self.policy(observation)
                self.logger.record_state_g(observation, action, self.qvalues[observation])
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                episode_over = terminated or truncated
                observation = next_observation
                if episode_over and record:
                    self.logger.export_trace_g()
            i += 1
            print(i, " episodes complete")
        self.env.close()
        return evaluation
    

class RBAgent(QLearner):
    def __init__(self, env, eval_env=None, dfa_list=None, ntrain=10000, gamma=0.9, alpha=0.2, epsilon=0.15):
        QLearner.__init__(self, env, eval_env, ntrain, gamma, alpha, epsilon)
        self.env = TimeLimit(SimpleMerchantRBWrapper(env, dfa_list), 35)
        self.name = 'Restraining Bolt Agent'
        self.logger = log.Log(self.name)
        self.qvalues = defaultdict(lambda: np.zeros(self.env.action_space.n)) 
        self.dfas = dfa_list if not None else []

    def train(self, save=None):
        fname = save if not None else 'qvals'
        for i in range(self.ntrain):
            observation, info = self.env.reset()
            episode_over = False
            while not episode_over:
                total_obs = (observation, )
                for dfa in self.dfas:
                    total_obs = total_obs + (dfa.state, )
                action = self.policy(total_obs, egreedy=True)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                n_total_obs = (next_observation, )
                for dfa in self.dfas:
                    n_total_obs = n_total_obs + (dfa.state, )
                self.update(total_obs, action, n_total_obs, reward, terminated)
                episode_over = terminated or truncated
                observation = next_observation
            i += 1
            if i % 1000 == 0:
                print(i, "episodes complete") 
        if save is not None:
            with open(fname+'.p', 'bw') as f:
                qvals = dict(self.qvalues.copy())
                pickle.dump(qvals, f)      

    def evaluate(self, runs=1, record=True):
        evaluation = []
        for i in range(runs):
            observation, info = self.env.reset()
            episode_over = False
            while not episode_over:
                total_obs = (observation, )
                for dfa in self.dfas:
                    total_obs = total_obs + (dfa.state, )
                action = self.policy(total_obs)
                self.logger.record_state_g(observation, action, self.qvalues[total_obs])
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                episode_over = terminated or truncated
                observation = next_observation
                if episode_over and record:
                    self.logger.export_trace_g()
            i += 1
            print(i, " episodes complete")
        self.env.close()
        return evaluation


class ONRBAgent1(QLearner):
    def __init__(self, env, eval_env=None, dfa_list=None, ntrain=10000, gamma=0.9, alpha=0.2, epsilon=0.15):
        QLearner.__init__(self, env, eval_env, ntrain, gamma, alpha, epsilon)
        self.env = MOMerchantRBWrapper(TimeLimit(env, 50), dfa_list)
        self.name = 'ONRB Agent (scalarization)'
        self.logger = log.Log(self.name)
        self.qvalues = defaultdict(lambda: np.zeros(self.env.action_space.n)) 
        self.dfas = dfa_list if not None else []
        self.allQValues = [self.qvalues]
        self.weights = [1]
        for a in self.dfas:
            aqvalues = defaultdict(lambda: np.zeros(self.env.action_space.n)) 
            self.allQValues.append(aqvalues)
            self.weights.append(a.reward)

    def getQValueVector(self, state, action):
        vec = []
        for q in self.allQValues:
            vec.append(q[state][action])
        return np.array(vec)

    def computeValueVector(self, state):
        excls = self.env.unwrapped.exclActions()
        possible = [a for a in range(self.env.action_space.n) if a not in excls]
        vals = [float('-inf')]*len(self.allQValues)
        for a in possible:
            qs = self.getQValueVector(state, a)
            for i in range(len(vals)):
                if qs[i] > vals[i]:
                    vals[i] = qs[i]
        return np.array(vals)
    
    def update(self, state, action, nextState, reward, terminated):
        future_q_values = np.zeros(len(self.allQValues)) if terminated else self.computeValueVector(nextState)
        temporal_difference = reward + self.gamma * future_q_values - self.getQValueVector(state, action)
        for i in range(len(self.allQValues)):
            self.allQValues[i][state][action] += self.alpha * temporal_difference[i]

    def policy(self, state, egreedy=False):
        acts = self.env.unwrapped.exclActions()
        if egreedy and random.random() < self.epsilon:
            action = random.choice([a for a in range(self.env.action_space.n) if a not in acts])
        else:
            excls = self.env.unwrapped.exclActions()
            values = []
            for a in range(self.env.action_space.n):
                qvec = self.getQValueVector(state, a)
                total = 0
                for i in range(len(self.allQValues)):
                    total += qvec[i]*self.weights[i]
                    values.append(total)
            filtered = np.array([-1*np.inf if a in excls else values[a] for a in range(self.env.action_space.n)])
            action = int(np.argmax(filtered))
        return action
     

class ONRBAgent1(QLearner):
    def __init__(self, env, eval_env=None, dfa_list=None, ntrain=10000, gamma=0.9, alpha=0.2, epsilon=0.15):
        QLearner.__init__(self, env, eval_env, ntrain, gamma, alpha, epsilon)
        self.env = MOMerchantRBWrapper(TimeLimit(env, 50), dfa_list)
        self.name = 'ONRB Agent (scalarization)'
        self.logger = log.Log(self.name)
        self.qvalues = defaultdict(lambda: np.zeros(self.env.action_space.n)) 
        self.dfas = dfa_list if not None else []
        self.allQValues = []
        self.weights = [1]
        for a in self.dfas:
            aqvalues = defaultdict(lambda: np.zeros(self.env.action_space.n)) 
            self.allQValues.append(aqvalues)
            self.weights.append(a.reward)
        self.allQValues += [self.qvalues]

    def getQValueVector(self, state, action):
        vec = []
        for q in self.allQValues:
            vec.append(q[state][action])
        return np.array(vec)

    def computeValueVector(self, state):
        excls = self.env.unwrapped.exclActions()
        possible = [a for a in range(self.env.action_space.n) if a not in excls]
        vals = [float('-inf')]*len(self.allQValues)
        for a in possible:
            qs = self.getQValueVector(state, a)
            for i in range(len(vals)):
                if qs[i] > vals[i]:
                    vals[i] = qs[i]
        return np.array(vals)
    
    def update(self, state, action, nextState, reward, terminated):
        future_q_values = np.zeros(len(self.allQValues)) if terminated else self.computeValueVector(nextState)
        temporal_difference = reward + self.gamma * future_q_values - self.getQValueVector(state, action)
        for i in range(len(self.allQValues)):
            self.allQValues[i][state][action] += self.alpha * temporal_difference[i]

    def policy(self, state, egreedy=False):
        if egreedy and random.random() < self.epsilon:
            action = random.choice([a for a in range(self.env.action_space.n) if a not in acts])
        else:
            excls = self.env.unwrapped.exclActions()
            values = []
            for a in range(self.env.action_space.n):
                qvec = self.getQValueVector(state, a)
                total = 0
                for i in range(len(self.allQValues)):
                    total += qvec[i]*self.weights[i]
                    values.append(total)
            filtered = np.array([-1*np.inf if a in excls else values[a] for a in range(self.env.action_space.n)])
            action = int(np.argmax(filtered))
        return action


class ONRBAgent2(QLearner):
    def __init__(self, env, eval_env=None, dfa_list=None, ntrain=10000, gamma=0.9, alpha=0.2, epsilon=0.15):
        QLearner.__init__(self, env, eval_env, ntrain, gamma, alpha, epsilon)
        self.env = MOMerchantRBWrapper(TimeLimit(env, 50), dfa_list)
        self.name = 'ONRB Agent (TLQL)'
        self.logger = log.Log(self.name)
        self.qvalues = defaultdict(lambda: np.zeros(self.env.action_space.n)) 
        self.dfas = dfa_list if not None else []
        self.allQValues = []
        for a in self.dfas:
            aqvalues = defaultdict(lambda: np.zeros(self.env.action_space.n)) 
            self.allQValues.append(aqvalues)
        self.allQValues += [self.qvalues]

    def getQValueVector(self, state, action):
        vec = []
        for q in self.allQValues:
            vec.append(q[state][action])
        return np.array(vec)

    def computeValueVector(self, state):
        excls = self.env.unwrapped.exclActions()
        possible = [a for a in range(self.env.action_space.n) if a not in excls]
        vals = [float('-inf')]*len(self.allQValues)
        for a in possible:
            qs = self.getQValueVector(state, a)
            for i in range(len(vals)):
                if qs[i] > vals[i]:
                    vals[i] = qs[i]
        return np.array(vals)
    
    def update(self, state, action, nextState, reward, terminated):
        future_q_values = np.zeros(len(self.allQValues)) if terminated else self.computeValueVector(nextState)
        temporal_difference = reward + self.gamma * future_q_values - self.getQValueVector(state, action)
        for i in range(len(self.allQValues)):
            self.allQValues[i][state][action] += self.alpha * temporal_difference[i]

    def policy(self, state, egreedy=False):
        acts = self.env.unwrapped.exclActions()
        if egreedy and random.random() < self.epsilon:
            action = random.choice([a for a in range(self.env.action_space.n) if a not in acts])
        else:
            excls = self.env.unwrapped.exclActions()
            possible = [a for a in range(self.env.action_space.n) if a not in excls]
            vvec = self.computeValueVector(state)
            for i in len(vvec):
                newpos = []
                for a in possible:
                    if self.allQValues[i][state][a] >= vvec[i]:
                        newpos.append(a)
                possible = newpos
        return random.choice(possible)
