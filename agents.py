import random
import log
from collections import defaultdict
from gymnasium.wrappers import TimeLimit
import numpy as np
import pickle
import log
from wrappers import SimpleMerchantRBWrapper


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
    def __init__(self, env, eval_env=None, ntrain=10000, gamma=0.99, alpha=0.2, epsilon=0.15):
        self.env = TimeLimit(env, 50)
        self.eval_env = eval_env
        self.ntrain = ntrain
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.logger = log.Log("QLearner")
        self.qvalues = defaultdict(lambda: np.zeros(self.env.action_space.n))     


    def update(self, state, action, nextState, reward, terminated):
        #print("qvals", self.qvalues[state][action])
        future_q_value = (not terminated) * np.max(self.qvalues[nextState])
        #print("future q", future_q_value)
        temporal_difference = reward + self.gamma * future_q_value - self.qvalues[state][action]
        #print("TD", temporal_difference)
        self.qvalues[state][action] = self.qvalues[state][action] + self.alpha * temporal_difference
        #print("qvals now", self.qvalues[state][action])

    def train(self, save=None):
        fname = save if not None else 'qvals'
        for i in range(self.ntrain):
            observation, info = self.env.reset()
            episode_over = False
            while not episode_over:
                acts = self.env.unwrapped.exclActions()
                if random.random() < self.epsilon:
                    action = random.choice([a for a in range(self.env.action_space.n) if a not in acts])
                else:
                    opts = self.qvalues[observation]
                    filtered = np.array([-1*np.inf if a in acts else opts[a] for a in range(self.env.action_space.n)])
                    action = int(np.argmax(filtered))
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

    def evaluate(self, runs=1, load_model=None, record=True):
        evaluation = []
        if load_model is None:
            qvalues = self.qvalues
        else:
            qvalues = load_model
        for i in range(runs):
            observation, info = self.env.reset()
            episode_over = False
            while not episode_over:
                acts = self.env.unwrapped.exclActions()
                opts = self.qvalues[observation]
                filtered = np.array([-1*np.inf if a in acts else opts[a] for a in range(self.env.action_space.n)])
                action = int(np.argmax(filtered))
                self.logger.record_state_g(observation, action)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                episode_over = terminated or truncated
                observation = next_observation
                if episode_over and record:
                    self.logger.export_trace_g()
            i += 1
            print(i, " episodes complete")
        self.env.close()
        return evaluation
    
#TODO: NEED TO INCLUDE ASTATES IN QVALUE TABLES/OBSERVATIONS
class RBAgent(QLearner):
    def __init__(self, env, eval_env=None, dfa_list=None, ntrain=10000, gamma=0.9, alpha=0.2, epsilon=0.15):
        QLearner.__init__(self, env, eval_env, ntrain, gamma, alpha, epsilon)
        self.env = SimpleMerchantRBWrapper(TimeLimit(env, 50), dfa_list)
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
                if random.random() < self.epsilon:
                    action = self.env.unwrapped.action_space.sample()
                else:
                    action = int(np.argmax(self.qvalues[observation]))
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

    def evaluate(self, runs=1, load_model=None, record=True):
        evaluation = []
        if load_model is None:
            qvalues = self.qvalues
        else:
            qvalues = load_model
        for i in range(runs):
            observation, info = self.env.reset()
            episode_over = False
            while not episode_over:
                action = int(np.argmax(qvalues[observation]))
                self.logger.record_state_g(observation, action)
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
        self.env = SimpleMerchantRBWrapper(TimeLimit(env, 50), dfa_list)
        self.name = 'ONRB Agent (scalarization)'
        self.logger = log.Log(self.name)
        self.qvalues = defaultdict(lambda: np.zeros(self.env.action_space.n)) 
        self.dfas = dfa_list if not None else []
        self.allQValues = [self.qvalues]
        self.weights = [1]
        for a in self.automata:
            aqvalues = defaultdict(lambda: np.zeros(self.env.action_space.n)) 
            self.allQValues.append(aqvalues)
            self.weights.append(a.reward)

    def getQVec(self, state, action, aStates):
        vec = []
        ast = ','.join(str(i) for i in aStates)
        for q in self.allQValues:
            vec.append(q[(ast, state, action)])
        return vec

    def train(self, save=None):
        fname = save if not None else 'qvals'
        for i in range(self.ntrain):
            observation, info = self.env.reset()
            episode_over = False
            while not episode_over:
                if random.random() < self.epsilon:
                    action = self.env.unwrapped.action_space.sample()
                else:
                    action = int(np.argmax(self.qvalues[observation]))
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

    def evaluate(self, runs=1, load_model=None, record=True):
        evaluation = []
        if load_model is None:
            qvalues = self.qvalues
        else:
            qvalues = load_model
        for i in range(runs):
            observation, info = self.env.reset()
            episode_over = False
            while not episode_over:
                action = int(np.argmax(qvalues[observation]))
                self.logger.record_state_g(observation, action)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                episode_over = terminated or truncated
                observation = next_observation
                if episode_over and record:
                    self.logger.export_trace_g()
            i += 1
            print(i, " episodes complete")
        self.env.close()
        return evaluation
    

