from numpy.ma.extras import average
from agents import Agent
import random
from collections import defaultdict
import log
import matplotlib.pyplot as plt


class QAgent(Agent):
    def __init__(self, env, dist=False, filter=None, occ=False, ngrl=False, alpha=0.5, epsilon=0.05, gamma=0.9, ntrain=10, export=''):
        Agent.__init__(self, env, dist, filter, occ)
        self.name = 'Q-learning Agent'
        self.logger = log.Log(self.name, self.env.map)
        self.qValues = {}
        self.qValues = defaultdict(lambda:0.0, self.qValues)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.learn = ngrl
        self.ntrain = ntrain

    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def computeValue(self, state, possible):
        #print(self.getPossibleActions(state))
        qvals = [self.getQValue(state, a) for a in possible]
        return max(qvals)

    def policy(self, state, train):
        possible = self.getLegalActions(state, train)
        v = self.computeValue(state, possible)
        acts = []
        for act in possible:
            if self.getQValue(state, act) == v:
                acts.append(act)
        return random.choice(acts)

    def act(self, state, train=False):
        action = self.policy(state, train)
        if train:
            if random.random() <= self.epsilon:
                return random.choice(self.getLegalActions(state, train))
            else:
                return action
        else:
            return action

    def update(self, state0, action, state1, reward):
        curQ = self.getQValue(state0, action)
        self.qValues[(state0, action)] = (1 - self.alpha) * curQ + self.alpha * (
                reward + self.gamma * self.computeValue(state1, self.getPossibleActions(state1)))

    def test(self, rec=False):
        self.env.reset()
        state = self.env.initialState()
        steps = 1
        act = []
        lst = []
        while not state.final:
            act = self.act(state)
            if rec:
                lst = []
                for p in self.getPossibleActions(state):
                    a = p + ' = ' + str(self.getQValue(state, p))
                    lst.append(a)
                self.logger.record_state(state, act, lst)
            state, r = self.env.stateTransition(state, act)
            steps += 1
        self.logger.record_state(state, act, lst)
        if rec:
            self.logger.export_trace()
        return steps, len(state.inventory), state.get_value(), state.damage

    def run(self, n, rec=False):
        for i in range(n):
            time, mass, value, damage = self.test(rec=False)
            if rec:
                self.logger.add_summary(i, time, mass, value, damage)
        if rec:
            self.logger.export_summary()


class RewardQAgent(QAgent):
    def __init__(self, env, dist=False, filter=None, occ=False, ngrl=False, alpha=0.5, epsilon=0.05, gamma=0.9, ntrain=100, export='', plot=False):
        QAgent.__init__(self, env, dist, filter, occ, ngrl, alpha, epsilon, gamma, ntrain, export)
        self.name = 'Reward Q-learning Agent'
        self.logger = log.Log(self.name, self.env.map)
        self.plot = plot

    def train(self):
        rwds = []
        for n in range(self.ntrain):
            self.env.reset()
            state = self.env.initialState()
            reward = 0
            while not state.final:
                act = self.act(state, train=True)
                nstate, r = self.env.stateTransition(state, act)
                self.update(state, act, nstate, r)
                reward += r
                state = nstate
            print('Episode '+str(n + 1)+ ' complete!')
            print('Total rewards: '+str(reward))
            print('Damage taken: ' + str(state.damage))
            print('Inventory value: ' + str(state.get_value()))
            rwds.append(reward)
        if self.plot:
            plot = []
            for i in range(int(self.ntrain / 10 - 1)):
                avg = average(rwds[i * 10:(i + 1) * 10])
                plot.append(avg)
            x = range(int(self.ntrain / 10 - 1))
            xs = [p * 10 for p in x]
            plt.plot(xs, plot)
            plt.show()


