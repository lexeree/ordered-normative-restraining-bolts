from agents import Agent
import random
from collections import defaultdict
import log
from numpy.ma.extras import average
import matplotlib.pyplot as plt


class DecayedMORLRBNormAgent(Agent):
    def __init__(self, env, dist=False, specs=None, alpha=0.5, epsilon_init=0.5, epsilon_end=0.01, gamma=0.9, ntrain=10, export='', plot=False, seed=None):
        Agent.__init__(self, env, dist)
        self.name = 'MORL Restraining Bolt Norm Agent'
        self.logger = log.Log(self.name, self.env.map)
        self.qValues = {}
        self.qValues = defaultdict(lambda:0.0, self.qValues)
        self.alpha = alpha
        self.epsilon_init = epsilon_init
        self.epsilon_end = epsilon_end
        #self.decay_rate = (epsilon_init-epsilon_end)/ntrain
        self.decay_rate = (epsilon_end/epsilon_init)**(1/ntrain)
        self.cur_eps = epsilon_init
        self.gamma = gamma
        self.ntrain = ntrain
        self.plot = plot
        self. automata = specs
        self.allQValues = [self.qValues]
        self.weights = [1]
        for a in self.automata:
            aqValues = {}
            aqValues = defaultdict(lambda: 0.0, aqValues)
            self.allQValues.append(aqValues)
            self.weights.append(a.reward)
        if seed is not None:
            random.seed(seed)


    def getQValueVector(self, state, action, aStates):
        vec = []
        ast = ','.join(str(i) for i in aStates)
        for q in self.allQValues:
            vec.append(q[(ast, state, action)])
        return vec

    def computeValue(self, state, possible, aStates, revise):
        totals = []
        for a in possible:
            total = 0
            for i in range(len(self.allQValues)):
                if revise[i]:
                    total += self.getQValueVector(state, a, aStates)[i]*self.weights[i]
            totals.append(total)
        return max(totals)

    def computeValueVector(self, state, possible, aStates):
        vals = [float('-inf')]*len(self.allQValues)
        for a in possible:
            qs = self.getQValueVector(state, a, aStates)
            for i in range(len(vals)):
                if qs[i] > vals[i]:
                    vals[i] = self.getQValueVector(state, a, aStates)[i]
        return vals

    def policy(self, state, aStates, revise, train):
        possible = self.getLegalActions(state, train)
        val = self.computeValue(state, possible, aStates, revise)
        acts = []
        for act in possible:
            q = 0
            for i in range(len(self.allQValues)):
                if revise[i]:
                    q += self.getQValueVector(state, act, aStates)[i]*self.weights[i]
            if q == val:
                acts.append(act)
        return random.choice(acts)

    def act(self, state, aStates, revise, train=False):
        action = self.policy(state, aStates, revise, train)
        if train:
            if random.random() <= self.cur_eps:
                return random.choice(self.getLegalActions(state, train))
            else:
                return action
        else:
            return action

    def update(self, state0, action, state1, aStates0, aStates1, reward):
        ast = ','.join(str(i) for i in aStates0)
        curQ = self.getQValueVector(state0, action, aStates0)
        val = self.computeValueVector(state1, self.getPossibleActions(state1), aStates1)
        for i in range(len(self.allQValues)):
            self.allQValues[i][(ast, state0, action)] = ((1 - self.alpha) * curQ[i] + self.alpha * (reward[i] + self.gamma * val[i]))

    def test(self, rec=False, revs=None):
        self.env.reset()
        state = self.env.initialState()
        steps = 1
        aStates = [a.state0 for a in self.automata]
        while not state.final:
            if revs is not None:
                revise = revs(steps)
            else:
                revise = [True]*len(self.weights)
            act = self.act(state, aStates, revise)
            if rec:
                lst = []
                for p in self.getPossibleActions(state):
                    a = p + ' = ' + str(self.getQValueVector(state, p, aStates))
                    lst.append(a)
                self.logger.record_state(state, act, lst)
            nstate, r = self.env.stateTransition(state, act)
            oldaStates = aStates.copy()
            for i in range(len(aStates)):
                inpt = self.get_labels(state) + [act]
                aStates[i] = self.automata[i].transition(aStates[i], inpt)
                if aStates[i] in self.automata[i].final:
                    r += -1*self.automata[i].reward
                    if self.automata[i].achievement:
                        aStates[i] = 0
                    else:
                        aStates[i] = oldaStates[i]
            steps += 1
            state = nstate
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

    def train(self):
        rwds = []
        revise = [True]*len(self.weights)
        for n in range(self.ntrain):
            self.env.reset()
            state = self.env.initialState()
            reward = [0]*len(self.weights)
            aStates = [a.state0 for a in self.automata]
            while not state.final:
                oldaStates = aStates.copy()
                labs = self.get_labels(state)
                act = self.act(state, aStates, revise,  train=True)
                nstate, r = self.env.stateTransition(state, act)
                rwd = [r]
                inpt = labs + [act]
                for i in range(len(aStates)):
                    aStates[i] = self.automata[i].transition(aStates[i], inpt)
                    if aStates[i] in self.automata[i].final:
                        rwd.append(-1)
                        if self.automata[i].achievement:
                            aStates[i] = 0
                        else:
                            aStates[i] = oldaStates[i]
                    else:
                        rwd.append(0)
                self.update(state, act, nstate, oldaStates, aStates, rwd)
                for i in range(len(reward)):
                    reward[i] += rwd[i]*self.weights[i]
                state = nstate
            #self.cur_eps = max(0, self.cur_eps-self.decay_rate)
            self.cur_eps = max(self.epsilon_end, self.cur_eps*self.decay_rate)
            print('Episode '+str(n + 1)+ ' complete!')
            print('Total rewards: '+str(reward))
            print('Damage taken: ' + str(state.damage))
            print('Inventory value: ' + str(state.get_value()))
            rwds.append(reward)
        if self.plot:
            plots = []
            allrw = []
            for i in range(len(self.weights)):
                rw = []
                for j in range(self.ntrain):
                    rw.append(rwds[j][i])
                allrw.append(rw)
            for i in range(len(self.weights)):
                plot = []
                for j in range(int(self.ntrain / 50)):
                    avg = average(allrw[i][j * 50:(j + 1) * 50])
                    plot.append(avg)
                plots.append(plot)
            fig, axs = plt.subplots(len(self.weights))
            x = range(int(self.ntrain / 50))
            xs = [(p + 1) * 50 for p in x]
            for i in range(len(self.weights)):
                axs[i].plot(xs, plots[i])
            plt.show()
