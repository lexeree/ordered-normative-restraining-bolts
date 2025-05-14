import random
import log


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


