

class DFA:
    def __init__(self, reward=0.0, reset="punctual", permission=False):
        self.state0 = 0
        self.reward = reward
        self.states = []
        self.final = []
        self.type = reset
        self.permission = permission
        self.state = 0

    def transition(self, inpt, state):
        pass

    def reset(self, prevState, state):
        if state in self.final:
            if self.type == "maintenance":
                return prevState
            elif self.type in ["achievement", "punctual"]:
                return self.state0
            else:
                return state
        return state
    


class DangerDFA(DFA):
    def __init__(self, reward=0):
        super().__init__(-reward, reset="punctual", permission=False)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "atDanger" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1


class PassiveDFA(DFA):
    def __init__(self, reward=0):
        super().__init__(-reward, reset="punctual", permission=False)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "Fight" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1


class EnvFriendlyDFA(DFA):
    def __init__(self, reward=0):
        super().__init__(-reward, reset="punctual", permission=False)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "atTree" in inpt and "Extract" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1


# O^A_sundown(atMarket | atHome)
# F(atHome & !atMarket U sundown)
class DeliveryDFA(DFA):
    def __init__(self, reward=0):
        super().__init__(-reward, reset="achievement", permission=False)
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "atHome" in inpt and "sundown" in inpt:
                return 2
            elif "atHome" in inpt and "atMarket" not in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "sundown" in inpt:
                return 2
            elif "atMarket" not in inpt:
                return 1
            else:
                return 0
        if state == 2:
            return 2


# TODO: permission DFAs
class EnvPerm(DFA):
    def __init__(self, reward=0):
        super().__init__(reward, reset="punctual", permission=True)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "atTree" in inpt and "extract" in inpt and "hasWood" not in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1