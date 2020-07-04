import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            Values = util.Counter()
            for each in states:
                QValues = []
                possibleActions = self.mdp.getPossibleActions(each)
                for action in possibleActions:
                    QValues.append(self.computeQValueFromValues(each, action))
                if len(QValues):
                    Values[each] = max(QValues)
            self.values = Values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0.0
        for each in transitions:
            QValue += each[1] * (self.mdp.getReward(state, action, each[0]) + self.discount * self.getValue(each[0]))
        return QValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.mdp.getPossibleActions(state)
        Values = util.Counter()
        if len(possibleActions):
            for action in possibleActions:
                Values[possibleActions.index(action)] = self.computeQValueFromValues(state, action)
            return possibleActions[Values.argMax()]
        else:
            return None
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        stateNum = len(states)
        for i in range(self.iterations):
            currentState = states[i % stateNum]
            Values = util.Counter()
            if not self.mdp.isTerminal(currentState):
                possibleActions = self.mdp.getPossibleActions(currentState)
                QValues = []
                for action in possibleActions:
                    QValues.append(self.computeQValueFromValues(currentState, action))
                if len(QValues):
                    self.values[currentState] = max(QValues)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = {}
        for each in states:
            predecessors[each] = set()
        for each in states:
            possibleActions = self.mdp.getPossibleActions(each)
            for action in possibleActions:
                transitions = self.mdp.getTransitionStatesAndProbs(each, action)
                for T in transitions:
                    predecessors[T[0]].add(each)

        priorityQueue = util.PriorityQueue()
        for each in states:
            if not self.mdp.isTerminal(each):
                possibleActions = self.mdp.getPossibleActions(each)
                QValues = []
                for action in possibleActions:
                    QValues.append(self.computeQValueFromValues(each, action))
                if len(QValues):
                    diff = abs(max(QValues)-self.values[each])
                    priorityQueue.update(each, -diff)

        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                break
            currentState = priorityQueue.pop()
            if not self.mdp.isTerminal(currentState):
                possibleActions = self.mdp.getPossibleActions(currentState)
                QValues = []
                for action in possibleActions:
                    QValues.append(self.computeQValueFromValues(currentState, action))
                if len(QValues):
                    self.values[currentState] = max(QValues)
            predecessorsList = list(predecessors[currentState])
            for p in predecessorsList:
                if not self.mdp.isTerminal(p):
                    possibleActions = self.mdp.getPossibleActions(p)
                    QValues = []
                    for action in possibleActions:
                        QValues.append(self.computeQValueFromValues(p, action))
                    if len(QValues):
                        diff = abs(max(QValues)-self.values[p])
                        print
                    else:
                        diff = 0.0
                    if diff > self.theta:
                        priorityQueue.update(p, -diff)
