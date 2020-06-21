from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        value=currentGameState.getScore()
        successorFoodList=newFood.asList()
        currentFoodList=currentGameState.getFood().asList()
        distanceToFood = []
        if len(successorFoodList) < len(currentFoodList):
            value += 10
        if len(successorFoodList):
            for food in successorFoodList:
                distanceToFood.append(manhattanDistance(newPos,food))
                value += 1.5/(manhattanDistance(newPos,food)+0.01)
            value += 2/min(distanceToFood)
        ghostPositions=[]
        for ghost in newGhostStates:
            ghostPositions.append(ghost.getPosition())
        if newPos in ghostPositions:
            value -= 1000
        for ghostPos in ghostPositions:
            if newScaredTimes[ghostPositions.index(ghostPos)] > 0:
                value+=10/(manhattanDistance(newPos,ghostPos)+0.01)
            else:
                value-=15/(manhattanDistance(newPos,ghostPos)+0.01)
        return value
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            value = float("-inf")
            for action in state.getLegalActions(agentIndex):
                value = max(value, minValue(state.generateSuccessor(0, action), depth, agentIndex + 1))
            return value

        agentNum = gameState.getNumAgents()
        def minValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            value = float("inf")
            for action in state.getLegalActions(agentIndex):
                if agentIndex < agentNum - 1:
                    value = min(value, minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
                else:
                    value = min(value, maxValue(state.generateSuccessor(agentIndex, action), depth - 1, 0))
            return value

        values = util.Counter()
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            values[legalActions.index(action)] = minValue(gameState.generateSuccessor(0, action), self.depth, 1)
        return legalActions[values.argMax()]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            value = float("-inf")
            for action in state.getLegalActions(agentIndex):
                value = max(value, minValue(state.generateSuccessor(0, action), depth, agentIndex + 1, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        agentNum = gameState.getNumAgents()
        def minValue(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            value = float("inf")
            for action in state.getLegalActions(agentIndex):
                if agentIndex < agentNum - 1:
                    value = min(value, minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                else:
                    value = min(value, maxValue(state.generateSuccessor(agentIndex, action), depth - 1, 0, alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        initialAlpha=float("-inf")
        initialBeta=float("inf")
        values = util.Counter()
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            values[legalActions.index(action)] = minValue(gameState.generateSuccessor(0, action), self.depth, 1, initialAlpha, initialBeta)
            initialAlpha = max(initialAlpha, values[values.argMax()])
        return legalActions[values.argMax()]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        agentNum = gameState.getNumAgents()
        def expectimaxValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            value = float("-inf")
            avgValue = 0.0
            for action in state.getLegalActions(agentIndex):
                if agentIndex == 0:
                    value = max(value, expectimaxValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
                else:
                    if agentIndex < agentNum - 1:
                        avgValue += expectimaxValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
                    else:
                        avgValue += expectimaxValue(state.generateSuccessor(agentIndex, action), depth - 1, 0)
                    value = avgValue/len(state.getLegalActions(agentIndex))
            return value

        values = util.Counter()
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            values[legalActions.index(action)] = expectimaxValue(gameState.generateSuccessor(0, action), self.depth, 1)
        return legalActions[values.argMax()]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #from search import breadthFirstSearch
    from searchAgents import PositionSearchProblem, optimalPath

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    value = currentGameState.getScore()
    foodList = newFood.asList()

    minFoodDistance = 10000
    avgFoodDistance = 0.0
    for food in foodList:
        if len(foodList) < 5:
            foodDistance = optimalPath(newPos, food, currentGameState)
        else:
            foodDistance = manhattanDistance(newPos, food)
        avgFoodDistance += foodDistance
        avgFoodDistance = min(minFoodDistance, foodDistance)
    avgFoodDistance = avgFoodDistance / float(len(foodList) + 0.1)
    value -= (0.9 * minFoodDistance + 0.1 * avgFoodDistance) / 2.0 + len(foodList)
    minGhostDistance = 10000
    for ghost in newGhostStates:
        if ghost.scaredTimer == 0:
            minGhostDistance = min(minGhostDistance, optimalPath(newPos, ghost.getPosition(), currentGameState))
    value -= 5.0 / (minGhostDistance + 1.0)
    value -= 50.0 * len(currentGameState.getCapsules()) + 0.2 * sum(newScaredTimes)
    return value
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
