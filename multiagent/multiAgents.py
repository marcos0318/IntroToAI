# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore
        newFood = newFood.asList()
        if len(newFood) == 0:
            return 10000
        foodScore = min(map( lambda pos : abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1]), newFood))

        dangerousRange = 2

        ghostPositions = map( lambda state : state.getPosition(), newGhostStates)
        nearestGhostDist = min(map( lambda pos : abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1]), ghostPositions))

        if nearestGhostDist <= dangerousRange:
            ghostScore = -10000000
        else:
            ghostScore = 0

        foodNumber = len(newFood) * 1000

        return - foodNumber - foodScore + ghostScore



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
        self.rootActionIndex = 0

    def setRootAction(self, num):
        self.rootActionIndex = num

    def getRootAction(self):
        return self.rootActionIndex

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
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()

        states = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [self.MinimaxDispatch(state, self.depth) for state in states]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0]

        return legalMoves[chosenIndex]

    def MinimaxDispatch(self, initState, depth):
        return self.MinSearch(initState, depth, 1)

    def MaxSearch(self, state, depth):
        # check if the depth is 0, if so return the value of evaluation function
        if depth <= 1:
            return self.evaluationFunction(state)
        # reduce the depth by 1
        depth = depth -1

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # get the actions of the pacman
        pacActions = state.getLegalActions(0)

        # nextStates = map( lambda action: state.generateSuccessor(0, action), pacActions)
        nextStates = [state.generateSuccessor(0, action) for action in pacActions]
        score = max([self.MinSearch(state, depth, 1) for state in nextStates])
        return score

    def MinSearch(self, state, depth, numMin):

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        Actions = state.getLegalActions(numMin)
        nextStates = [state.generateSuccessor(numMin, action) for action in Actions]
        numMin = numMin + 1

        if numMin == state.getNumAgents():
            score = min([self.MaxSearch(state, depth) for state in nextStates])
        else:
            score = min([self.MinSearch(state, depth, numMin) for state in nextStates])

        return score



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
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
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()

        self.MinimaxDispatch(gameState, self.depth)

        chosenIndex = self.getRootAction()

        return legalMoves[chosenIndex]


    def MinimaxDispatch(self, initState, depth):
        return self.MaxSearch(initState, depth)

    def MaxSearch(self, state, depth, a=float("-inf"), b=float("inf")):
        # check if the depth is 0, if so return the value of evaluation function
        if depth <= 0:
            return self.evaluationFunction(state)
        # reduce the depth by 1
        depth = depth - 1

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        v = float("-inf")

        # get the actions of the pacman
        pacActions = state.getLegalActions(0)

        localA = a
        localB = b

        for eachAction in pacActions:
            eachState = state.generateSuccessor(0, eachAction)
            minResult = self.MinSearch(eachState, depth, 1, localA, localB)
            if v < minResult:
                v = minResult
                if depth == self.depth - 1:
                    self.setRootAction(pacActions.index(eachAction))
            if v > b:
                return v
            localA = max(localA, v)
        return v

    def MinSearch(self, state, depth, numMin, a=float("-inf"), b=float("inf")):

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        Actions = state.getLegalActions(numMin)
        # nextStates = [state.generateSuccessor(numMin, action) for action in Actions]
        numMin = numMin + 1
        localA = a
        localB = b

        if numMin == state.getNumAgents():
            v = float("inf")
            # score = min([self.MaxSearch(state, depth) for state in nextStates])
            for eachAction in Actions:
                eachState = state.generateSuccessor(numMin - 1, eachAction)
                v = min(v, self.MaxSearch(eachState, depth, localA, localB))
                if v < localA:
                    return v
                localB = min(localB, v)
            return v

        else:
            v = float("inf")
            for eachAction in Actions:
                eachState = state.generateSuccessor(numMin - 1, eachAction)
                v = max(v, self.MinSearch(eachState, depth, numMin, localA, localB))

                if v < localA:
                    return v
                localB = min(localB, v)
            return v






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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

