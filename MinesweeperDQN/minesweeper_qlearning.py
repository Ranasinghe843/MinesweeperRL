import sys
import minesweeper
from random import randint
import matplotlib.pyplot as plt  # Import matplotlib for plotting

def main(args):
    # chaneg to 500
    numLearningIterations = 10000
    numPlayingIterations = 1000
    numRows = 9
    numCols = 9
    difficulty = 1
    shouldPrintMap = True

    cont = False
    for i in range(len(args)):
        if cont:
            cont = False
            continue
        elif args[i] == '-p':
            if i+1 >= len(args):
                raise Exception("Expected integer argument after -p")
            else:
                numPlayingIterations = int(args[i+1])
                cont = True
        elif args[i] == '-l':
            if i+1 >= len(args):
                raise Exception("Expected integer argument after -l")
            else:
                numLearningIterations = int(args[i+1])
                cont = True
        elif args[i] == '-r':
            if i+1 >= len(args):
                raise Exception("Expected integer argument after -r")
            else:
                numRows = int(args[i+1])
                cont = True
        elif args[i] == '-c':
            if i+1 >= len(args):
                raise Exception("Expected integer argument after -c")
            else:
                numCols = args[i+1]
                cont = True
        elif args[i] == 'q':
            shouldPrintMap = True
        else:
            raise Exception("Wrong input")

    if numRows < 1 or numCols < 1 or numPlayingIterations < 1 or numLearningIterations < 1:
        raise Exception("Wrong input")

    qMap = minesweeper.generate_state_map_using_label(numLearningIterations, numRows, numCols, difficulty)
    print("Training Finished")

    if shouldPrintMap:
        pass
        #print_map_by_state(qMap)

    # Simulate the game play and determine error rate
    avgPercentTilesCleared = 0.0
    numWins = 0
    numGames = 0

    # Prepare lists to store data for plotting
    win_rates = []
    avg_cleared = []

    for i in range(numPlayingIterations):
        game = minesweeper.MineSweeper(numRows, numCols, difficulty)
        firstMove = (0, 0)

        while game.is_bomb(game.get_square(firstMove)):
            game = minesweeper.MineSweeper(numRows, numCols, difficulty)

        currentState = game.get_next_state(game.get_square(firstMove))

        # Play the game until winning or losing
        while not game.gameEnd:
            nextMove = minesweeper.getNextMove(qMap, game)
            game.get_next_state(nextMove)

        if game.gameWon:
            numWins += 1

        numTilesCleared = sum([1 for tile in game.get_state() if tile != game.covered_value])
        avgPercentTilesCleared = avgPercentTilesCleared * (float(numGames)/(numGames + 1)) + \
                                     (float(numTilesCleared)/(numRows * numCols))/(numGames + 1)

        numGames += 1

        # Store the data for plotting
        win_rates.append((float(numWins)/numGames) * 100)
        avg_cleared.append(avgPercentTilesCleared)

    avgPercentTilesCleared *= 100
    successRate = (float(numWins)/numGames)*100
    print ("Number of training games: ", numLearningIterations)
    print ("Number of testing games ", numPlayingIterations)
    print ("Percentage of games won: ", successRate)
    print ("Cleared an average of %f%% of the board." % (avgPercentTilesCleared))

    # Plot the performance
    plt.figure(figsize=(10, 5))

    # Plot win rate over time
    plt.subplot(1, 2, 1)
    plt.plot(win_rates, label="Win Rate (%)")
    plt.xlabel('Iterations')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate vs Training Iterations')
    plt.legend()

    # Plot average tiles cleared over time
    plt.subplot(1, 2, 2)
    plt.plot(avg_cleared, label="Avg Tiles Cleared (%)", color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('Avg Tiles Cleared (%)')
    plt.title('Avg Tiles Cleared vs Training Iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # This prints out the qMap in a more readable manner
    def print_map_by_state(qValueMap):
        otherMap = {}
        
        for state, action in qMap:
            if state in otherMap:
                otherMap[state].append(action)
            else:
                otherMap[state] = [action]

        for state in otherMap:
            print (state)
            for action in otherMap[state]:
                pair = (state, action)
                print ("        ", action, ": ", str(qMap[pair]))


if __name__ == "__main__":
    main(sys.argv[1:])

# For each game
# Take average
# minimal number of moves to win the game / total number of moves taken by agent to win the game
# Bound between 0 and 1