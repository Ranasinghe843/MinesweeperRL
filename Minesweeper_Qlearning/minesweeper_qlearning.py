import sys
import os
import time
import minesweeper
from random import randint

def cls():
	
	time.sleep(1)
	print ("\n" * 50)

def cls2():
	time.sleep(1)
	os.system("clear")

def main(args):
	def print_map_by_state(qValueMap):

		otherMap = {}
		
		for state, action in qMap:
			if state in otherMap:
				otherMap[state].append(action)
			else:
				otherMap[state] = [action]

		for state in otherMap:
			print state
			for action in otherMap[state]:
				pair = (state, action)
				print "        ", action, ": ", str(qMap[pair])

	numLearningIterations = 10
	numPlayingIterations = 100
	numRows = 4
	numCols = 4
	difficulty = 1
	shouldPrintMap = False
	display_game=True

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

	qMap,totalgamestrained = minesweeper.generate_state_map_by_random_playing(numLearningIterations, numRows, numCols, difficulty)

	if shouldPrintMap:
		print_map_by_state(qMap)

	# Simulate the game play and determine error rate
	avgPercentTilesCleared = 0.0
	numWins = 0
	numGames = 0
	# print_map_by_state(qMap)
	for i in xrange(numPlayingIterations):
		game = minesweeper.MineSweeper(numRows, numCols, difficulty)

		if display_game:
			cls2()
			for l in  game.board:
				for val in l:
					if(val.isUncovered):
						if val.value==0:
							print ' ',
						else:
							print val.value,
					else:
						# print '10',					
						# print u'\u25A0',
						print u'\u25FC',



				print
		# print "-----------------"

		firstMove = (0, 0)

		while game.is_bomb(game.get_square(firstMove)):
			game = minesweeper.MineSweeper(numRows, numCols, difficulty)

		currentState = game.get_next_state(game.get_square(firstMove))
		# Play the game until winning or losing
		while not game.gameEnd:
			if display_game:

				cls2()

				for l in  game.board:
		
					for val in l:
						if(val.isUncovered):
							if val.value==0:
								print u'\u25FB',
							# print ' ',
							else:
								print val.value,
						else:
						# print '10',
						# print u'\u25A0',
							print u'\u25FC',


					print
			# print "-----------------"
			nextMove = minesweeper.getNextMove(qMap, game)
			game.get_next_state(nextMove)
		if display_game:
			cls2()
			for l in  game.board:
		
				for val in l:
					if(val.isUncovered):
						if val.value==0:
							print u'\u25FB',
						else:
							print val.value,
					else:
						# print u'U25A0',
						# print u'\u25A0',
						print u'\u25FC',


				print
		# print "-----------------"

		if game.gameWon:
			if display_game:
				cls2()
				for l in  game.board:
		
					for val in l:
						if(val.isUncovered):
							if(val.value==9):
								print u'\u2600',
							# print u'\u260X',
							elif val.value==0:
								print u'\u25FB',
							
							else:
								print val.value,

						else:
							print u'\u2600',
						
					print
				print "\nGAME WON"
				time.sleep(0.50)
			numWins += 1
		else:
			if display_game:
				cls2()
				for l in  game.board:
		
					for val in l:
						if(val.value==9):
							print u'\u2620',
							continue
						if(val.isUncovered):
							if(val.value==0):
								print u'\u25FB',
							else:	
								print val.value,
						else:
							print u'\u25A0',
						
					print			
				print "\nGAME LOST"
				time.sleep(0.50)


		numTilesCleared = sum([1 for tile in game.get_state() if tile != game.covered_value])
		avgPercentTilesCleared = avgPercentTilesCleared * (float(numGames)/(numGames + 1)) + \
									 (float(numTilesCleared)/(numRows * numCols))/(numGames + 1)

		numGames += 1
	cls2()
	avgPercentTilesCleared *= 100
	successRate = (float(numWins)/numGames)*100
	print "After training and testing for %d seconds" % numLearningIterations
	print "Number of training games played: ", totalgamestrained
	print "Number of testing games played:", numPlayingIterations
	print "Percentage of games won: ", successRate
	print "Cleared an average of %f%% of the board." % (avgPercentTilesCleared)

	# This prints out the qMap in a more readable manner



if __name__ == "__main__":
    main(sys.argv[1:])