This was the honours project for my degree. This suite of Python files uses reinforcement learning to train AI agents to play three games, those games were also developed as part of the project. Several AI algorithms are covered, including Q-learning and Proximal Policy Optimisation.
The AI players are trained, then their performance is compared. Significant research was required to understand the relevant algorithms, and to design appropriate games and reward schemes.	
Pictured: Maze, Tag, Tic-Tac-Toe

<div>
	<img style="width:33%" src="https://i.imgur.com/tADwHlc.gif" alt="Video game gameplay: A green person moves around a grid in pursuit of coins.">
	<img style="width:33%" src="https://i.imgur.com/iNQxKeV.gif" alt="">
	<img style="width:33%" src="https://i.imgur.com/yhOINDA.gif" alt="Video game gameplay: Tic-Tac-Toe, noughts win">
</div>

Instructions for running the project.
	- Setting up your environment.
		- The code was tested on Python 3.11.3.
		- Remember to install the requirements (pip install -r requirements.txt).
	- agents.py
		- Contains definitions for the reinforcement learning agents, and three non-machine-learning agents used for benchmarking.
	- environments.py.
		- Contains code related to the environments.
	- unit_tests.py.
		- Contains the unit tests.
	- Running train_maze.py, train_tag.py, train_ttt.py will train new agents for the corresponding environment.
		- This typically takes a long time (1000 episodes = 30-40 minutes on my machine).
		- The final weights of the agents are written to disk in /checkpoints
		- Training metrics are written to disk in /metrics. Re-plot a metrics file using plot.py (change the file path on line 5).
		- Ensure that the constants (N_EPISODES, RNG_SEED) are set correctly.
	- demo_maze.py, demo_tag.py, demo_ttt.py can be used to observe the AI agents playing the game directly.
		- Maze & Tag are autonomous, but in TTT you play against the agent. Click a cell to place your symbol.
		- By default these are set up to demonstrate an agent from the report.
			- Maze: ActorCritic (seed 3000).
			- Tag:  PPO (seed 0).
			- TTT:  DQN (seed 2000).
		- Demonstrating a different agent.
			- Instantiate the correct agent type, layer structure, actionSpace & validAction.
			- Ensure that the correct weight file is loaded (change the file path).
			- Most hyperparameters don't matter as they're only used when learning, epsilon & epsilon decay do however.
	- human_interactive_maze, human_interactive_tag.py, human_interactive_ttt, can be used to play the games directly.
		- Maze: Move using the arrow keys. Collect as many coins as possible.
		- Tag: Turn using left & right arrow keys. Try to avoid the red blob as long as possible.
		- Tic-Tac-Toe: Click a cell to place your symbol.
	- baseline_maze.py and baseline_ttt.py
		- These run a baseline agent and print the average score.
		- These were used to determine the baseline values used in the report, and should replicate those values.