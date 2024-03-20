- Maze Tests.
	1. Controls Check.
		- Environment Parameters: render_mode="human", nCoins=0.
		- Procedure: Press every control button in order.
		- Expected:
			- Control mapping is correct.
				- Up Arrow: Agent moves one square up.
				- Down Arrow: Agent moves one square down.
				- Left Arrow: Agent moves one square left.
				- Right Arrow: Agent moves one square left.
			- No visual errors.
	2. Collect Coins.
		- Environment Parameters: render_mode="human", rewardExploration=False.
		- Procedure: Collect as many coins as possible.
		- Expected:
			- All collected coins are immediately replaced.
			- Final score = 50 * number of coins collected.
			- No visual errors.
	3. Exploration.
		- Environment Parameters: render_mode="human", nCoins=0.
		- Procedure: Visit every square, then move between the same two tiles repeatedly.
		- Expected:
			- Final score = number of empty squares on the grid - 1.
			- No visual errors.
	4. Collide With Edge.
		- Environment Parameters: render_mode="human", nCoins=0.
		- Procedure: Walk off the edges, all four of them.
		- Expected:
			- Attempting to move off the edge results in no movement.
			- No visual errors.
	5. Collide With Walls.
		- Environment Parameters: render_mode="human", nCoins=0.
		- Procedure: Walk into every accessible solid square during the epoch.
		- Expected:
			- Attempting to move into solid squares results in no movement.
			- No visual errors.
- Tag Tests.
	6. Controls Check.
		- Environment Parameters: render_mode="human", nSeekers=0.
		- Procedure: Press every control button in order.
		- Expected:
			- Control mapping is correct.
				- No Input: Agent continues straight.
				- Hold Left Arrow: Agent rotates counterclockwise while moving.
				- Hold Right Arrow: Agent rotates clockwise while moving.
			- No visual errors.
	7. Collide With Seeker.
		- Environment Parameters: render_mode="human".
		- Procedure: Attempt to touch the seeker during the episode.
		- Expected:
			- Player collides with the seeker, ending the epoch.
			- No visual errors.
	8. Collide With Arena.
		- Environment Parameters: render_mode="human", nSeekers=0.
		- Procedure: Attempt to move outside the arena during the episode.
		- Expected:
			- Epoch ends once the runner is no longer in contact with the white arena.
			- No visual errors.
- Tic Tac Toe Tests.
	9. Controls Check.
		- Environment Parameters: render_mode="human".
		- Procedure: Input a move in each cell, in order, resetting the game after each input.
		- Expected:
			- Control mapping is correct. Clicking on a cell places the correct symbol in that cell.
			- No visual errors.
	10. Win.
		- Environment Parameters: render_mode="human", TTTSearchAgent(None,epsilon=1).
		- Procedure: Play a full match and win.
		- Expected:
			- Match ends with correct win message.
			- No visual errors.
	11. Draw.
		- Environment Parameters: render_mode="human".
		- Procedure: Play a full match and draw.
		- Expected:
			- Match ends with correct draw message.
			- No visual errors.
	12. Loss.
		- Environment Parameters: render_mode="human", TTTSearchAgent(None,epsilon=0).
		- Procedure: Play a full match and lose.
		- Expected:
			- Match ends with correct loss message.
			- No visual errors.