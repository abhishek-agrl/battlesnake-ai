# Battlesnake Agent
As part of my Masters studies, I have developed a bot that competed in the [Battlesnake Competition](https://play.battlesnake.com). This bot is my teams third attempt in making a competitive agent that beats the base agent developed by our TAs.

## Bot Description
As a broad overview, the agent evaluates the board each turn, and scores each grid entry. Based on the lowest score entry the bot then A-star searches a path to that gridpoint. The evaluation is the key here, and we evaluate the board on various metrics.

## File Description

1. `run_game.py` - Starts a game locally and the adversary can be either a Random Agent or A-Star Search Agent (Specify in the file before running)
2. `run_optuna.py` - Starts an optuna study to get the optimal values of scores for each metric in the evaluation function. Creates a local SQLite DB to track the trials.
3. `run_server.py` - Starts a server which hosts the agent and can then be used to enter the Battlesnake Competition.

## Credits - 
1. [Abhishek Agrawal](https://github.com/abhishek-agrl)
2. [Til Goldan](https://github.com/tilgoldan)
3. [Robin Gieseke](https://github.com/TheDarkchip)
