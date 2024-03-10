- Running train.py will train agents.
    - The final weights of the agents are written to disk in /checkpoints
    - Training metrics are written to disk in /metrics.
    
    - The training duration can be adjusted by changing the value of TRAINING_TIME_SECONDS.
    - For the paper, an RNG_SEED_INIT value of TODO was used. Using a different value will probably fail to replicate the results. 
- The files in /human_interactive can be used to play the games directly.
- The agents and algorithms are defined in agents.py, environment code is in environments.py.
- Remember to install the requirements (from requirements.txt).