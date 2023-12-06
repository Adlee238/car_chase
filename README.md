# MADQL: Multi-Agent Deep Q-Learning for Autonomous Driving in Adversarial Environments
This repository builds on [`MultiCarRacing-v0`](https://github.com/igilitschenski/multi_car_racing) (Wilko Schwarting et al.), a multiplayer variant of Gym's original [`CarRacing-v0` environment](https://gym.openai.com/envs/CarRacing-v0/) and [OpenAI GYM CarRacing DQN](https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN) (Andy Wu), a single player Deep Q-Network model.

Modern autonomous driving systems operate through a system of mitigating risk and avoiding collisions, but that begs the question: what if a collision is unavoidable? This project aims to bring autonomous driving systems to their very limits, by training with adversarial opponents. We present an autonomous system with advanced capability in collision avoidance, able to mitigate passenger injury even in near-imminent conditions. This will have immense utility in general autonomous driving cars, humanitarian aid vehicles, racecars, and robotic delivery systems.

This environment involves a multi-player car chase game with an avoidant model and a collision-motivated opponent.

For more information about the project and results, reference the project video [MADQL: Multi-Agent Deep Q-Learning](https://youtu.be/RhXk6ENILLc?si=h5FFnzEWaK2QJ2wl).

## Installation

```bash
git clone https://github.com/Adlee238/car_chase.git
cd car_chase
```

## Dependencies

```bash
pip install swig
pip intall box2d
pip install box2d-py
pip install pyglet==1.5.27
pip install gym==0.21.0 
```

## Basic Usage
After installation, the environment can be tried out by running:

```bash
python -m gym_multi_car_racing.multi_car_racing
```


To train the model/opponent, run the following command:
```bash
python train_model.py [-m save/model_trial_XXX.h5] [-o save/opponent_trial_XXX.h5] [-s 1] [-e 1000] [-p 1.0]
```

- `-m` The path to the trained model to continue training from
- `-o` The path to the trained opponent to continue training from
- `-s` The starting training episode, default 1
- `-e` The ending training episode, default 1000
- `-p` The starting epsilon for both agents, default 1.0


To run a saved trial, execute the following command:
```bash
python play_car_racing_by_the_model.py -m save/model_trial_XXX.h5 -o save/opponent_trial_XXX.h5 [-e 1]
```

- `-m` The path to the trained model
- `-o` The path to the trained opponent
- `-e` The number of episodes to run

## File Structure
- `train_model.py` The training program
- `common_functions.py` Utility functions
- `CarRacingDQNAgent.py` The MADQL model class
- `play_car_racing_by_the_model.py` The program for playing CarRacing by the model and opponent
- `save/` The default folder to save the trained model
- `gym_multi_car_racing/` The `MultiCarRacing-v0` environment

This project was created for CS221 by Harviel Arcilla, Colette Do, and Andrew Lee.
