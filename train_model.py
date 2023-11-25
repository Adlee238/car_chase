'''
File: train_model.py

The training program.
'''
import numpy as np
import argparse
import gym
import gym_multi_car_racing
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
from gym_multi_car_racing import MultiCarRacing

RENDER                        = True
STARTING_EPISODE              = 1
ENDING_EPISODE                = 1000
SKIP_FRAMES                   = 2
TRAINING_BATCH_SIZE           = 64
SAVE_TRAINING_FREQUENCY       = 25
UPDATE_TARGET_MODEL_FREQUENCY = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-o', '--opponent', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    args = parser.parse_args()

    env = gym.make("MultiCarRacing-v0", num_agents=2, use_random_direction=False, 
        backwards_flag=False)
    
    # env = gym.make("MultiCarRacing-v0", num_agents=2, direction='CCW',
    #     use_random_direction=True, backwards_flag=True, h_ratio=0.25,
    #     use_ego_color=False)
    # env = gym.make('CarRacing-v0')

    # list of agents: optimized for 2, first agent is going to be trained, second is already trained
    agents = []
    for i in range(env.num_agents):
        if i == 0: agents.append(CarRacingDQNAgent(epsilon=args.epsilon))
        else: agents.append(CarRacingDQNAgent(epsilon=0))

    if args.model:
        agents[0].load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end
    if args.opponent:
        agents[1].load(args.opponent)

    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        init_state = env.reset()
        init_state = process_state_image(init_state)
        total_rewards = np.zeros(env.num_agents)
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state]*agents[0].frame_stack_num, maxlen=agents[0].frame_stack_num)
        time_frame_counter = 1
        done = False
        
        while True:
            if RENDER:
                env.render()
            
            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            actions = []
            for i in range(env.num_agents):
                actions.append(agents[i].act(current_state_frame_stack[i]))

            rewards = np.zeros(env.num_agents)
            for _ in range(SKIP_FRAMES+1):
                next_state, r, done, info = env.step(actions)
                rewards += r
                if done:
                    break

            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and rewards[0] < 0 else 0

            # Extra bonus for the model if it uses full gas
            if actions[0][1] == 1 and actions[0][2] == 0:
                rewards[0] *= 0.5

            total_rewards += rewards

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agents[0].memorize(current_state_frame_stack[0], actions[0], rewards[0], next_state_frame_stack[0], done)

            if done or negative_reward_counter >= 25 or total_rewards[0] < 0:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Epsilon: {:.2}'.format(e, ENDING_EPISODE, time_frame_counter, float(total_rewards[0]), float(agents[0].epsilon)))
                break
            if len(agents[0].memory) > TRAINING_BATCH_SIZE:
                agents[0].replay(TRAINING_BATCH_SIZE)
            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agents[0].update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            print("SAVING")
            agents[0].save('./save/trial_{}.h5'.format(e))

    env.close()