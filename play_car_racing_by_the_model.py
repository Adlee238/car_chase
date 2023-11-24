import argparse
import gym
import gym_multi_car_racing
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.h5` file of the trained model.')
    parser.add_argument('-o', '--opponent', help='The `.h5` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes should the model plays.')
    args = parser.parse_args()
    train_model = args.model
    opponent_model = args.opponent
    play_episodes = args.episodes

    env = gym.make("MultiCarRacing-v0", num_agents=2, use_random_direction=False, 
        backwards_flag=False)
    # env = gym.make('CarRacing-v0')
    agents = []
    for i in range(env.num_agents):
        agents.append(CarRacingDQNAgent(epsilon=0)) # Set epsilon to 0 to ensure all actions are instructed by the agent

    agents[0].load(train_model)
    if env.num_agents > 1: agents[1].load(opponent_model)

    for e in range(play_episodes):
        init_state = env.reset()
        init_state = process_state_image(init_state)

        total_rewards = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agents[0].frame_stack_num, maxlen=agents[0].frame_stack_num)
        time_frame_counter = 1
        
        while True:
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            actions = []
            for i in range(env.num_agents):
                actions.append(agents[i].act(current_state_frame_stack[i]))
            next_state, reward, done, info = env.step(actions)

            total_rewards += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e+1, play_episodes, time_frame_counter, float(total_rewards[0])))
                break
            time_frame_counter += 1
