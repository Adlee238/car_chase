'''
File: common_functions.py

Some functions that will be used in multiple programs will be put in here.
'''
import cv2
import numpy as np

def process_state_image(state):
    newState = np.zeros(shape=((state.shape[0], state.shape[1], state.shape[2])))
    for s in range(len(state)):
        newState[s,:,:] = cv2.cvtColor(state[s], cv2.COLOR_RGB2GRAY)
    newState = newState.astype(float)
    newState /= 255.0
    return newState

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 3, 0))