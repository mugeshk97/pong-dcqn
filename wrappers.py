import gym

def make_env(env_name, shape = 84, grayscale = True, scale = True, stack = 4):
    env  = gym.make(env_name)
    env = gym.wrappers.AtariPreprocessing(env= env, frame_skip= 1, screen_size = shape, scale_obs= scale, grayscale_obs= grayscale)
    env = gym.wrappers.FrameStack(env, num_stack= stack)
    
    return env