import gym
from gym import spaces

class CustomEnv(gym.Env):
    # Custom Environment that follows gym interface
    metadata = {'render.modes': ['human']}

    def __init__(self, arg1, arg2, ...):
        super(CustomEnv, self).__init()

        # Define action and observation space
        # THey must be gy.spaces objects

        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        # Example for using image as inut:
        self.observation_space = spaces.Box (low=0, high=255, shape=(HEIGHT,WIDTH,N_CHANNELS), dtype=np.uint8)

    def step(self,action):
        # Execute one time step within the environment

    
    def reset(self):
        # Reset the state of the environment to an initial state

    def render(self, mode='human', close=False):
        # Render the environment to the screen

