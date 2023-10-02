import os

import numpy as np
import torch
import yaml
from .src.agent import Agent
from agent_code.GlasHoch_Rangers.src.State import State

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT',"BOMB"]

def setup(self):
    np.random.seed(42)

    self.logger.debug('Successfully entered setup code')

    with open(os.environ.get("AGENT_CONF", "./configs/default.yaml"), "r") as ymlfile:
        configs = yaml.safe_load(ymlfile)
    self.AGENT_CONFIG = configs["AGENT_CONFIG"]
    self.REWARD_CONFIG = configs["REWARD_CONFIG"]
    self.agent = Agent(self.AGENT_CONFIG, self.REWARD_CONFIG, training=self.train)
    print(self.agent)
    self.state_processor = State(window_size=int((self.AGENT_CONFIG["features_dim"][1] - 1) / 2))
    self.draw_plot = self.AGENT_CONFIG["draw_plot"]
    if self.draw_plot:
        self.mode_plot = self.AGENT_CONFIG["mode_plot"]

def act(self, game_state: dict) -> str:
    features = self.state_processor.get_features(game_state)
    self.last_features = features
    action_idx = self.agent.act(features, game_state)
    return actions[action_idx]


