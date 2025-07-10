import math
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from simulation.DS_unroll import TrajectoryStateIndices


class KinematicModel(ABC):
    """Base class interface for kinematic model."""
    #: The prefix that will identify the model class
    model_prefix: str

    @abstractmethod
    def reset(self, init_state: np.ndarray) -> None:
        """Reset the model state when new episode starts.

        :param init_state: the initial state of the ego
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, input_action: TrajectoryStateIndices) -> Dict[str, np.ndarray]:
        """Update the model state based on the action at a particular time-step during the episode.

        :param input_action: the prediction for next time-step
        :return: reward at a particular frame index (time-step) during the episode.
        """
        raise NotImplementedError


class UnicycleModel(KinematicModel):
    """This class is responsible for controlling kinematics using the Unicycle model.

    :param model_prefix: the prefix that will identify this model class
    :param min_acc: the threshold for minimum acceleration
    :param max_acc: the threshold for maximum acceleration
    :param min_steer: the threshold for minimum steering
    :param max_steer: the threshold for maximum steering
    """

    def __init__(self, model_prefix: str = "Unicycle",
                 min_acc: float = -0.6,  # min acceleration: -6 mps2
                 max_acc: float = 0.6,   # max acceleration: 6 mps2
                 min_steer: float = -math.radians(45) * 0.1,  # max yaw rate: 45 degrees per second
                 max_steer: float = math.radians(45) * 0.1,   # max yaw rate: 45 degrees per second
                 ) -> None:
        """Constructor method
        """
        self.model_prefix = model_prefix
        # Thresholds
        self.min_acc = min_acc
        self.max_acc = max_acc
        self.min_steer = min_steer
        self.max_steer = max_steer

    def reset(self, init_state: np.ndarray) -> None:
        """Reset the model state when new episode starts.

        :param init_state: the initial state of the target agent
        """
        self.old_x = init_state[TrajectoryStateIndices.X]
        self.old_y = init_state[TrajectoryStateIndices.Y]
        self.old_r = init_state[TrajectoryStateIndices.THETA]
        self.old_v = init_state[TrajectoryStateIndices.SPEED]
        self.old_v_x = init_state[TrajectoryStateIndices.VELOCITY_X]
        self.old_v_y = init_state[TrajectoryStateIndices.VELOCITY_Y]

    def update(self, input_action: np.ndarray) -> Dict[str, np.ndarray]:
        """Update the model state based on the action at a particular time-step during the episode.

        :param input_action: the prediction [steer, acc] for next time-step
        :return: reward at a particular frame index (time-step) during the episode.
        """
        #both steer and acc are in /s unit (rad/s and m/s2)
        steer = input_action[0]
        acc = input_action[1]

        # Clip
        steer = np.clip(steer, self.min_steer, self.max_steer)
        acc = np.clip(acc, self.min_acc, self.max_acc)

        # Update x, y, r, v
        # x, y and r are already zero-centered in the raster image
        # there might be a small error induced with the predicted position and x and y velocities
        # this might happen because the position x and y are derived from new predicted speed.
        # the predicted speed and predicted velocity_x has a little difference
        self.new_r = steer
        self.new_v = self.old_v + acc
        if self.new_v < 0:  # if the agent was stopped, old_v =0; if acc is negative, it will result velocity backward
            self.new_v = 0  # keep the agent stopped, rather than sending it back
        acc_x = math.cos(self.new_r) * acc
        acc_y = math.sin(self.new_r) * acc
        self.new_v_x = self.old_v_x + acc_x  # or self.step_time
        self.new_v_y = self.old_v_y + acc_y
        self.new_x = math.cos(self.new_r) * self.new_v  # in unit meter and m/0.1s
        self.new_y = math.sin(self.new_r) * self.new_v

        # predicted features for the next frame. predicted velocities are in m/0.1s
        return_dict = {"positions": np.array([[[self.new_x, self.new_y]]]),
                       "velocities": np.array([[[self.new_v_x, self.new_v_y]]]), "yaws": np.array([[[self.new_r]]])}

        # Saved updated states
        self.old_v = self.new_v
        self.old_v_x = self.new_v_x
        self.old_v_y = self.new_v_y

        return return_dict
