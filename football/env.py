from absl import flags
from absl import logging

import gym
from seed_rl.common import common_flags
from seed_rl.football import observation

FLAGS = flags.GLAGS
flags.DEFINE_string('game', '11_vs_11_easy_stochastic', 'Game/scenario name.')
flags.DEFINE_enum('reward_experiment', 'scoring', ['scoring', 'scoring,checkpoints'], 'Reward to be used for training.')
flags.DEFINE_enum('smm_size', 'default', ['default', 'medium', 'large'], 'size of the super mini map.')


def create_environment(_):
    logging.info('Creating environment: %s', FLAGS.game)
    assert FLAGS.num_action_repeats == 1, 'Only action repeat of 1 is supported.'
    channel_dimensions = {
        'default': (96, 72),
        'medium': (120, 90),
        'large': (144, 108),
    }[FLAGS.smm_size]
    env = gym.make(
        'gfootball:GFootball-%s-SMM-v0' % FLAGS.game,
        stacked=True,
        rewards=FLAGS.reward_experiment,
        channel_dimensions=channel_dimensions)
    return observation.PackedBitsObservation(env)