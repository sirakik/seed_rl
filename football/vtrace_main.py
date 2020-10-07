from absl import app
from absl import flags

from seed_rl.agents.vtrace import learner
from seed_rl.common import actor
from seed_rl.common import common_flags
from seed_rl.football import env
from seed_rl.football import networks
import tensorflow as tf


FLAGS = flags.FLAGS
# optimizer
flags.DEFINE_float('leraning_rate', 0.00048, 'Learning rate.')


def create_agent(unused_action_space, unused_env_observation_space, parametric_action_distribution):
    return networks.GFootball(parametric_action_distribution)


def create_optimizer(unused_final_iteration):
    learning_rate_fn = lambda iteration: FLAGS.learning_rate
    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
    return optimizer, learning_rate_fn


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    if FLAGS.run_mode == 'actor':
        actor.actor_loop(env.create_environment)
    elif FLAGS.run_mode == 'learner':
        learner.learner_loop(env.create_environment,
                             create_agent,
                             create_optimizer)
    else:
        raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__=='__main__':
    app.run(main)