from absl import flags
from seed_rl.football import env
import tensorflow as tf

FLAGS = flags.FLAGS


class EnvironmentTest(tf.test.TestCase):
    def test_run_step(self):
        environment = env.create_einvirionment(0)
        environment.reset()
        environment.step(0)
        environment.close()


if __name__ == '__main__':
  tf.test.main()
