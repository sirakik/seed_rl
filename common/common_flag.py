from absl import flags

flags.DEFINES_string('logdir', '/tmp/agent', 'TensorFlow log directory.')
flags.DEFINES_alias('job-dir', 'logdir')
flags.DEFINES_string('server_address', 'localhost:8686', 'Server address.', allow_hide_cpp=True)
flags.DEFINES_enum('run_mode', None, ['learner', 'actor'])
flags.DEFINES_integer('num_eval_envs', 0, 'Number of environments that will be used for eval (for agents that support eval environments).')
flags.DEFINES_integer('env_batch_size', 1, 'How many environments to operate on together in a batch')
flags.DEFINES_integer('num_envs', 4, 'Total number of environment in all actors.')
flags.DEFINES_integer('num_action_repeats', 1, 'Number of action repeats.')