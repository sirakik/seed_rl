import os
import timeit

from absl import flags
from absl import logging
import numpy as np
from seed_rl import grpc
from seed_rl.common import common_flags
from seed_rl.common import env_wrappers
from seed_rl.common import profiling
from seed_rl.common import utils
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_integer('task', 0, 'Task id.')
flags.DEFINE_integer('num_actors_with_summaries', 4, 'Number of actors that will log debug/profiling TF summaries.')
flags.DEFINE_bool('render', False)


def are_summaries_enable():
    return FLAGS.task < FLAGS.num_actors_with_summaries


def actor_loop(create_env_fn):
    env_batch_size = FLAGS.env_batch_size
    logging.info('# Starting actor loop. Task: %r. Environment batch size: %r', FLAGS.task, env_batch_size)
    is_rendering_enable = FLAGS.render and FLAGS.task == 0
    if are_summaries_enable():
        summary_writer = tf.summary.create_file_writer(
            os.path.join(FLAGS.logdir, 'actor_{}'.format(FLAGS.task)),
            flush_millis=20000, max_queue=1000)
        timer_cls = profiling.ExportingTimer
    else:
        summary_writer = tf.summary.create_noop_writer()
        timer_cls = utils.nullcontext

    actor_step = 0
    with summary_writer.as_default():
        while True:
            try:
                client = grpc.Client(FLAGS.server_address)
                batched_env = env_wrappers.BatchedEnvironment(create_env_fn, env_batch_size, FLAGS.task * env_batch_size)

                env_id = batched_env.env_ids
                run_id = np.random.randint(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    size=env_batch_size,
                    dtype=np.int64)
                observation = batched_env.reset()
                reward = np.zeros(env_batch_size, np.float32)
                raw_reward = np.zeros(env_batch_size, np.float32)
                done = np.zeros(env_batch_size, np.bool)
                abandoned = np.zeros(env_batch_size, np.bool)

                global_step = 0
                episode_step = np.zeros(env_batch_size, np.int32)
                episode_return = np.zeros(env_batch_size, np.float32)
                episode_raw_return = np.zeros(env_batch_size, np.float32)
                episode_step_sum = 0
                episode_return_sum = 0
                episode_raw_return_sum = 0
                episodes_in_report = 0

                elapsed_inference_s_timer = timer_cls('actor/elapsed_inference_s', 1000)
                last_log_time = timeit.default_timer()
                last_global_step = 0
                while True:
                    tf.summary.experimental.set_step(actor_step)
                    env_output = utils.EnvOutput(reward, done, observation, abandoned, episode_step)
                    with elapsed_inference_s_timer:
                        action = client.inference(env_id, run_id, env_output, raw_reward)
                    with timer_cls('actor/elapsed_env_step_s', 1000):
                        observation, reward, done, info = batched_env.step(action.numpy())
                    if is_rendering_enabled:
                        batched_env.render()
                    for i in range(env_batch_size):
                        episode_step[i] += 1
                        episode_return[i] += reward[i]
                        raw_reward[i] = float((info[i] or {}).get('score_reward', reward[i]))
                        episode_raw_return[i] += raw_reward[i]
                        abandoned[i] = (info[i] or {}).get('abandoned', False)
                        assert done[i] if abandoned[i] else True
                        if done[i]:
                            if abandoned[i]:
                                assert env_batch_size == 1 and i == 0, (
                                    'Mixing of batched and non-batched inference calls is not '
                                    'yet supported')
                                env_output = utils.EnvOutput(
                                    reward,
                                    np.array([False]), observation,
                                    np.array([False]), episode_step)
                                with elapsed_inference_s_timer:
                                    client.inference(env_id, run_id, env_output, raw_reward)
                                reward[i] = 0.0
                                raw_reward[i] = 0.0

                            # log
                            current_time = timeit.default_timer()
                            episode_step_sum += episode_step[i]
                            episode_return_sum += episode_return[i]
                            episode_raw_return_sum += episode_raw_return[i]
                            global_step += episode_step[i]
                            episodes_in_report += 1
                            if current_time - last_log_time > 1:
                                logging.info(
                                    'Actor steps: %i, Return: %f Raw return: %f '
                                    'Episode steps: %f, Speed: %f steps/s', global_step,
                                    episode_return_sum / episodes_in_report,
                                    episode_raw_return_sum / episodes_in_report,
                                    episode_step_sum / episodes_in_report,
                                    (global_step - last_global_step) /
                                    (current_time - last_log_time))
                                last_global_step = global_step
                                episode_return_sum = 0
                                episode_raw_return_sum = 0
                                episode_step_sum = 0
                                episodes_in_report = 0
                                last_log_time = current_time

                            episode_step[i] = 0
                            episode_return[i] = 0
                            episode_raw_return[i] = 0

                    with timer_cls('actor/elapsed_env_reset_s', 10):
                        observation = batched_env.reset_if_done(done)

                    if is_rendering_enabled and done[0]:
                        batched_env.render()

                    actor_step += 1
            except (tf.errors.UnavailableError, tf.errors.CancelledError) as e:
                logging.exception(e)
                batched_env.close()
