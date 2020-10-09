# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for V-trace.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.
"""

from absl.testing import parameterized
import numpy as np
from seed_rl.common import vtrace
from seed_rl.common.parametric_distribution import CategoricalDistribution
import tensorflow as tf


def _shaped_arange(*shape):
  """Runs np.arange, converts to float and reshapes."""
  return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)


def _softmax(logits):
  """Applies softmax non-linearity on inputs."""
  return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def _ground_truth_calculation(discounts, behaviour_action_log_probs,
                              target_action_log_probs, rewards, values,
                              bootstrap_value, clip_rho_threshold,
                              clip_pg_rho_threshold):
  """Calculates the ground truth for V-trace in Python/Numpy."""
  log_rhos = target_action_log_probs - behaviour_action_log_probs
  vs = []
  seq_len = len(discounts)
  rhos = np.exp(log_rhos)
  cs = np.minimum(rhos, 1.0)
  clipped_rhos = rhos
  if clip_rho_threshold:
    clipped_rhos = np.minimum(rhos, clip_rho_threshold)
  clipped_pg_rhos = rhos
  if clip_pg_rho_threshold:
    clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

  # This is a very inefficient way to calculate the V-trace ground truth.
  # We calculate it this way because it is close to the mathematical notation of
  # V-trace.
  # v_s = V(x_s)
  #       + \sum^{T-1}_{t=s} \gamma^{t-s}
  #         * \prod_{i=s}^{t-1} c_i
  #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
  # Note that when we take the product over c_i, we write `s:t` as the notation
  # of the paper is inclusive of the `t-1`, but Python is exclusive.
  # Also note that np.prod([]) == 1.
  values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]], axis=0)
  for s in range(seq_len):
    v_s = np.copy(values[s])  # Very important copy.
    for t in range(s, seq_len):
      v_s += (
          np.prod(discounts[s:t], axis=0) * np.prod(cs[s:t],
                                                    axis=0) * clipped_rhos[t] *
          (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t]))
    vs.append(v_s)
  vs = np.stack(vs, axis=0)
  pg_advantages = (
      clipped_pg_rhos * (rewards + discounts * np.concatenate(
          [vs[1:], bootstrap_value[None, :]], axis=0) - values))

  return vtrace.VTraceReturns(vs=vs, pg_advantages=pg_advantages)


class LogProbsFromLogitsAndActionsTest(tf.test.TestCase,
                                       parameterized.TestCase):

  def test_log_probs_from_logits_and_actions(self):
    """Tests log_probs_from_logits_and_actions."""
    batch_size = 2
    seq_len = 7
    num_actions = 3

    policy_logits = _shaped_arange(seq_len, batch_size, num_actions) + 10
    actions = np.random.randint(
        0, num_actions - 1, size=(seq_len, batch_size), dtype=np.int32)

    categorical_distribution = CategoricalDistribution(num_actions, 'int32')
    action_log_probs_tensor = categorical_distribution.log_prob(
        policy_logits, actions)

    # Ground Truth
    # Using broadcasting to create a mask that indexes action logits
    action_index_mask = actions[..., None] == np.arange(num_actions)

    def index_with_mask(array, mask):
      return array[mask].reshape(*array.shape[:-1])

    # Note: Normally log(softmax) is not a good idea because it's not
    # numerically stable. However, in this test we have well-behaved values.
    ground_truth_v = index_with_mask(
        np.log(_softmax(policy_logits)), action_index_mask)

    self.assertAllClose(ground_truth_v, action_log_probs_tensor)


class VtraceTest(tf.test.TestCase, parameterized.TestCase):

  def test_vtrace(self):
    """Tests V-trace against ground truth data calculated in python."""
    batch_size = 5
    seq_len = 5

    # Create log_rhos such that rho will span from near-zero to above the
    # clipping thresholds. In particular, calculate log_rhos in [-2.5, 2.5),
    # so that rho is in approx [0.08, 12.2).
    log_rhos = _shaped_arange(seq_len, batch_size) / (batch_size * seq_len)
    log_rhos = 5 * (log_rhos - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
    values = {
        'behaviour_action_log_probs': tf.zeros_like(log_rhos),
        'target_action_log_probs': log_rhos,
        # T, B where B_i: [0.9 / (i+1)] * T
        'discounts': np.array([[0.9 / (b + 1) for b in range(batch_size)]  
                               for _ in range(seq_len)]),
        'rewards': _shaped_arange(seq_len, batch_size),
        'values': _shaped_arange(seq_len, batch_size) / batch_size,
        'bootstrap_value': _shaped_arange(batch_size) + 1.0,
        'clip_rho_threshold': 3.7,
        'clip_pg_rho_threshold': 2.2,
    }

    output = vtrace.from_importance_weights(**values)
    ground_truth_v = _ground_truth_calculation(**values)
    self.assertAllClose(output, ground_truth_v)


if __name__ == '__main__':
  tf.test.main()
