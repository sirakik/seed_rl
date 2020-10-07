import collections
import time
import tensorflow as tf


class Aggregator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.count = 0

    def average(self):
        return self.sum / self.count if self.count else 0.

    def add(self, v):
        self.sum += v
        self.count += 1


class ExportingTimer(object):
  aggregators = collections.defaultdict(Aggregator)

  def __init__(self, summary_name, aggregation_window_size):
    self.summary_name = summary_name
    self.aggregation_window_size = aggregation_window_size

  def __enter__(self):
    self.start_time_s = time.time()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.elapsed_s = time.time() - self.start_time_s
    aggregator = self.aggregators[self.summary_name]
    aggregator.add(self.elapsed_s)
    if aggregator.count >= self.aggregation_window_size:
      tf.summary.scalar(self.summary_name, aggregator.average())
      aggregator.reset()