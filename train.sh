# Training launcher script.

# Make SEED RL visible to Python.
export PYTHONPATH=$PYTHONPATH:$(pwd)
ENVIRONMENT=$1
AGENT=$2
NUM_ACTORS=$3
shift 3

# Start actor tasks which run environment loop.
actor=0
while [ "$actor" -lt ${NUM_ACTORS} ]; do
  python3 seed_rl/${ENVIRONMENT}/${AGENT}_main.py --run_mode=actor --logtostderr $@ --num_actors=${NUM_ACTORS} --task=${actor} 2>/dev/null >/dev/null &
  actor=$(( actor + 1 ))
done
# Start learner task which performs training of the agent.
python3 seed_rl/${ENVIRONMENT}/${AGENT}_main.py --run_mode=learner --logtostderr $@ --num_actors="${NUM_ACTORS}"
