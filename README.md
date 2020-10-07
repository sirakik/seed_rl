It's just a copy of [SEED RL](https://github.com/google-research/seed_rl).


---
``git clone https://github.com/sirakik/seed_rl.git``  

``mkdir seed_rl/kaggle_simulations && mkdir seed_rl/kaggle_simulations/agent``

``bash seed_rl/train.sh football vtrace 4 '--total_environment_frames=10000 --game=11_vs_11_kaggle --reward_experiment=scoring,checkpoints --logdir=/kaggle_simulations/agent/'
``
