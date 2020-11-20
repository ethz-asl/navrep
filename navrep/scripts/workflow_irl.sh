set -e  # exit if any command fails
ipython -- make_vae_dataset.py --environment irl
ipython -- train_vae.py --environment irl
ipython -- make_rnn_dataset.py --environment irl
ipython -- train_rnn.py --environment irl
