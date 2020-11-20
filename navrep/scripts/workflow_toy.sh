set -e  # exit if any command fails
ipython -- make_vae_dataset.py --environment toy
ipython -- train_vae.py --environment toy
ipython -- make_rnn_dataset.py --environment toy
ipython -- train_rnn.py --environment toy
