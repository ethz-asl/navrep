set -e  # exit if any command fails
ipython -- make_vae_dataset.py --environment marktwo
ipython -- train_vae.py --environment marktwo
ipython -- make_rnn_dataset.py --environment marktwo
ipython -- train_rnn.py --environment marktwo
ipython -- train_gym_marktwoencodedenv.py
ipython -- test_marktwo_VMC.py
