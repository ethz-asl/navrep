set -e  # exit if any command fails
ipython -- make_vae_dataset.py --environment markone
ipython -- train_vae.py --environment markone
ipython -- make_rnn_dataset.py --environment markone
ipython -- train_rnn.py --environment markone
ipython -- train_gym_markoneencodedenv.py
ipython -- test_markone_VMC.py
