navreptrain_v_dataset : part0 part1 part2 part3 part4 part5 part6 part7 \
			part8 part9 part10 part11 part12 part13 part14 part15
part0 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 0 16
part1 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 1 16
part2 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 2 16
part3 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 3 16
part4 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 4 16
part5 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 5 16
part6 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 6 16
part7 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 7 16
part8 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 8 16
part9 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 9 16
part10 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 10 16
part11 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 11 16
part12 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 12 16
part13 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 13 16
part14 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 14 16
part15 :
			ipython -- navrep/scripts/make_vae_dataset.py --env navreptrain --subprocess 15 16
