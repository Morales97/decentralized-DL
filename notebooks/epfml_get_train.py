import numpy as np
import epfml.store
import sys

dict = epfml.store.get(sys.argv[1])
np.save('cos_sim.npy', dict['cos_sim'])
np.save('pred_disag.npy', dict['pred_disag'])
np.save('pred_dist.npy', dict['pred_dist'])
np.save('models_pca.npy', dict['models_pca'])


