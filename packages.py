import numpy as np
import argparse
import pickle
import os
import time
import torch
import pandas as pd 
import scipy as sp
import torch.nn as nn
import torch.optim as optim
import random
from collections import defaultdict
import scipy.io as sio
import scipy.sparse as spp
import scipy as sp
from sklearn.preprocessing import normalize
import json
import networkx as nx
import time
from datetime import datetime
import sklearn.metrics.pairwise as Kernel
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding


# import subprocess
# import sys
# from io import StringIO
# import pandas as pd
#
# def get_free_gpu():
#     gpu_stats = str(subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]))
#     gpu_df = pd.read_csv(StringIO(u"".join(gpu_stats)),
#                          names=['memory.used [MiB]', 'memory.free [MiB]'],
#                          skiprows=1, lineterminator='\n\r')
#     print('GPU usage:\n{}'.format(gpu_df))
#     gpu_df['memory.free [MiB]'] = gpu_df['memory.free [MiB]'].map(lambda x: x.rstrip(' [MiB]'))
#     print("aa: ", gpu_df['memory.free [MiB]'])
#     idx = gpu_df['memory.free [MiB]'].idxmax()
#     print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free [MiB]']))
#     return idx
#
#
# free_gpu_id = get_free_gpu()
# torch.cuda.set_device(free_gpu_id)


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

