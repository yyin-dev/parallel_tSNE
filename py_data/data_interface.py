# These 2 methods translate between our binary format and numpy array
# binary format specification:
# number of points (int), dimension size (int)
# array of size number of points x dimension size (float)

import struct
import numpy as np
from matplotlib import pyplot as plt

def pack_bin_file(data, prefix):
  with open(f"{prefix}_{data.shape[0]}x{data.shape[1]}.bin", "wb") as file:
    file.write(struct.pack('2i', *data.shape))
    file.write(data.astype(np.float32).tobytes())

def load_bin_file(path):
  with open(path, "rb") as file:
    data = file.read() # byte array
  data_shape = struct.unpack('2i', data[:8])
  return np.frombuffer(data[8:], dtype=np.float32).reshape(data_shape)

def visualize_tsne_result(tsne_embedded, labels=None, dot_size=8):
  fig = plt.figure(1, (12., 10.))
  sc = plt.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], alpha=0.8, s=dot_size,
              c=labels, cmap="plasma")
  legend1 = plt.legend(*sc.legend_elements(),
                      loc="upper right", title="Classes")
  fig.add_artist(legend1)