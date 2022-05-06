# These 2 methods translate between our binary format and numpy array
# binary format specification:
# number of points (int), dimension size (int)
# array of size number of points x dimension size (float)

import struct
import numpy as np
from matplotlib import pyplot as plt

def pack_bin_file(data, prefix):
  filename = f"{prefix}_{data.shape[0]}x{data.shape[1]}.bin"
  with open(filename, "wb") as file:
    file.write(struct.pack('2i', *data.shape))
    file.write(data.astype(np.float32).tobytes())
  return filename

def load_bin_file(path):
  with open(path, "rb") as file:
    data = file.read() # byte array
  data_shape = struct.unpack('2i', data[:8])
  return np.frombuffer(data[8:], dtype=np.float32).reshape(data_shape)

def visualize_tsne_result(tsne_embedded, labels=None, dot_size=8, filename=None, title=None):
  fig = plt.figure(1, (12., 10.), dpi=150)
  sc = plt.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], alpha=0.8, s=dot_size,
              c=labels, cmap="plasma")
  legend1 = plt.legend(*sc.legend_elements(),
                      bbox_to_anchor=(1.04,1), loc="upper left", title="Classes")
  fig.add_artist(legend1)
  if title is not None:
    plt.title(title)
  if filename is not None:
    plt.savefig(filename, bbox_inches="tight")
    plt.close()