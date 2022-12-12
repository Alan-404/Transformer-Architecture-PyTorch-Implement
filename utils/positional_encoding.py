import torch
import numpy as np


class PostionalEncoding:
    def encode_position(self, length: int, embeded_dim: int):
        angles = np.arange(embeded_dim)
        angles[1::2] = angles[0::2]

        angles = 1/10000**(angles/embeded_dim)

        angles = np.expand_dims(angles, axis=0)

        length_arr = np.arange(length)
        length_arr = np.expand_dims(length_arr, axis=1)

        pos_angles = np.dot(length_arr, angles)

        pos_angles[0::2] = np.sin(pos_angles[0::2])
        pos_angles[1::2] = np.cos(pos_angles[1::2])

        pos_angles = np.expand_dims(pos_angles, axis=0)

        return torch.tensor(pos_angles).type(torch.float32)