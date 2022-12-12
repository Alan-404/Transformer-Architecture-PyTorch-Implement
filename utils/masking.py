import torch
import numpy as np

class MaskingGenerator:
    def generate_padding_mask(self, inp):
        result = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                if inp[i][j] == 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        result = result[:, np.newaxis, np.newaxis, :]
        return torch.tensor(result).type(torch.float32)

    def generate_look_ahead_mask(self, inp_len):
        mask = np.ones((inp_len, inp_len))
        row = 0
        col = 1
        for _ in range(inp_len):
            mask[row][:col] = 0
            row += 1
            col += 1
        return torch.tensor(mask).type(torch.float32)

    def combine_max_mask(self, look_ahead_mask, padding_mask):
        result = np.zeros((padding_mask.shape[0], 1, look_ahead_mask.shape[0], look_ahead_mask.shape[1]))
        for i in range(padding_mask.shape[0]):
            for j in range(padding_mask.shape[0]):
                result[i][0][j] = torch.maximum(look_ahead_mask[j], padding_mask[i][0][0])
        return torch.tensor(result).type(torch.float32)

    def generate_mask(self, inp, targ):
        encoder_padding_mask = self.generate_padding_mask(inp)

        decoder_padding_mask = self.generate_padding_mask(inp)

        decoder_look_ahead_mask = self.generate_look_ahead_mask(targ.size(1))

        decoder_inp_padding_mask = self.generate_padding_mask(targ)

        decoder_look_ahead_mask = self.combine_max_mask(decoder_look_ahead_mask, decoder_inp_padding_mask)

        return encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask



    