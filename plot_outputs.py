import numpy as np
import torch
# import matplotlib.pyplot as plt
# from matplotlib import colors
import model_params as params


def plot_agnostic_results(exs, vocabulary, thresh, ind=-1, return_arrays=False):

    if ind < 0:
        ind = np.random.choice(exs['input'].shape[0])
    target = exs['target'][ind].detach().cpu().numpy().astype('int')

    sig_output = torch.sigmoid(exs['output'][ind])
    output = (sig_output.detach().cpu().numpy() > thresh).astype('int')
    
    orig = exs['orig'][ind].detach().cpu().numpy().astype('int')
    input = exs['input'][ind].detach().cpu().numpy().astype('int')

    input_words = vocabulary.vec_to_words(input)
    orig_words = vocabulary.vec_to_words(orig)
    space = ' '
    mark = 'X'

    if return_arrays:
        res = []
        for i in range(len(input)):
            res.append([orig_words[i], input_words[i], target[i], output[i]])
        return res

    lines = ['ORIG | ERRORED INPUT | TARGET | OUTPUT \n']
    for i in range(len(input)):
        line = (
            f'{orig_words[i]:25} | {input_words[i]:25} '
            f' | {mark if target[i] else space}'
            f' | {mark if output[i] else space} | \n'
            )
        lines.append(line)

    return lines
