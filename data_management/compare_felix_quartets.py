import h5py
import numpy as np
import difflib

dset_path = r'./felix_comparison.h5'
# voice, start, duration, midi_pitch, notated_pitch, accidental
inds_subset = np.array([0, 2, 4, 5])


class HashableNote(object):
    def __init__(self, arr):
        self.arr = arr
    def __hash__(self):
        return hash(tuple(self.arr))

with h5py.File(dset_path, 'a') as f:

    correct_fnames = [x for x in f.keys() if 'aligned' in x and 'op80' not in x]
    error_fnames = [x for x in f.keys() if 'omr' in x]

    correct_dset = [f[x][:, inds_subset] for x in correct_fnames]
    error_dset = [f[x][:, inds_subset] for x in error_fnames]


ind = 5

correct_seq = [HashableNote(x) for x in correct_dset[ind]]
error_seq = [HashableNote(x) for x in error_dset[ind]]

s = difflib.SequenceMatcher(None, correct_seq, error_seq)
s.get_opcodes()




