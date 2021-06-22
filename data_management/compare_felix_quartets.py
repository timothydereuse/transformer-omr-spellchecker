import h5py
import numpy as np
import data_management.needleman_wunsch_alignment as align
from numba import njit

dset_path = r'./felix_comparison.h5'
# voice, start, duration, midi_pitch, notated_pitch, accidental
inds_subset = np.array([0, 2, 4, 5])

with h5py.File(dset_path, 'a') as f:

    correct_fnames = [x for x in f.keys() if 'aligned' in x and 'op80' not in x]
    error_fnames = [x for x in f.keys() if 'omr' in x]

    correct_dset = [f[x][:, inds_subset] for x in correct_fnames]
    error_dset = [f[x][:, inds_subset] for x in error_fnames]

ind = 7

correct_seq = [x for x in correct_dset[ind]]
error_seq = [x for x in error_dset[ind]]
# correct_seq = correct_dset[ind]
# error_seq = error_dset[ind]

a, b, r, score = align.perform_alignment(correct_seq[:1000], error_seq[:1000], match_weights=[4, -1], gap_penalties=[-1, -1, -2, -2])


sa = ''
sb = ''

for n in range(len(a)):
    spacing = str(max(len(a[n]), len(b[n])))
    sa += (f'{str(a[n]):4}')
    sb += (f'{str(a[n]):4}')

print(sa)
print(sb)
print(''.join(r))