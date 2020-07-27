import os
import pretty_midi as pm
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from factorizations import pm_to_runlength

midis_path = r"D:\Documents\kernscores\files\midi"

midi_fnames = []
for subdir, dirs, files in os.walk(midis_path):
    for file in files:
        midi_fnames.append(rf"{midis_path}/{file}")

kern_file = pm.PrettyMIDI(midi_fnames[1949])

run_length = pm_to_runlength(kern_file)
# scale tick length values for similarity calculation convenience
asdf = [x[-1] for x in run_length]
reduced = np.gcd.reduce(list(set(asdf)))
run_length[:, -1] = run_length[:, -1] // reduced


def compare_tokens(A, B):

    time_factor = A[-1] * B[-1]

    notes_factor = np.dot(A[:-1], B[:-1])
    # print(time_factor, notes_factor)

    return np.log(time_factor * notes_factor + 1)


ssm = np.empty([len(run_length), len(run_length)])
for x, y in product(range(ssm.shape[0]), range(ssm.shape[1])):
    ssm[x, y] = compare_tokens(run_length[x], run_length[y])

plt.clf()
plt.imshow(ssm)
plt.show()
