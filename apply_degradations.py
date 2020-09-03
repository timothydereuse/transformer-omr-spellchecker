from mdtk import fileio, degradations
import numpy as np
import pandas as pd

fpath = r"D:\Documents\datasets\meertens_tune_collection\mtc-fs-1.0.tar\midi\NLB015532_01.mid"
# x = fileio.midi_to_df(fpath)




def arr_to_df(inp):
    rearranged_song = np.stack([
        inp[:, 1],
        np.zeros(inp.shape[0]),
        inp[:, 0],
        inp[:, 2]
    ], 1)
    res = pd.DataFrame(rearranged_song, columns=['onset', 'track', 'pitch', 'dur'])
    return res


def df_to_arr(inp):
    pass

if __name__ == '__main__':

    import h5py

    dset_fname = r'essen_meertens_songs.hdf5'
    f = h5py.File(dset_fname, 'r')

    fnames = []
    for k in f.keys():
        for n in f[k].keys():
            fnames.append(rf'{k}/{n}')

    song = f[fnames[4]][:]

    x = arr_to_df(song)
    x = degradations.pitch_shift(x)
