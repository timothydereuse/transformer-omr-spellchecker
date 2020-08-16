import h5py
import numpy as np
import music21 as m21
import os

beat_multiplier = 48

f = h5py.File('essen_meertens_songs.hdf5', 'a')
f.attrs['beat_multiplier'] = beat_multiplier

paths = {
    'essen': r"D:\Documents\datasets\essen\europa",
    'meertens': r"D:\Documents\datasets\meertens_tune_collection\mtc-fs-1.0.tar\krn"
}

for p in paths.keys():

    grp = f.create_group(p)

    all_krns = []
    for root, dirs, files in os.walk(paths[p]):
        for name in files:
            if '.krn' in name:
                all_krns.append(os.path.join(root, name))

    for krn_fname in all_krns:

        print(f'processing {krn_fname}...')

        krn = m21.converter.parse(krn_fname)
        arr = np.array([[
            n.pitch.midi if n.isNote else 0,
            int(n.offset * beat_multiplier),
            int((n.offset + n.duration.quarterLength) * beat_multiplier)
        ] for n in krn.flat.notesAndRests])

        dset = grp.create_dataset(
            name=krn_fname.split('\\')[-1],
            data=arr
        )

        try:
            time_sig = list(krn.flat.getElementsByClass('TimeSignature'))[0]
            dset.attrs['time_signature'] = (time_sig.numerator, time_sig.denominator)
        except IndexError:
            dset.attrs['time_signature'] = (-1, -1)

        try:
            key = list(krn.flat.getElementsByClass('Key'))[0]
            dset.attrs['key'] = key.tonicPitchNameWithCase
        except IndexError:
            dset.attrs['key'] = -1
