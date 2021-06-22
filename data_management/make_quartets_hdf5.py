import music21 as m21
import os
from collections import namedtuple
import numpy as np
import h5py

test_proportion = 0.1
validate_proportion = 0.1
dset_path = r'./felix_comparison.h5'
beat_multiplier = 48
quartets_root = r"D:\Documents\datasets\just_quartets"
train_val_test_split = False

keys = ['felix', 'felix_errors']
c = m21.converter.Converter()

with h5py.File(dset_path, 'a') as f:
    f.attrs['beat_multiplier'] = beat_multiplier
    if train_val_test_split:
        train_grp = f.create_group('train')
        test_grp = f.create_group('test')
        validate_grp = f.create_group('validate')

# voice, start, duration, midi_pitch, notated_pitch, accidental

def m21_note_to_tuple(x, voice_num, add_offset=0, make_list=True):
    if x.isChord:
        off = x.offset
        return [m21_note_to_tuple(z, voice_num, off, False) for z in x.notes]
    n = (
        # voice number
        voice_num, 
        # start beat in quarter notes
        x.offset * beat_multiplier + add_offset,
        # duration in quarter notes
        x.duration.quarterLength * beat_multiplier,
        # MIDI pitch
        x.pitch.midi if not x.isRest else 0,
        # diatonic pitch
        x.pitch.diatonicNoteNum if not x.isRest else 0,
        # accidental
        x.pitch.accidental.alter if (not x.isRest and x.pitch.accidental) else 0
    )
    n = tuple([int(z) for z in n])
    if make_list:
        n = [n]
    return n


for k in keys:
    files = os.listdir(os.path.join(quartets_root, k))
    np.random.shuffle(files)

    split_test = np.round(test_proportion * len(files))
    split_validate = np.round(validate_proportion * len(files)) + split_test

    for i, fname in enumerate(files):

        print(f'parsing {k}/{fname}...')

        fpath = os.path.join(quartets_root, k, fname)
        try:
            parsed_file = m21.converter.parse(fpath)
        except Exception:
            print(f'parsing {k}/{fname} failed, skipping file')
            continue
        parts = list(parsed_file.getElementsByClass(m21.stream.Part))

        notes = []
        for i, p in enumerate(parts):
            for item in p.flat.notesAndRests:
                notes.extend(m21_note_to_tuple(item, i))

        arr = np.array(notes)

        with h5py.File(dset_path, 'a') as f:
            if not train_val_test_split:
                selected_subgrp = f
            elif i <= split_test:
                selected_subgrp = f['test']
            elif i <= split_validate:
                selected_subgrp = f['validate']
            else:
                selected_subgrp = f['train']

            name = rf'{k}-{fname}'
            dset = selected_subgrp.create_dataset(
                name=name,
                data=arr,
                compression='gzip'
            )
