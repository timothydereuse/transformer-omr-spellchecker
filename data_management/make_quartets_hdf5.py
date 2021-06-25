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

# voice, start, time_to_next_offset, duration, midi_pitch, notated_pitch, accidental

def m21_note_to_tuple(x, voice_num, previous_offset):

    pitches = x.pitches if not x.isRest else (m21.pitch.Pitch(midi=0),)

    n = [(
        # voice number
        voice_num, 
        # start beat in quarter notes
        int(x.offset * beat_multiplier),
        # time since previous offset, in quarter notes
        max(0, int(x.offset * beat_multiplier - previous_offset)),
        # duration in quarter notes
        int(x.duration.quarterLength * beat_multiplier),
        # MIDI pitch
        p.midi if not x.isRest else 0,
        # diatonic pitch
        p.diatonicNoteNum if not x.isRest else 0,
        # accidental
        p.accidental.alter if (not x.isRest and p.accidental) else 0
    ) for p in pitches]

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
        prev_note_offset = 0
        for i, p in enumerate(parts):
            for item in p.flat.notesAndRests:
                notes.extend(m21_note_to_tuple(item, i, prev_note_offset))
                prev_note_offset = notes[-1][1]

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
