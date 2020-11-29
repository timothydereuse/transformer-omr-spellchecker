import pretty_midi as pm
from collections import namedtuple
import numpy as np
import h5py
import model_params as params

lmd_root = r"D:\Documents\datasets\lakh_midi_dataset\lmd_full"
cleansed_ids_path = r"D:\Documents\datasets\lakh_midi_dataset\cleansed_ids.txt"
json_key = r"D:\Documents\datasets\lakh_midi_dataset\md5_to_paths.json"
dset_path = r"./lmd_cleansed.hdf5"

with h5py.File(dset_path, 'a') as f:
    f.attrs['beat_multiplier'] = params.beat_multiplier
    train_grp = f.create_group('train')
    test_grp = f.create_group('test')
    validate_grp = f.create_group('validate')

beat_multiplier = 24
with open(cleansed_ids_path) as f:
    rows = f.readlines()
md5s = [x.split(' ')[0] for x in rows]
md5s = list(set(md5s))
np.random.shuffle(md5s)

MIDINote = namedtuple('midi_note', 'program, start, end, pitch, velocity')

split_test = np.round(params.test_proportion * len(md5s))
split_validate = np.round(params.validate_proportion * len(md5s)) + split_test

for i, md5 in enumerate(md5s):

    if not i % 250:
        print(f'{i} of {len(md5s)}...')

    fname = rf'{lmd_root}/{md5[0]}/{md5}.mid'
    mid = pm.PrettyMIDI(fname)

    notes = []
    for inst in mid.instruments:
        if inst.is_drum:
            continue
        notes += [
            MIDINote(inst.program, n.start, n.end, n.pitch, n.velocity)
            for n
            in inst.notes
        ]

    notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    starts = np.array([n.start for n in notes])
    ends = np.array([n.end for n in notes])

    # get start and end times in terms of the beat structure of the midi file
    beats = mid.get_beats()
    start_beats = np.interp(starts, beats, np.arange(len(beats)))
    end_beats = np.interp(ends, beats, np.arange(len(beats)))
    start_beats = np.round(starts * beat_multiplier)
    end_beats = np.round(ends * beat_multiplier)

    arr = np.zeros([len(notes), 6], dtype='uint16')
    for i in range(len(notes)):
        # onset, duration, time to next onset, pitch, velocity, program
        n = notes[i]
        duration = end_beats[i] - start_beats[i]
        onset = start_beats[i]
        pitch = n.pitch
        velocity = n.velocity
        program = n.program
        arr[i] = [onset, duration, 0, pitch, velocity, program]

    onset_diffs = arr[1:, 0] - arr[:-1, 0]
    arr[:-1, 2] = onset_diffs

    with h5py.File(dset_path, 'a') as f:
        if i <= split_test:
            selected_subgrp = f['test']
        elif i <= split_validate:
            selected_subgrp = f['validate']
        else:
            selected_subgrp = f['train']

        dset = selected_subgrp.create_dataset(
            name=md5,
            data=arr,
            compression='gzip'
        )
