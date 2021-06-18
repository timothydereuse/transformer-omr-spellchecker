import music21 as m21
import os
from collections import namedtuple
import numpy as np
import h5py

test_proportion = 0.1
validate_proportion = 0.1
dset_path = r'./quartets_dset.h5'
beat_multiplier = 48
quartets_root = r"D:\Documents\datasets\just_quartets"

keys = os.listdir(quartets_root)
c = m21.converter.Converter()

with h5py.File(dset_path, 'a') as f:
    f.attrs['beat_multiplier'] = beat_multiplier
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
    split_test = np.round(test_proportion * len(files))
    split_validate = np.round(validate_proportion * len(files)) + split_test

    for f in files:
        fname = os.path.join(quartets_root, k, f)
        parsed_file = m21.converter.parse(fname)
        parts = list(parsed_file.getElementsByClass(m21.stream.Part))

        notes = []
        for i, p in enumerate(parts):
            for item in p.flat.notesAndRests:
                notes.extend(m21_note_to_tuple(item, i))
        print(notes)
        

    # notes = []
    # for inst in mid.instruments:
    #     if inst.is_drum:
    #         continue
    #     notes += [
    #         MIDINote(inst.program, n.start, n.end, n.pitch, n.velocity)
    #         for n
    #         in inst.notes
    #     ]

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
