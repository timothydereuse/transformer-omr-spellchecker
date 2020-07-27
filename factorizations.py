import pretty_midi as pm
import numpy as np
from collections import namedtuple, Counter
import os


def get_tick_deltas_for_runlength(mids_path, num_dur_vals=16):
    midi_fnames = os.listdir(mids_path)

    c = Counter()

    min_pitches = []
    max_pitches = []

    for i, fname in enumerate(midi_fnames):
        pm_file = pm.PrettyMIDI(f"{mids_path}/{fname}")
        all_notes = []
        for voice in pm_file.instruments:
            if voice.is_drum:
                continue
            all_starts = [pm_file.time_to_tick(n.start) for n in voice.notes]
            all_notes += all_starts

            min_pitches.append(min([n.pitch for n in voice.notes]))
            max_pitches.append(max([n.pitch for n in voice.notes]))

        diffs = np.diff(all_starts)
        c.update(diffs)

        if not i % 200:
            print(f"processing tick deltas: {i} of {len(midi_fnames)}")

    most = [x[0] for x in c.most_common(num_dur_vals)]
    most = np.sort(most)
    res_dict = {v: i for i, v in enumerate(most)}

    pitch_range = (min(min_pitches), max(max_pitches))

    return res_dict, pitch_range


def pm_to_runlength(pm_file, tick_deltas_mapping, pitch_range, monophonic=False):
    '''
    takes in a single prettymidi object and turns it into a run-length encoding. for now, it
    collapses all non-drum channels down into a single piano roll.
    '''

    MIDIEvent = namedtuple("MIDIEvent", "type voice note tick")

    def dict_add(dict, event):
        if event.tick in dict.keys():
            dict[event.tick].append(event)
        else:
            dict[event.tick] = [event]

    events = {}
    for i, voice in enumerate(pm_file.instruments):

        if voice.is_drum:
            continue

        for n in voice.notes:

            start_ticks = pm_file.time_to_tick(n.start)
            start_event = MIDIEvent(type='start', voice=i, note=n.pitch, tick=start_ticks)
            end_ticks = pm_file.time_to_tick(n.end)
            end_event = MIDIEvent(type='end', voice=i, note=n.pitch, tick=end_ticks)

            dict_add(events, start_event)
            dict_add(events, end_event)

    # structure of a change point:
    # [1] absolute tick time of event in natural numbers
    # [128] onset marker in {0, 1}
    # [128] active note in {0, 1}

    # alternative way is to use new arrays for each monophonic voice.

    note_ons_rl = []
    note_holds_rl = []
    note_deltas_rl = []

    num_notes = pitch_range[1] - pitch_range[0]
    pitch_start = pitch_range[0]

    sorted_events = sorted(list(events.keys()))

    for i, t in enumerate(sorted_events):

        note_ons = np.zeros(num_notes, dtype='int16') - 1
        note_holds = np.zeros(num_notes, dtype='int16') - 1
        deltas = np.zeros(len(tick_deltas_mapping))

        # get length after this change point
        try:
            delta = sorted_events[i + 1] - t
            delta_ind = tick_deltas_mapping[delta]
        except IndexError:
            # if this is at the end of the file, assign the longest possible value
            delta_ind = -1
        except KeyError:
            # if this delta is uncommon, find the nearest common one
            delta_keys = np.array(list(tick_deltas_mapping.keys()))
            best_ind = np.argmin(np.abs(delta_keys - delta))
            delta_ind = tick_deltas_mapping[delta_keys[best_ind]]

        deltas[delta_ind] = 1

        # for notes whose statuses are set directly by notes involved in the current change point,
        # what do do is obvious:
        for change in events[t]:
            ind = change.note - pitch_start - 1
            if change.type == 'start':
                note_ons[ind] = 1
                note_holds[ind] = 1
            elif change.type == 'end':
                note_ons[ind] = 0
                note_holds[ind] = 0

        # for onset-controlling indices, they're always off if not set explicitly on
        note_ons[note_ons == -1] = 0

        # for hold-controlling indices, we copy their previous state
        try:
            prev_note_hold = note_holds_rl[-1]
        except IndexError:
            prev_note_hold = np.zeros(num_notes, dtype='int16')

        note_holds[note_holds == -1] = prev_note_hold[note_holds == -1]

        # run_length.append(pt)
        note_ons_rl.append(note_ons)
        note_holds_rl.append(note_holds)
        note_deltas_rl.append(deltas)

    if monophonic:
        new_note_holds = np.sum(np.stack(note_holds_rl), 1)
        new_note_holds = 1 - np.clip(new_note_holds, 0, 1)
        note_holds_rl = np.expand_dims(new_note_holds, 1)

    run_length = np.concatenate([
        np.stack(note_ons_rl),
        np.stack(note_holds_rl),
        np.stack(note_deltas_rl)], 1)

    return run_length


def pm_to_note_tuple(pm_file):

    NoteTuple = namedtuple("NoteTuple", "note start_ticks delta duration_ticks")

    notes = []
    for i, voice in enumerate(pm_file.instruments):
        if voice.is_drum:
            continue
        notes += voice.notes

    sorted_notes = sorted(notes, key=lambda x: x.start)
    tuples = []
    for i, n in enumerate(sorted_notes):
        start_ticks = pm_file.time_to_tick(n.start)
        end_ticks = pm_file.time_to_tick(n.end)
        duration_ticks = end_ticks - start_ticks
        delta = start_ticks - tuples[i - 1].start_ticks if i > 0 else 0

        tuples.append(NoteTuple(n.pitch, start_ticks, delta, duration_ticks))

    result = np.array([[x.note, x.delta, x.duration_ticks] for x in tuples])

    return result
