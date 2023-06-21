from tokenize import group
import music21 as m21
import os
import numpy as np
from collections import namedtuple
from itertools import groupby
from data_management.seq_builder import SequenceBuilder, MusicSeqRecord, sorting_order


Notetuple = namedtuple(
    "Notetuple", ["name", "delta", "duration", "part_idx", "start_time", "end_time"]
)
MIDILike = namedtuple(
    "MIDILike", ["name", "part_idx", "delta", "time", "on_off_instant"]
)


def m21_parts_to_notetuple(parts, interleaved=True):

    if not interleaved:
        result = []
        for i, p in enumerate(parts):
            processed = m21_parts_to_notetuple([p], interleaved=True)
            tup = Notetuple("next_part", 0, 0, i, 0, 0)
            rec = MusicSeqRecord(tup, 0, 0, 0, i)
            result.extend([rec] + processed)
        return result

    sb = SequenceBuilder()

    stripped_parts = [s.stripTies() for s in parts]

    # zip together simultaneous measures
    part_measures = zip(
        *[part.getElementsByClass(m21.stream.Measure)[:] for part in stripped_parts]
    )

    measure_cumulative_dur = 0
    for measure_idx, measure_group in enumerate(part_measures):

        # get all elements of this measure into a single list, augmented with
        # info on their idx locations in the original m21 stream
        all_elements = []
        for part_idx, measure in enumerate(measure_group):
            for event_idx, e in enumerate(measure.flat):
                all_elements.append((e, measure_idx, event_idx, part_idx))

        disallowed_element_types = [
            m21.layout.SystemLayout,
            m21.layout.StaffLayout,
            m21.layout.PageLayout,
            m21.expressions.TextExpression,
            m21.tempo.MetronomeMark,
            m21.harmony.ChordSymbol,
        ]

        all_elements = [
            x for x in all_elements if not (type(x[0]) in disallowed_element_types)
        ]

        this_measure_dur = measure_group[0].duration.quarterLength
        sorted_measure = sorted(
            all_elements, key=lambda x: sorting_order(x[0]), reverse=False
        )

        measure_offsets = [x[0].offset for x in sorted_measure]
        deltas = np.diff(measure_offsets + [this_measure_dur])

        for i in range(len(sorted_measure)):
            e = sorted_measure[i][0]
            delta = deltas[i]
            sb.measure_idx = sorted_measure[i][1]
            sb.event_idx = sorted_measure[i][2]
            sb.part_idx = sorted_measure[i][3]

            # handle chords separately since they represent multiple events
            if type(e) == m21.chord.Chord:
                duration = e.duration.quarterLength
                end_time = measure_cumulative_dur + duration + measure_offsets[i]
                start_time = measure_cumulative_dur + measure_offsets[i]
                for chord_idx, note in enumerate(e.notes):
                    name = f"{note.pitch.nameWithOctave}"
                    note_delta = delta if chord_idx == len(e.notes) + 1 else 0
                    rec = Notetuple(
                        name, note_delta, duration, sb.part_idx, start_time, end_time
                    )
                    sb.add_record(rec)
                continue

            if type(e) == m21.note.Note:
                duration = e.duration.quarterLength
                name = f"{e.pitch.nameWithOctave}"
            elif type(e) == m21.note.Rest:
                duration = e.duration.quarterLength
                name = e.name
            elif type(e) in [m21.bar.Barline, m21.bar.Repeat]:
                name = f"bar_{e.type}"
                duration = 0
            elif type(e) == m21.dynamics.Dynamic:
                name = e.value
                duration = 0
            elif type(e) == m21.key.KeySignature:
                duration = 0
                name = f"keysig_{e.sharps}fifths"
            elif type(e) == m21.meter.TimeSignature:
                duration = 0
                name = f"timesig_{e.ratioString}"
            else:
                duration = 0
                name = e.name

            end_time = measure_cumulative_dur + duration + measure_offsets[i]
            start_time = measure_cumulative_dur + measure_offsets[i]

            rec = Notetuple(name, delta, duration, sb.part_idx, start_time, end_time)
            sb.add_record(rec)

        measure_cumulative_dur += this_measure_dur

    return sb.records


def notetuple_to_MIDILike(records):
    new_records = []
    intermed = []
    for r in records:
        e = r.music_element
        if e.name == "rest":
            continue
        if e.duration == 0:
            intermed.append([r, e.start_time, "instant"])
        else:
            intermed.append([r, e.start_time, "on"])
            intermed.append([r, e.end_time, "off"])

    sorted_recs = sorted(
        intermed,
        key=lambda x: (
            x[1],
            x[2],
            x[0].music_element.part_idx,
            x[0].music_element.name,
        ),
    )

    deltas = np.diff([x[1] for x in sorted_recs])
    deltas = np.concatenate([deltas, [0]])

    for i in range(len(sorted_recs)):
        r, event_time, event_type = sorted_recs[i]
        e = r.music_element
        delta = deltas[i]
        el = MIDILike(e.name, e.part_idx, delta, event_time, event_type)
        rec = MusicSeqRecord(el, r.chord_idx, r.measure_idx, r.event_idx, r.part_idx)
        new_records.append(rec)

    return new_records


def MIDILike_to_EventLike(records, include_parts=False):
    event_records = []
    current_state_on = True
    current_part = -1

    for r in records:
        new_els = []
        e = r.music_element

        if (include_parts == False) and not (current_part == e.part_idx):
            new_els.append(f"part {e.part_idx}")
            current_part = e.part_idx

        if e.delta > 0:
            new_els.append(f"delta_{e.delta}")

        if (e.on_off_instant == "off") and current_state_on:
            # switch state to off
            new_els.append("notes_off")
            current_state_on = False
        elif (e.on_off_instant == "on") and (not current_state_on):
            # switch state to on
            new_els.append("notes_on")
            current_state_on = True

        main_name = f"{e.name}_{e.part_idx}" if include_parts else e.name
        new_els.append(main_name)

        new_recs = [
            MusicSeqRecord(x, r.chord_idx, r.measure_idx, r.event_idx, r.part_idx)
            for x in new_els
        ]
        event_records.extend(new_recs)

    return event_records


def notetuple_string(records):
    els = [x.music_element for x in records]
    return [f"({x.name}, {x.delta}, {x.duration})" for x in els]


def midilike_string(records):
    els = [x.music_element for x in records]
    return [f"({x.name}, {x.delta}, {x.on_off_instant})" for x in els]


def eventlike_string(records):
    els = [x.music_element for x in records]
    return els


def m21_parts_to_eventlike(parts, interleaved):
    x1 = m21_parts_to_notetuple(parts, interleaved=interleaved)
    x2 = notetuple_to_MIDILike(x1)
    return MIDILike_to_EventLike(x2, include_parts=False)


def m21_parts_to_MIDILike(parts, interleaved):
    x1 = m21_parts_to_notetuple(parts, interleaved=interleaved)
    return notetuple_to_MIDILike(x1)


if __name__ == "__main__":
    from collections import Counter

    # files = [r"C:\Users\tim\Documents\felix_quartets_got_annotated\1_op12\C3\1_op12_1_aligned.musicxml",
    #         r"C:\Users\tim\Documents\felix_quartets_got_annotated\1_op12\C3\1_op12_2_aligned.musicxml"]

    # xml_dir = r"C:\Users\tim\Documents\datasets\just_quartets\musescore_misc"
    # files = [os.path.join(xml_dir, x) for x in os.listdir(xml_dir)]

    files = [r"C:\Users\tim\Documents\datasets\solo_piano\gstring_snippet2.mxl"]

    all_tokens = Counter()

    for fpath in files:
        try:
            parsed_file = m21.converter.parse(fpath)
            parts = list(parsed_file.getElementsByClass(m21.stream.Part))
        except Exception:
            print(f"parsing {fpath} failed, skipping file")
            continue

        # part = parts[0].getElementsByClass(m21.stream.Measure)
        print(f"ntokens {len(all_tokens)}")

        # for p in parts:
        #     agnostic = m21_part_to_agnostic(p)
        #     print(len(agnostic), len(set(agnostic)))
        #     all_tokens.update(agnostic)
        records = m21_parts_to_notetuple(parts, interleaved=False)
        midilike_records = notetuple_to_MIDILike(records)
        eventlike_records = MIDILike_to_EventLike(midilike_records, include_parts=False)
        # all_tokens.update([(
        #     x.music_element.name,
        #     x.music_element.delta,
        #     x.music_element.on_off_instant
        # ) for x in midilike_records])
        all_tokens.update([x.music_element for x in eventlike_records])
