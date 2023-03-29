from tokenize import group
import music21 as m21
import os
import numpy as np
from collections import namedtuple
from itertools import groupby
from data_management.seq_builder import SequenceBuilder, MusicSeqRecord, sorting_order


def m21_parts_to_notetuple(parts):
    sb = SequenceBuilder()

    # zip together simultaneous measures
    part_measures = zip(*[part.getElementsByClass(m21.stream.Measure)[:] for part in parts])

    for measure_idx, measure_group in enumerate(part_measures):

        # get all elements of this measure into a single list, augmented with
        # info on their idx locations in the original m21 stream
        all_elements = []
        for part_idx, measure in enumerate(measure_group):
            for event_idx, e in enumerate(measure.flat):
                all_elements.append((e, measure_idx, event_idx, part_idx))

        disallowed_element_types = [
            m21.layout.SystemLayout,
            m21.key.KeySignature,
            m21.meter.TimeSignature,
            m21.layout.StaffLayout,
            m21.layout.PageLayout
            ]

        all_elements = [x for x in all_elements if not (type(x[0]) in disallowed_element_types)]

        sorted_measure = sorted(all_elements, key=lambda x: sorting_order(x[0]), reverse=False)
        deltas = np.diff([x[0].offset for x in sorted_measure] + [measure_group[0].duration.quarterLength])

        for i in range(len(sorted_measure)):
            e = sorted_measure[i][0]
            delta = deltas[i]
            sb.measure_idx = sorted_measure[i][1]
            sb.event_idx = sorted_measure[i][2]
            sb.part_idx = sorted_measure[i][3]

            if type(e) == m21.note.Note:
                duration = e.duration.quarterLength
                name = e.pitch.name
                sb.add_record((name, delta, duration, sb.part_idx))
            elif type(e) == m21.note.Rest:
                duration = e.duration.quarterLength
                name = e.name
                sb.add_record((name, delta, duration, sb.part_idx))
            elif type(e) == m21.bar.Barline:
                name = e.type
                sb.add_record((name, delta, 0, sb.part_idx))
            elif type(e) == m21.dynamics.Dynamic:
                name = e.value
                sb.add_record((name, delta, 0, sb.part_idx))
            elif type(e) == m21.chord.Chord:
                for chord_idx, note in enumerate(e.notes):
                    duration = e.duration.quarterLength
                    name = note.name
                    note_delta = delta if chord_idx == len(e.notes) + 1 else 0
                    sb.add_record((name, note_delta, duration, sb.part_idx), chord_idx=chord_idx)          
            else:
                duration = 0
                name = e.name
                sb.add_record((name, delta, duration, sb.part_idx))     

    return sb.records


if __name__ == '__main__':
    from collections import Counter

    files = [r"C:\Users\tim\Documents\felix_quartets_got_annotated\1_op12\C3\1_op12_1_aligned.musicxml",
            r"C:\Users\tim\Documents\felix_quartets_got_annotated\1_op12\C3\1_op12_2_aligned.musicxml"]

    # xml_dir = r"C:\Users\tim\Documents\datasets\just_quartets\musescore_misc"
    # files = [os.path.join(xml_dir, x) for x in os.listdir(xml_dir)]

    # files = [r"C:\Users\tim\Documents\tex\dissertation\score examples\agnostic encoding example 1.mxl"]

    all_tokens = Counter()

    for fpath in files:
        try:
            parsed_file = m21.converter.parse(fpath)
            parts = list(parsed_file.getElementsByClass(m21.stream.Part))
        except Exception:
            print(f'parsing {fpath} failed, skipping file')
            continue

        # part = parts[0].getElementsByClass(m21.stream.Measure)
        print(f'ntokens {len(all_tokens)}')

        # for p in parts:
        #     agnostic = m21_part_to_agnostic(p)
        #     print(len(agnostic), len(set(agnostic)))
        #     all_tokens.update(agnostic)
        agnostic = m21_parts_to_notetuple(parts)
        assert False
        # all_tokens.update([x.music_element for x in agnostic])
