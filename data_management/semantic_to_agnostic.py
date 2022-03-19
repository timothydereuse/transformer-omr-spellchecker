import music21 as m21
import os
import math
from collections import namedtuple

AgnosticRecord = namedtuple('AgnosticRecord', ['agnostic_item', 'measure_idx', 'event_idx', 'part_idx'])
DEFAULT_PITCH_STATUS = {x: 0 for x in list('ABCDEFG')}
def get_reset_pitch_status(current_key_sig):
    cps = dict(DEFAULT_PITCH_STATUS)
    for p in current_key_sig.alteredPitches:
        cps[p.name[0]] = int(p.alter)
    return cps


def resolve_accidentals(p, current_pitch_status):

    # check if this note's pitch alteration matches what is currently in the pitch status
    this_accid_num = int(p.alter)
    prev_accid_num = current_pitch_status[p.name[0]]

    if this_accid_num == prev_accid_num:
        # if it does, no need to do anything.
        accid_str = None
    else:
        # previously, no accidental on this note, so add one:
        accid_str = p.accidental.name if p.accidental else 'natural'
        current_pitch_status[p.name[0]] = this_accid_num
    
    return accid_str, current_pitch_status


def resolve_duration(d):
    dots = d.dots
    d.dots = 0
    if not d.tuplets:
        dur_string = str.lower(d.fullName)
        is_tuplet = 0
    else:
        dur_string = str.lower(d.tuplets[0].durationActual.type)
        is_tuplet = d.tuplets[0].tupletActual[0]
    
    if d.isGrace:
        dur_string = 'grace.' + dur_string 
    return dots, dur_string, is_tuplet


def resolve_note(e, is_chord, current_clef, current_pitch_status):
    res = []

    accid_str, current_pitch_status = resolve_accidentals(e.pitch, current_pitch_status)

    # notes contain: duration-position-beamStatus
    dots, dur_name, is_tuplet = resolve_duration(e.duration)

    # get staff position
    p = e.pitch.diatonicNoteNum - current_clef.lowestLine

    beams = list(e.beams)
    b = 'noBeam' if len(beams) == 0 else beams[0].type

    if(accid_str):
        res.extend(['+', f'accid.{accid_str}.pos{p}'])

    res.extend(['+', f'{dur_name}.pos{p}.{b}'])

    # then add dot
    if dots > 0:
        res.extend(['+', f'duration.dot.pos{p}'])

    return res, is_tuplet, current_pitch_status


def resolve_tuplet_record(tuplet_record):
    # replaces runs of the same tuplet value in tuplet record (3 triplets, 5 quintuplets, 7 septuplets, etc)
    # with just a central one
    tr = {k:tuplet_record[k] for k in tuplet_record.keys() if tuplet_record[k] > 0}
    ks = list(tr.keys())
    to_insert = {}

    for i in range(len(ks)):
        k = ks[i]
        try:
            to_check = tr[k]
        except KeyError:
            continue

        next_keys_indices = [i + x for x in range(to_check) if i + x < len(ks)]
        next_trs = [tr[ks[x]] == to_check for x in next_keys_indices]

        if all(next_trs):
            # get the middle + 1 index here
            mid_idx = min(len(next_keys_indices) // 2 + 1, len(next_keys_indices) - 1)
            idx_to_insert = ks[next_keys_indices[mid_idx]]
            to_insert[idx_to_insert] = to_check
            for x in next_keys_indices:
                del tr[ks[x]]

    return to_insert


def m21_part_to_agnostic(part, part_idx):

    part = part.getElementsByClass(m21.stream.Measure)
        
    agnostic = []
    tuplet_record = {}
    all_clefs = part.flat.getElementsByClass(m21.clef.Clef)
    current_clef = all_clefs[0] if all_clefs else m21.clef.TrebleClef

    all_keysigs = part.flat.getElementsByClass(m21.key.KeySignature)
    current_key_sig = all_keysigs[0] if all_keysigs else m21.key.KeySignature(0)
    current_pitch_status = get_reset_pitch_status(current_key_sig)

    all_timesigs = part.flat.getElementsByClass(m21.meter.TimeSignature)
    current_time_sig = all_timesigs[0] if all_keysigs else m21.meter.TimeSignature()


    # types of things we want to deal with:
    # notes, chords, rests, barlines, dynamics, time signature, key signature, clef, SystemLayout
    for measure_idx, measure in enumerate(part):
        for event_idx, e in enumerate(measure):

            # case if the current m21 element is a Note
            if type(e) == m21.note.Note:
                glyphs, tuplet, current_pitch_status = resolve_note(e, False, current_clef, current_pitch_status)

                records = [AgnosticRecord(g, measure_idx, event_idx, part_idx) for g in glyphs]
                agnostic.extend(records)

                tuplet_record[len(agnostic)] = tuplet

            # case if the current m21 element is a Rest
            elif type(e) == m21.note.Rest:

                dots, dur_name, is_tuplet = resolve_duration(e.duration)
                agnostic.append(AgnosticRecord('+', measure_idx, event_idx, part_idx))
                agnostic.append(AgnosticRecord(f'rest.{dur_name}', measure_idx, event_idx, part_idx))
                # agnostic.extend(['+', f'rest.{dur_name}'])
                tuplet_record[len(agnostic)] = is_tuplet

                # add dot at position 5 for all rests that have a dot, which is fairly standard
                if dots > 0:
                    agnostic.append(AgnosticRecord('+', measure_idx, event_idx, part_idx))
                    agnostic.append(AgnosticRecord('duration.dot.pos5', measure_idx, event_idx, part_idx))

            # case if the current m21 element is a Chord
            # (must be split into notes and individually processed)
            elif type(e) == m21.chord.Chord:
                is_chord_list = [False] + [True for _ in e.notes]
                glyph_lists = []
                
                # sort notes in chord from highest to lowest
                sorted_notes = sorted(e.notes, key=lambda x: x.pitch.midi, reverse=True)

                for i, x in enumerate(sorted_notes):
                    gl, tuplet, current_pitch_status = resolve_note(
                        x, is_chord_list[i], current_clef, current_pitch_status) 
                    glyph_lists.extend(gl)
                
                sustain_list = ['<' if x == '+' else x for x in glyph_lists]
                sustain_list[0] = '+'

                tuplet_record[len(agnostic)] = tuplet
                # agnostic.extend(sustain_list)
                records = [AgnosticRecord(g, measure_idx, event_idx, part_idx) for g in sustain_list]
                agnostic.extend(records)

            # case if the current m21 element is a Dynamic marking
            elif type(e) == m21.dynamics.Dynamic:
                agnostic.append(AgnosticRecord('+', measure_idx, event_idx, part_idx))
                agnostic.append(AgnosticRecord(f'dynamics.{e.value}', measure_idx, event_idx, part_idx))
                # agnostic.extend(['+', f'dynamics.{e.value}'])
                pass

            # case if the current m21 element is a Key Signature
            elif type(e) == m21.key.KeySignature:
                for p in e.alteredPitches:
                    position = p.diatonicNoteNum - current_clef.lowestLine - 12

                    agnostic.append(AgnosticRecord('+', measure_idx, event_idx, part_idx))
                    agnostic.append(AgnosticRecord(f'accid.{p.accidental.name}.pos{position}', measure_idx, event_idx, part_idx))
                    # agnostic.extend(['+', f'accid.{p.accidental.name}.pos{position}'])
                current_key_sig = e 
            
            # case if the current m21 element is a Time Signature
            elif type(e) == m21.meter.TimeSignature:
                current_time_sig = e
                agnostic.append(AgnosticRecord('+', measure_idx, event_idx, part_idx))
                agnostic.append(AgnosticRecord(f'timeSig.{e.ratioString}', measure_idx, event_idx, part_idx))
                # agnostic.extend(['+', f'timeSig.{e.ratioString}'])

            # case if the current m21 element is a Clef
            elif issubclass(type(e), m21.clef.Clef):
                # agnostic.extend(['+', f'clef.{e.name}'])
                agnostic.append(AgnosticRecord('+', measure_idx, event_idx, part_idx))
                agnostic.append(AgnosticRecord(f'clef.{e.name}', measure_idx, event_idx, part_idx))
                current_clef = current_clef if (e.lowestLine is None) else e

            # case where the current m21 element is a systemLayout
            # (must check to be sure it actually makes a new system)
            elif type(e) == m21.layout.SystemLayout and e.isNew:
                # restate clef and key signature
                glyphs = ['+', f'lineBreak', '+', f'clef.{current_clef.name}']
                records = [AgnosticRecord(g, measure_idx, event_idx, part_idx) for g in glyphs]
                agnostic.extend(records)
                
                for p in current_key_sig.alteredPitches:
                    position = p.diatonicNoteNum - current_clef.lowestLine - 12
                    agnostic.append(AgnosticRecord('+', measure_idx, event_idx, part_idx))
                    agnostic.append(AgnosticRecord(f'accid.{p.accidental.name}.{position}', measure_idx, event_idx, part_idx))
                    # agnostic.extend(['+', f'accid.{p.accidental.name}.{position}'])

            # case where the current m21 element is a barline
            elif type(e) == m21.bar.Barline:
                # agnostic.extend(['+', f'barline.{e.type}'])
                agnostic.append(AgnosticRecord('+', measure_idx, event_idx, part_idx))
                agnostic.append(AgnosticRecord(f'barline.{e.type}', measure_idx, event_idx, part_idx))

        # at the end of every measure, add a bar if there isn't one already
        if not 'bar' in agnostic[-1]:
            # agnostic.extend(['+', 'barline.regular'])
            agnostic.append(AgnosticRecord('+', measure_idx, event_idx, part_idx))
            agnostic.append(AgnosticRecord(f'barline.regular', measure_idx, event_idx, part_idx))

        # at the end of every measure, reset the pitch status to the key signature
        current_pitch_status = get_reset_pitch_status(current_key_sig)

    # handle tuplet records by putting in tuplet indicators
    insert_tuplet_marks = resolve_tuplet_record(tuplet_record)
    # insert starting from the end to not mess up later indices
    for pos in sorted(list(insert_tuplet_marks.keys()), reverse=True):
        tuplet_str = f'tuplet.{insert_tuplet_marks[pos]}'
        measure_idx = agnostic[pos].measure_idx
        event_idx = agnostic[pos].event_idx
        tuplet_record = AgnosticRecord(tuplet_str, measure_idx, event_idx, part_idx)
        over_record = AgnosticRecord('<', measure_idx, event_idx, part_idx)
        agnostic.insert(pos, tuplet_record)
        agnostic.insert(pos, over_record)

    # # make sure no spaces remain in any of the entries
    # for ind in range(len(agnostic)):
    #     agnostic[ind] = agnostic[ind].replace(' ', '')

    return agnostic


def m21_parts_to_interleaved_agnostic(parts, remove=None, transpose=None, fallback_num_bars_per_line=8, just_tokens=False):

    # get agnostic representation of each part
    if transpose:
        agnostic_parts = [m21_part_to_agnostic(p.transpose(transpose), i) for i, p in enumerate(parts)]
    else:
        agnostic_parts = [m21_part_to_agnostic(p, i) for i, p in enumerate(parts)]

    if remove:
        agnostic_parts = [
            [x for x in part if not x.agnostic_item in remove]
            for part
            in agnostic_parts
        ]

    # get locations of barlines in each part
    bar_break_points = [
        [0] + [i for i, j in enumerate(part) if 'barline' in j.agnostic_item]
        for part
        in agnostic_parts
    ]

    # get locations of linebreaks in each part
    staff_break_points = [
        [0] + [i for i, j in enumerate(part) if j.agnostic_item == 'lineBreak']
        for part
        in agnostic_parts
    ]

    # if there are no notated line breaks, insert them manually every couple of bars
    if any([len(x) == 1 for x in staff_break_points]):
        staff_break_points = [x[::fallback_num_bars_per_line] for x in bar_break_points]

    # if there is somehow a discrepancy in the number of staves per part, choose the minimum number per part
    num_bars = [len(x) for x in staff_break_points]
    if not all([num_bars[0] == x for x in num_bars]):
        staff_break_points = [x[:min(num_bars)] for x in staff_break_points]

    # interleave parts together every few line breaks
    interleaved_agnostic = []
    for i in range(min(num_bars) - 1):
        for j in range(len(agnostic_parts)):
            start = staff_break_points[j][i]
            end = staff_break_points[j][i + 1]
            interleaved_agnostic += agnostic_parts[j][start:end]

    if just_tokens:
        interleaved_agnostic = [x.agnostic_item for x in interleaved_agnostic]

    return interleaved_agnostic



if __name__ == '__main__':
    from collections import Counter

    k = 'kernscores'
    quartets_root = r'D:\Documents\datasets\just_quartets'
    files = os.listdir(os.path.join(quartets_root, k))
    all_tokens = Counter()

    for fname in files:
        fpath = os.path.join(os.path.join(quartets_root, k, fname))
        parsed_file = m21.converter.parse(fpath)
        parts = list(parsed_file.getElementsByClass(m21.stream.Part))
        # part = parts[0].getElementsByClass(m21.stream.Measure)
        print(f'processing {k}.{fname}')
        print(f'ntokens {len(all_tokens)}')

        # for p in parts:
        #     agnostic = m21_part_to_agnostic(p)
        #     print(len(agnostic), len(set(agnostic)))
        #     all_tokens.update(agnostic)
        agnostic = m21_parts_to_interleaved_agnostic(parts, remove=['+'])
        all_tokens.update([x.agnostic_item for x  in agnostic])

            
