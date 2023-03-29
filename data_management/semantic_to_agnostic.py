from tokenize import group
import music21 as m21
import os
import math
from collections import namedtuple
from itertools import groupby
from data_management.seq_builder import SequenceBuilder, MusicSeqRecord, sorting_order

def resolve_duration(d):
    dots = d.dots

    # if there's a durationexception when just trying to change the number
    # of dots then just return something - honestly, not sure how to
    # handle this in any productive way.
    try:
        d.dots = 0
    except m21.duration.DurationException:
        return dots, str.lower(d.fullName), 0

    if not d.tuplets:
        dur_string = str.lower(d.fullName)
        is_tuplet = 0
    else:
        dur_string = str.lower(d.tuplets[0].durationActual.type)
        is_tuplet = d.tuplets[0].tupletActual[0]
    if d.isGrace and d.slash:
        dur_string = 'acciaccatura.' + dur_string
    elif d.isGrace and not d.slash:
        dur_string = 'appoggiatura.' + dur_string 
    return dots, dur_string, is_tuplet


def resolve_note(e, current_clef, separate_sides=False):
    res = []
    
    show_accidental = e.pitch.accidental.displayStatus if e.pitch.accidental else None
    accid_str = e.pitch.accidental.name if show_accidental else None

    # collect all the items that will appear to the left of the note, then at the
    # note's position, then to the right of the note. merge them together at the end
    left_of_note = []
    at_note = []
    right_of_note = []

    # notes contain: duration-position-beamStatus
    dots, dur_name, is_tuplet = resolve_duration(e.duration)

    # get staff position
    p = e.pitch.diatonicNoteNum - current_clef.lowestLine 

    beams = list(e.beams)
    b = 'noBeam' if len(beams) == 0 else beams[0].type

    if(accid_str):
        left_of_note.extend([f'accid.{accid_str}.pos{p}'])

    # add actual note to result list
    at_note.extend([f'{dur_name}.{b}.{e.stemDirection}.pos{p}'])

    # then add dot
    if dots > 0:
        # duration dots are moved to only be on spaces
        dot_pos = p if (p % 2 == 0) else p + 1
        right_of_note.extend([f'duration.dot.pos{dot_pos}'])

    # if there's a tie, put it right after or before the note
    if e.tie and e.tie.type in ['start', 'continue']:
        right_of_note.extend([f'tie.start.pos{p}'])

    if e.tie and e.tie.type in ['end', 'continue']:
        left_of_note.insert(0, f'tie.end.pos{p}')

    if not separate_sides:
        res = vert_token(left_of_note) + vert_token(at_note) + vert_token(right_of_note)
    else:
        res = (vert_token(left_of_note), vert_token(at_note), vert_token(right_of_note))
    
    return res, is_tuplet


def vert_token(glyphs, marker='>'):
    # adds a given marker between every alternate element in the given list of glyphs,
    # excluding the last one and excluding already present markers
    removed_g = [g for g in glyphs if not g == marker]
    markers = [marker for _ in range(len(removed_g))]
    new_glyphs = [x for t in zip(removed_g, markers) for x in t][:-1]
    return new_glyphs


def resolve_articulations(e, current_clef, glyphs=None):

    if type(e) == m21.chord.Chord:
        is_stem_down = (e.notes[0].stemDirection == 'down')
        positions = [n.pitch.diatonicNoteNum - current_clef.lowestLine for n in e.notes]
        p = max(positions) if is_stem_down else min(positions)
        artic_pos = p if (p % 2 == 0) else p + 1
    else: 
        is_stem_down = (e.stemDirection == 'down')
        p = e.pitch.diatonicNoteNum - current_clef.lowestLine
        artic_pos = p if (p % 2 == 0) else p - 1

    # add articulations below note if the stem direction is up, or above note if the
    # stem direction is down. this probably isn't perfectly accurate but in order to
    # make this look "good" i would have to essentially program an entire engraving
    # system myself, right here, to know where the symbols "should" go

    if not glyphs:
        glyphs = []

    for articulation in e.articulations:
        art_name = f'articulation.{articulation.name}.pos{artic_pos}'
        if is_stem_down:
            glyphs = glyphs + ['>', art_name]
        else:
            # if there's no stem or anything assume stuff should be below the note
            glyphs = [art_name, '>'] + glyphs
    return glyphs


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
            mid_idx = min(len(next_keys_indices) // 2, len(next_keys_indices) - 1)
            idx_to_insert = ks[next_keys_indices[mid_idx]]
            to_insert[idx_to_insert] = to_check
            for x in next_keys_indices:
                del tr[ks[x]]

    return to_insert


def resolve_chord(e, current_clef):
    is_stem_down = e.notes[0].stemDirection == 'down'
    glyph_lists = []

    num_notes = len(e.notes)
    # keeps track of which note in the chord is being accessed
    index_trackers = [[], [], []]
    
    # sort notes in chord from lowest to highest
    sorted_notes = sorted(e.notes, key=lambda x: x.pitch.midi, reverse=False)

    lefts = []
    centers = []
    rights = []

    for i, x in enumerate(sorted_notes):
        sides, tuplet = resolve_note(x, current_clef, separate_sides=True)
        temp1 = (['>'] if len(lefts) > 0 and sides[0] else []) + sides[0]
        lefts.extend(temp1)
        index_trackers[0].extend([i for _ in range(len(temp1))])

        temp2 = (['>'] if len(centers) > 0 and sides[1] else []) + sides[1]
        centers.extend(temp2)
        index_trackers[1].extend([i for _ in range(len(temp2))])

        temp3 = (['>'] if len(rights) > 0 and sides[2] else []) + sides[2]
        rights.extend(temp3)
        index_trackers[2].extend([i for _ in range(len(temp3))])

    # have to do this separately using glyphs=None for the index tracker to still work. oof
    artics = resolve_articulations(e, current_clef, glyphs=None)
    if artics and is_stem_down:
        centers = centers + artics
        index_trackers[1] = [i for _ in range(len(artics) + 1)] + index_trackers[1]
    elif artics and not is_stem_down:
        centers = artics + centers
        index_trackers[1] = index_trackers[1] + [0 for _ in range(len(artics) + 1)]

    end_list = lefts + centers + rights
    end_tracker = [x for y in index_trackers for x in y]

    return end_list, end_tracker, tuplet


def m21_part_to_agnostic(part, part_idx):

    sb = SequenceBuilder()
    sb.part_idx = part_idx

    part = part.getElementsByClass(m21.stream.Measure)
        
    agnostic = []
    tuplet_record = {}
    all_clefs = part.flat.getElementsByClass(m21.clef.Clef)
    current_clef = all_clefs[0] if all_clefs else m21.clef.TrebleClef()

    all_keysigs = part.flat.getElementsByClass(m21.key.KeySignature)
    current_key_sig = all_keysigs[0] if all_keysigs else m21.key.KeySignature(0)
    # current_pitch_status = get_reset_pitch_status(current_key_sig)

    all_timesigs = part.flat.getElementsByClass(m21.meter.TimeSignature)

    # types of things we want to deal with:
    # notes, chords, rests, barlines, dynamics, time signature, key signature, clef, SystemLayout
    for measure_idx, measure in enumerate(part):
        sb.measure_idx = measure_idx
        last_offset = None
        last_type = None

        # expand voices to flatten voices out
        flat_measure = sorted(measure.flat[:], key=sorting_order, reverse=False)

        for event_idx, e in enumerate(flat_measure):
            sb.event_idx = event_idx

            if e.offset == last_offset and last_type in [
                m21.note.Note,
                m21.chord.Chord,
                m21.note.Rest,
                m21.dynamics.Dynamic
                ]:
                sb.add_record('>')
            last_offset = e.offset
            last_type = type(e)

            # case if the current m21 element is a Note
            if type(e) == m21.note.Note:
                glyphs, tuplet = resolve_note(e, current_clef, separate_sides=True)
                centers = resolve_articulations(e, current_clef, glyphs[1])
                glyphs = glyphs[0] + centers + glyphs[2]

                for g in glyphs:
                    sb.add_record(g)

                tuplet_record[len(agnostic)] = tuplet

            # case if the current m21 element is a Rest
            elif type(e) == m21.note.Rest:

                dots, dur_name, is_tuplet = resolve_duration(e.duration)
                sb.add_record(f'rest.{dur_name}')
                tuplet_record[len(agnostic)] = is_tuplet

                # add dot at position 5 for all rests that have a dot, which is fairly standard
                if dots > 0:
                    sb.add_record('duration.dot.pos5')

            # case if the current m21 element is a Chord
            # (must be split into notes and individually processed)
            elif type(e) == m21.chord.Chord:
                glyph_list, end_tracker, is_tuplet = resolve_chord(e, current_clef)

                for i, g in enumerate(glyph_list):
                    sb.add_record(g, chord_idx=end_tracker[i])
                tuplet_record[len(agnostic)] = is_tuplet

            # case if the current m21 element is a Dynamic marking
            elif type(e) == m21.dynamics.Dynamic:
                sb.add_record(f'dynamics.{e.value}')

            # case if the current m21 element is a Key Signature
            elif type(e) == m21.key.KeySignature:
                for p in e.alteredPitches:
                    position = (p.diatonicNoteNum - current_clef.lowestLine) % 12
                    sb.add_record(f'accid.{p.accidental.name}.pos{position}')
                current_key_sig = e 
            
            # case if the current m21 element is a Time Signature
            elif type(e) == m21.meter.TimeSignature:
                current_time_sig = e
                sb.add_record(f'timeSig.{e.ratioString}')

            # case if the current m21 element is a Clef
            elif issubclass(type(e), m21.clef.Clef):
                sb.add_record(f'clef.{e.name}')
                current_clef = current_clef if (e.lowestLine is None) else e

            # case where the current m21 element is a systemLayout
            # (must check to be sure it actually makes a new system)
            elif type(e) == m21.layout.SystemLayout and e.isNew:
                # restate clef and key signature
                glyphs = [f'lineBreak', f'clef.{current_clef.name}']
                for p in current_key_sig.alteredPitches:
                    position = p.diatonicNoteNum - current_clef.lowestLine - 12
                    glyphs.append(f'accid.{p.accidental.name}.{position}')
                for g in glyphs:
                    sb.add_record(g)

            # case where the current m21 element is a barline
            elif type(e) == m21.bar.Barline:
                sb.add_record(f'barline.{e.type}')

        # at the end of every measure, add a bar if there isn't one already
        if not 'barline' in sb.records[-1].music_element:
            sb.add_record(f'barline.regular')

    # handle tuplet records by putting in tuplet indicators
    insert_tuplet_marks = resolve_tuplet_record(tuplet_record)

    agnostic = sb.records

    # insert starting from the end to not mess up later indices
    for pos in sorted(list(insert_tuplet_marks.keys()), reverse=True):
        tuplet_str = f'tuplet.{insert_tuplet_marks[pos]}'
        measure_idx = agnostic[pos].measure_idx
        event_idx = agnostic[pos].event_idx
        tuplet_record = MusicSeqRecord(tuplet_str, 0, measure_idx, event_idx, part_idx)
        over_record = MusicSeqRecord('>', 0, measure_idx, event_idx, part_idx)
        agnostic.insert(pos, tuplet_record)
        agnostic.insert(pos, over_record)

    return agnostic


def m21_parts_to_interleaved_agnostic(parts, remove=None, transpose=None, interleave=True, fallback_num_bars_per_line=8, just_tokens=False):

    # get agnostic representation of each part
    if transpose:
        agnostic_parts = [m21_part_to_agnostic(p.transpose(transpose), i) for i, p in enumerate(parts)]
    else:
        agnostic_parts = [m21_part_to_agnostic(p, i) for i, p in enumerate(parts)]        

    # remove things specified in remove parameter
    if remove:
        agnostic_parts = [
            [x for x in part if not x.music_element in remove]
            for part
            in agnostic_parts
        ]

    # return all parts concatenated if no interleave needed
    if not interleave:
        all_parts_concat = [item for sublist in agnostic_parts for item in sublist]
        if just_tokens:
            return [x.music_element for x in all_parts_concat]
        return all_parts_concat

    last_measure = 1 + max([p[-1].measure_idx for p in agnostic_parts])

    # gets all parts grouped by measure index. nested comprehension to make sure it's a list
    # before groupby's weird ways mess up the iterators
    agnostic_parts_grouped = [
        [list(group) for key, group in groupby(p, key=lambda t: t.measure_idx)] 
        for p in agnostic_parts
    ]

    interleaved = []
    for i in range(last_measure):
        for p in agnostic_parts_grouped:
            try:
                interleaved.extend(p[i])
            except IndexError:
                pass
        l = interleaved[-1]
        interleaved.append(
            MusicSeqRecord('barline.return-to-top', 0, l.measure_idx, l.event_idx, l.part_idx)
            )

    # get locations of linebreaks in each part
    # staff_break_points = [
    #     [0] + [i for i, j in enumerate(part) if j.music_element == 'lineBreak']
    #     for part
    #     in agnostic_parts
    # ]

    # # get locations of barlines in each part
    # bar_break_points = [
    #     [0] + [i for i, j in enumerate(part) if 'barline' in j.music_element]
    #     for part
    #     in agnostic_parts
    # ]

    # # if there are no notated line breaks, insert them manually every couple of bars
    # if any([len(x) == 1 for x in staff_break_points]):
    #     staff_break_points = [x[::fallback_num_bars_per_line] for x in bar_break_points]

    # # if there is somehow a discrepancy in the number of staves per part, choose the minimum number per part
    # num_bars = [len(x) for x in staff_break_points]
    # if not all([num_bars[0] == x for x in num_bars]):
    #     staff_break_points = [x[:min(num_bars)] for x in staff_break_points]

    # # interleave parts together every few line breaks
    # interleaved = []
    # for i in range(min(num_bars) - 1):
    #     for j in range(len(agnostic_parts)):
    #         start = staff_break_points[j][i]
    #         end = staff_break_points[j][i + 1]
    #         interleaved += agnostic_parts[j][start:end]
    #     l = interleaved[-1]
    #     interleaved.append(
    #         MusicSeqRecord('barline.return-to-top', l.measure_idx, l.event_idx, l.part_idx)
    #         )

    if just_tokens:
        interleaved = [x.music_element for x in interleaved]

    return interleaved


def m21_streams_to_agnostic(mus_xmls, remove=None, transpose=None, interleave=True, fallback_num_bars_per_line=8, just_tokens=False):
    outputs = []
    for parsed_file in mus_xmls:
        parts = list(parsed_file.getElementsByClass(m21.stream.Part))

        agnostic = m21_parts_to_interleaved_agnostic(parts, remove=['+'])
        outputs.append(agnostic)
    return outputs


if __name__ == '__main__':
    from collections import Counter

    # files = [r"C:\Users\tim\Documents\felix_quartets_got_annotated\1_op12\C3\1_op12_1_aligned.musicxml",
    #         r"C:\Users\tim\Documents\felix_quartets_got_annotated\1_op12\C3\1_op12_2_aligned.musicxml"]

    xml_dir = r"C:\Users\tim\Documents\datasets\just_quartets\musescore_misc"
    files = [os.path.join(xml_dir, x) for x in os.listdir(xml_dir)]

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
        agnostic = m21_parts_to_interleaved_agnostic(parts, remove=['+'])
        all_tokens.update([x.music_element for x in agnostic])
