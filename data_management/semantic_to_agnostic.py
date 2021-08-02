import music21 as m21
import os
import math

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

    # notes contain: duration-position-isChord-beamStatus
    dots, dur_name, is_tuplet = resolve_duration(e.duration)

    # get staff position
    p = e.pitch.diatonicNoteNum - current_clef.lowestLine

    beams = list(e.beams)
    b = 'noBeam' if len(beams) == 0 else beams[0].type

    if(accid_str):
        res.append(f'accid.{accid_str}.pos{p}')

    c = 1 if is_chord else 0

    res.append(f'{dur_name}.pos{p}.{b}.c{c}')

    # then add dot
    if dots > 0:
        res.append(f'duration.dot.pos{p}')

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

def m21_part_to_agnostic(part):

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
    for measure in part:
        for e in measure:

            if type(e) == m21.note.Note:
                glyphs, tuplet, current_pitch_status = resolve_note(e, False, current_clef, current_pitch_status) 
                agnostic.extend(glyphs)

                tuplet_record[len(agnostic)] = tuplet

            elif type(e) == m21.note.Rest:

                dots, dur_name, is_tuplet = resolve_duration(e.duration)
                agnostic.append(f'rest.{dur_name}')
                tuplet_record[len(agnostic)] = is_tuplet

                # add dot at position 5 for all rests that have a dot, which is fairly standard
                if dots > 0:
                    agnostic.append(f'duration.dot.pos5')

            elif type(e) == m21.chord.Chord:
                is_chord_list = [False] + [True for _ in e.notes]
                glyph_lists = []
                
                # sort notes in chord from highest to lowest
                sorted_notes = sorted(e.notes, key=lambda x: x.pitch.midi, reverse=True)

                for i, x in enumerate(sorted_notes):
                    gl, tuplet, current_pitch_status = resolve_note(
                        x, is_chord_list[i], current_clef, current_pitch_status) 
                    glyph_lists.extend(gl)

                tuplet_record[len(agnostic)] = tuplet
                    
                agnostic.extend(glyph_lists)

            elif type(e) == m21.dynamics.Dynamic:
                agnostic.append(f'dynamics.{e.value}')

            elif type(e) == m21.key.KeySignature:
                for p in e.alteredPitches:
                    position = p.diatonicNoteNum - current_clef.lowestLine
                    agnostic.append(f'accid.{p.accidental.name}.pos{position}')
                current_key_sig = e 
            
            elif type(e) == m21.meter.TimeSignature:
                current_time_sig = e
                agnostic.append(f'timeSig.{e.ratioString}')

            elif issubclass(type(e), m21.clef.Clef):
                agnostic.append(f'clef.{e.name}')
                current_clef = current_clef if (e.lowestLine is None) else e

            elif type(e) == m21.layout.SystemLayout:
                # restate clef and key signature
                agnostic.append(f'systemBreak')
                agnostic.append(f'clef.{current_clef.name}')
                for p in current_key_sig.alteredPitches:
                    position = p.diatonicNoteNum - current_clef.lowestLine
                    agnostic.append(f'accid.{p.accidental.name}.{position}')

            elif type(e) == m21.bar.Barline:
                agnostic.append(f'barline.{e.type}')

        if not 'bar' in agnostic[-1]:
            agnostic.append('bar.regular')

        # at the end of every measure, reset the pitch status to the key signature
        current_pitch_status = get_reset_pitch_status(current_key_sig)

    # handle tuplet records by putting in tuplet indicators
    insert_tuplet_marks = resolve_tuplet_record(tuplet_record)
    # insert starting from the end to not mess up later indices
    for pos in sorted(list(insert_tuplet_marks.keys()), reverse=True):
        tuplet_str = f'tuplet.{insert_tuplet_marks[pos]}'
        agnostic.insert(pos, tuplet_str)

    # make sure no spaces remain in any of the entries
    for ind in range(len(agnostic)):
        agnostic[ind] = agnostic[ind].replace(' ', '')

    return agnostic

if __name__ == '__main__':
    from collections import Counter

    k = 'felix'
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

        for p in parts:
            agnostic = m21_part_to_agnostic(p)
            print(len(agnostic), len(set(agnostic)))
            all_tokens.update(agnostic)
            