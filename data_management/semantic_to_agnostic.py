import music21 as m21

# fpath = r"D:\Documents\datasets\just_quartets\felix_errors\5_op44iii_4_omr.musicxml"
fpath = r"D:\Documents\datasets\just_quartets\kernscores\46072_op20n6-01.krn"
parsed_file = m21.converter.parse(fpath)
parts = list(parsed_file.getElementsByClass(m21.stream.Part))
part = parts[0].getElementsByClass(m21.stream.Measure)


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


def resolve_note(e, is_chord, current_clef, current_pitch_status):
    res = []

    accid_str, current_pitch_status = resolve_accidentals(e.pitch, current_pitch_status)

    # notes contain: duration-position-isChord-beamStatus
    dots = e.duration.dots
    e.duration.dots = 0
    d = str.lower(e.duration.fullName)

    # get staff position
    p = e.pitch.diatonicNoteNum - current_clef.lowestLine

    beams = list(e.beams)
    b = 'noBeam' if len(beams) == 0 else beams[0].type

    if(accid_str):
        res.append(f'accid.{accid_str}.pos{p}')

    c = 1 if is_chord else 0

    res.append(f'{d}.pos{p}.{b}.c{c}')

    # then add dot
    if dots > 0:
        res.append('duration.dot')

    return res, current_pitch_status


agnostic = []
current_clef = m21.clef.TrebleClef()
current_key_sig = m21.key.KeySignature(5)
current_pitch_status = get_reset_pitch_status(current_key_sig)
current_time_sig = m21.meter.TimeSignature('4/4')

# types of things we want to deal with:
# notes, chords, rests, barlines, dynamics, time signature, key signature, clef, SystemLayout
for measure in part:
    for e in measure:

        if type(e) == m21.note.Note:
            glyphs, current_pitch_status = resolve_note(e, False, current_clef, current_pitch_status) 
            agnostic.extend(glyphs)

        elif type(e) == m21.note.Rest:
            agnostic.append(f'rest.{str.lower(e.duration.fullName)}')

        elif type(e) == m21.chord.Chord:
            is_chord_list = [False] + [True for _ in e.notes]
            glyph_lists = []
            
            for i, x in enumerate(e.notes):
                gl, current_pitch_status = resolve_note(
                    x, is_chord_list[i], current_clef, current_pitch_status) 
                glyph_lists.extend(gl)
                
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

        elif issubclass(type(part[1]), m21.clef.Clef):
            current_clef = e
            agnostic.append(f'clef.{e.name}')

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
