import music21 as m21

fpath = r"D:\Documents\datasets\just_quartets\felix_errors\5_op44iii_4_omr.musicxml"
parsed_file = m21.converter.parse(fpath)
parts = list(parsed_file.getElementsByClass(m21.stream.Part))

part = parts[0].getElementsByClass(m21.stream.Measure)

agnostic = []
current_clef = m21.clef.TrebleClef()
current_key_sig = m21.key.KeySignature(5)
current_key_sig_mod = []
current_time_sig = m21.meter.TimeSignature('4/4')

# types of things we want to deal with:
# notes, chords, rests, barlines, dynamics, time signature, key signature, clef, SystemLayout

def resolve_note(e, is_chord, current_clef, current_key_sig, current_key_sig_mod):
    res = []

    # notes contain: duration-position-isChord-beamStatus
    dots = e.duration.dots
    e.duration.dots = 0
    d = str.lower(e.duration.fullName)

    # get staff position
    p = e.pitch.diatonicNoteNum - current_clef.lowestLine

    beams = list(e.beams)
    b = 'noBeam' if len(beams) == 0 else beams[0].type

    if(e.pitch.accidental):
        res.append(f'accid.{e.pitch.accidental.name}.{p}')

    c = 1 if is_chord else 0

    res.append(f'{d}.{c}.{p}.{b}')

    # then add dot
    if dots > 0:
        res.append('duration.dot')

    return res


for measure in part:
    for e in measure:

        if type(e) == m21.note.Note:
            glyphs = resolve_note(e, False, current_clef, current_key_sig, current_key_sig_mod) 
            agnostic.extend(glyphs)

        elif type(e) == m21.note.Rest:
            agnostic.append(f'rest.{e.duration.quarterLength}')

        elif type(e) == m21.chord.Chord:
            is_chord_list = [False] + [True for _ in e.notes]
            glyph_lists = [
                resolve_note(x, is_chord_list[i], current_clef, current_key_sig, current_key_sig_mod) 
                for i, x in enumerate(e.notes)
            ]
            for gl in glyph_lists:
                agnostic.extend(gl)

        elif type(e) == m21.dynamics.Dynamic:
            agnostic.append(f'dynamics.{e.value}')

        elif type(e) == m21.key.KeySignature:
            for p in e.alteredPitches:
                position = p.diatonicNoteNum - current_clef.lowestLine
                agnostic.append(f'accid.{p.accidental.name}.{position}')
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
        
# print(agnostic)
