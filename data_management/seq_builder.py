from collections import namedtuple
import music21 as m21

AgnosticRecord = namedtuple('AgnosticRecord', ['music_element', 'chord_idx', 'measure_idx', 'event_idx', 'part_idx'])

class SequenceBuilder():

    def __init__(self) -> None:
        self.records = []
        self.part_idx = 0
        self.measure_idx = 0
        self.event_idx = 0
        self.chord_idx = 0

    def add_record(self, token, chord_idx=None, measure_idx=None, event_idx=None, part_idx=None) -> None:
        new_rec = AgnosticRecord(
            token,
            chord_idx if chord_idx else self.chord_idx,
            measure_idx if measure_idx else self.measure_idx,
            event_idx if event_idx else self.event_idx,
            part_idx if part_idx else self.part_idx,
        )
        self.records.append(new_rec)
        

def sorting_order(e):
    # returns a tuple by which you can sort elements to get the order
    # in which they should be processed by m21_part_to_agnostic
    if hasattr(e, 'pitch'):
        return (e.offset, e.pitch.diatonicNoteNum)
    elif type(e) == m21.chord.Chord:
        lowest_note = min(e.notes, key=lambda x: x.pitch.midi)
        return (e.offset, lowest_note.pitch.diatonicNoteNum)
    elif type(e) == m21.note.Rest:
        return (e.offset, 1)
    else:
        return (e.offset, -100)