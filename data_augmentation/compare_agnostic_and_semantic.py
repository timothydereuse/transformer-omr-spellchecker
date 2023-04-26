import h5py
import numpy as np
import data_augmentation.needleman_wunsch_alignment as align
import data_augmentation.error_gen_logistic_regression as elgr
import data_management.semantic_to_notetuple as stn
import data_management.semantic_to_agnostic as sta
import music21 as m21
from collections import Counter
import data_management.vocabulary as vocabulary

omr_file = r"C:\Users\tim\Documents\datasets\solo_piano\J S Bach - Air on the G String  Piano solo arr.  from BWV 1068_snippeterrors.mxl"
correct_file = r"C:\Users\tim\Documents\datasets\solo_piano\J S Bach - Air on the G String  Piano solo arr.  from BWV 1068_snippet.mxl"

omr_parsed = m21.converter.parse(omr_file).getElementsByClass(m21.stream.Part)
correct_parsed = m21.converter.parse(correct_file).getElementsByClass(m21.stream.Part)

def get_reps(m21_stream):
    reps = [
        stn.notetuple_string(stn.m21_parts_to_notetuple(m21_stream)),
        stn.midilike_string(stn.m21_parts_to_MIDILike(m21_stream)),
        stn.eventlike_string(stn.m21_parts_to_eventlike(m21_stream)),
        [x.music_element for x in sta.m21_parts_to_interleaved_agnostic(m21_stream)],
    ]
    return reps

omr_groups = get_reps(omr_parsed)
correct_groups = get_reps(correct_parsed)


groups = list(zip(omr_groups, correct_groups))

res = []
error_rates = []
for i, group in enumerate(groups):
    print(f'processing group {i}...')
    omr_version, correct_version = group

    v = vocabulary.Vocabulary()
    c = Counter()
    c.update(omr_version + correct_version)
    v.update(c, min_freq=1)

    omr_vec = list(v.words_to_vec(omr_version))
    correct_vec = list(v.words_to_vec(correct_version))

    correct_align, error_align, r, score = align.perform_alignment(
        omr_vec,
        correct_vec,
        match_weights=[2, -2],
        gap_penalties=[-1, -1, -1, -1]
        )

    num_operations = sum(np.array(r) != 'O')
    error_rate = num_operations / len(correct_vec)

    omr_align_tokens = v.vec_to_words(error_align)
    error_align_tokens = v.vec_to_words(correct_align)

    align_viz = list(zip(r, omr_align_tokens, error_align_tokens))
    
    error_rates.append((error_rate, [num_operations, len(correct_vec)]))
    res.append((correct_align, error_align, r, score, align_viz, v, c))
    

