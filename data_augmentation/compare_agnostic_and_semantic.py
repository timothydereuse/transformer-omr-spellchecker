import h5py
import numpy as np
import data_augmentation.needleman_wunsch_alignment as align
import data_augmentation.error_gen_logistic_regression as elgr
import data_management.semantic_to_notetuple as stn
import data_management.semantic_to_agnostic as sta
import music21 as m21
from collections import Counter
import data_management.vocabulary as vocabulary

# omr_file = r"C:\Users\tim\Documents\datasets\just_quartets\felix_omr\FP_op81_1_omr.musicxml"
omr_file = r"C:\Users\tim\Documents\datasets\solo_piano\gstring_snippet2.mxl"
# correct_file = r"C:\Users\tim\Documents\datasets\just_quartets\felix_onepass\FP_op81_1_corrected.musicxml"
correct_file = r"C:\Users\tim\Documents\datasets\solo_piano\gstring_snippet2_errors.mxl"

omr_parsed = m21.converter.parse(omr_file).getElementsByClass(m21.stream.Part)
correct_parsed = m21.converter.parse(correct_file).getElementsByClass(m21.stream.Part)

def get_reps(m21_stream):
    reps = [
        stn.notetuple_string(stn.m21_parts_to_notetuple(m21_stream, interleaved=False)),
        stn.midilike_string(stn.m21_parts_to_MIDILike(m21_stream, interleaved=False)),
        stn.eventlike_string(stn.m21_parts_to_eventlike(m21_stream, interleaved=False)),
        [x.music_element for x in sta.m21_parts_to_interleaved_agnostic(m21_stream, interleave=False)],
    ]
    return reps

omr_groups = get_reps(omr_parsed)
correct_groups = get_reps(correct_parsed)


groups = list(zip(omr_groups, correct_groups))

res = []
error_rates = []
viz = []
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
        correct_vec,
        omr_vec,
        match_weights=[2, -2],
        gap_penalties=[-1, -1, -1, -1]
        )

    num_operations = sum(np.array(r) != 'O')
    num_replacements = sum(np.array(r) == '~')
    num_deletions = sum(np.array(r) == '-')
    num_insertions = sum(np.array(r) == '+')
    error_rate = num_operations / len(correct_vec)

    omr_align_tokens = [(x if type(x) is str else v.vtw[x]) for x in error_align]
    correct_align_tokens = [(x if type(x) is str else v.vtw[x]) for x in correct_align]

    align_viz = list(zip(r, omr_align_tokens, correct_align_tokens))
    
    error_rates.append({
        'num_operations': num_operations,
        'num_replacements': num_replacements,
        'num_deletions': num_deletions,
        'num_insertions': num_insertions,
        'error_rate': error_rate,
        'insert_rate': num_insertions / num_operations,
        'replace_rate': num_replacements / num_operations,
        'delete_rate': num_deletions / num_operations,
        'len_correct': len(correct_vec),
        'len_omr': len(omr_vec)
    })
    res.append((correct_align, error_align, r, score, v, c))
    viz.append(align_viz)

for x in viz[0]:
    print(f'{x[0]};{x[1]};{x[2]}')
    
for x in viz[0]:
    print(f'{x[1]}')

for x in error_rates:
    print('{len_correct}, {num_operations}, {error_rate}, {replace_rate}, {delete_rate}, {insert_rate}'.format(**x))