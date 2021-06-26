import h5py
import numpy as np
import data_management.needleman_wunsch_alignment as align
from numba import njit
from collections import Counter

dset_path = r'./quartets_felix_omr.h5'
# voice, onset, time_to_next_onset, duration, midi_pitch, notated_pitch, accidental
inds_subset = np.array([0, 2, 3, 4])

with h5py.File(dset_path, 'a') as f:

    correct_fnames = [x for x in f.keys() if 'aligned' in x and 'op80' not in x]
    error_fnames = [x for x in f.keys() if 'omr' in x]

    correct_dset = [f[x][:, inds_subset] for x in correct_fnames]
    error_dset = [f[x][:, inds_subset] for x in error_fnames]


error_accum = {x:[] for x in ['replace', 'replace_mod', 'delete', 'insert', 'insert_mod']}
correct_seqs_all = []


for ind in range(len(correct_dset)):
        
    print(f'aligning {correct_fnames[ind]}...' )
    correct_seq = [x for x in correct_dset[ind]]
    error_seq = [x for x in error_dset[ind]]
    correct_align, error_align, r, score = align.perform_alignment(correct_seq, error_seq, match_weights=[1, -1], gap_penalties=[-4, -4, -3, -3])

    # sa = ''
    # sb = ''
    # for n in range(len(correct_align)):
    #     spacing = str(max(len(correct_align[n]), len(error_align[n])))
    #     sa += (f'{str(correct_align[n]):4}')
    #     sb += (f'{str(error_align[n]):4}')
    # print(sa)
    # print(sb)
    print(''.join(r))

    correct_seqs_all.extend(correct_seq)
    errors = []
    current_error = []
    for i, t in enumerate(r):
        if t == 'O' and len(current_error) > 0:
            errors.append(np.array(current_error))
            current_error = []
            continue
        elif t == 'O':
            continue
        current_error.append(i)

    for e in np.array(errors):
        prev_correct_note = np.zeros(inds_subset.shape)
        for i in e:
            error_note = error_align[i]
            correct_note = correct_align[i]
            if r[i] == '~':
                res = correct_note - error_note
                error_accum['replace'].append(correct_note)
                error_accum['replace_mod'].append(res)
            elif type(error_note) == str:
                error_accum['delete'].append(correct_note)
            elif type(correct_note) == str:
                error_accum['insert'].append(prev_correct_note) # THIS LINE IS THE PROBLEM
                error_accum['insert_mod'].append(error_note)

            if not type(correct_note) == str:
                prev_correct_note = correct_note

# we want to know, given that there are (for example) N eighth notes in the corrected piece, how many of them are deleted (on average)
# by the OMR process? what is the likelihood that an eighth note will be added?
# how likely is it that a given note will be deleted?

en = [(1, 'dur'), (2, 'iot'), (3, 'pitch')]

counts = {}
counts['cor'] = {k: Counter([x[i] for x in correct_seqs_all]) for i, k in en}
counts['del'] = {k: Counter([x[i] for x in error_accum['delete']]) for i, k in en}
counts['ins'] = {k: Counter([x[i] for x in error_accum['insert']]) for i, k in en}
counts['repl'] = {k: Counter([x[i] for x in error_accum['replace']]) for i, k in en}
counts['repl_mod'] = {k: Counter([x[i] for x in error_accum['replace_mod']]) for i, k in en}
counts['ins_mod'] = {k: Counter([x[i] for x in error_accum['insert_mod']]) for i, k in en}

# if there are Np notes of pitch p, and Mp of them were deleted, then every note with pitch p has an (Mp / Np) probability of
# getting removed by the OMR on the basis of pitch alone. 
# if there are Ni notes of iot i and Mi of them were deleted, every note with iot i has a (Mi / Ni) probability of getting
# removed by the OMR on the basis of OMR alone.
# so what an we say about a note with pitch p AND iot i? well, assuming independence of these two factors (which is definitely
# not the case, but just roll with it for now) the chance of that note being selected for deletion is
# 1 - (1 - Mp/Np)(1 - Mi/Ni)
#
# so we need to calculate this (Mx/Nx) quantity for every type of operation X every feature index.

stats = {}
for op_name in counts.keys():
    if op_name == 'cor':
        continue
    stats[op_name] = {}
    for feat in counts[op_name].keys():
        stats[op_name][feat] = {}
        for item in counts['cor'][feat].keys():
            if op_name in ['ins_mod', 'repl_mod']:
                N = sum(counts[op_name][feat].values())
            else:
                N = counts['cor'][feat][item]
            M = counts[op_name][feat][item]
            stats[op_name][feat][item] = (M / N)

def probs_that_note_is_modified(note, stats):
    en = [(1, 'dur'), (2, 'iot'), (3, 'pitch')]
    ops = ['del', 'ins', 'repl']
    probs = {}
    for op in ops:
        feat_probs = []
        for f_ind, feat in en:
            feat_probs.append(stats[op][feat][note[f_ind]])
        prob = 1 - np.product([1 - x for x in feat_probs])
        probs[op] = prob
    probs['nop'] = 1 - sum(list(probs.values()))
    return probs

def generate_note(orig_note, t, stats):
    note = np.array(orig_note)
    en = [(1, 'dur'), (2, 'iot'), (3, 'pitch')]
    for f_ind, feat in en:
        d = stats[t][feat]
        p = np.array(list(d.values()))
        mod_select = np.random.choice(list(d.keys()), 1, p=p / np.sum(p))
        if t == 'ins_mod':
            note[f_ind] = mod_select
        elif t == 'repl_mod':
            note[f_ind] = mod_select + note[f_ind]
    return note

def add_errors_to_note_sequence(seq, stats):
    new_seq = np.array(seq)
    i = 0
    while i < len(new_seq):
        probs = probs_that_note_is_modified(new_seq[i], stats)
        mod_select = np.random.choice(list(probs.keys()), 1, p=list(probs.values()))
        if mod_select == 'nop':
            i += 1
            continue
        elif mod_select == 'del':
            new_seq = np.delete(new_seq, i, 0)
            continue
        elif mod_select == 'ins':
            new_note = generate_note(new_seq[i], 'ins_mod', stats)
            new_seq = np.insert(new_seq, i + 1, new_note, 0)
            i += 2
        elif mod_select == 'repl':
            new_seq[i] = generate_note(new_seq[i], 'repl_mod', stats)
            i += 1

        if i > 2 * len(seq):
            print('something has gone wrong')
            break
    return new_seq


        
            
