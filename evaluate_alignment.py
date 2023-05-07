import music21 as m21
import numpy as np
import data_management.semantic_to_agnostic as sta
from data_management.vocabulary import Vocabulary
import data_augmentation.needleman_wunsch_alignment as nwa

c = m21.converter.Converter()

orig_fname = (
    "/Users/tim/Documents/felix_quartets_got_annotated/1_op12/C0/1_op12_2_omr.musicxml"
)
onepass_fname = "/Users/tim/Documents/felix_quartets_got_annotated/1_op12/C1/1_op12_2_corrected.musicxml"
twopass_fname = "/Users/tim/Documents/felix_quartets_got_annotated/1_op12/C2/1_op12_2_revised.musicxml"
correct_fname = "/Users/tim/Documents/felix_quartets_got_annotated/1_op12/C3/1_op12_2_aligned.musicxml"

fnames = [orig_fname, onepass_fname, twopass_fname, correct_fname]

streams = [m21.converter.parse(x) for x in fnames]

parts = [
    sta.m21_parts_to_interleaved_agnostic(
        x.getElementsByClass(m21.stream.Part),
        remove=["+"],
        interleave=True,
        just_tokens=True,
    )
    for x in streams
]

v = Vocabulary(load_from_file="./data_management/vocab_big.txt")

pre = [(parts[1][i], parts[3][i]) for i in range(min(len(parts[1]), len(parts[3])))]

onepass_tokens = list(v.words_to_vec(parts[1]))
correct_tokens = list(v.words_to_vec(parts[3]))

match_weights = [3, -2]
gap_penalties = [-1, -1, -1, -1]
a, b, align_record, score = nwa.perform_alignment(
    onepass_tokens, correct_tokens, match_weights, gap_penalties, ignore_case=True
)

onepass_aligned = [x if x == "_" else v.vec_to_words([x])[0] for x in a]
correct_aligned = [x if x == "_" else v.vec_to_words([x])[0] for x in b]

res = [(onepass_aligned[i], correct_aligned[i]) for i in range(len(correct_aligned))]
