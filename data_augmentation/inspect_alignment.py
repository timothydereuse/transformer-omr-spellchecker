import csv
import music21 as m21
import data_management.semantic_to_agnostic as sta
from data_management.vocabulary import Vocabulary
from data_augmentation.error_gen_logistic_regression import ErrorGenerator

out_file = "compare_agnostics_2.csv"
correct_stream_fpath = r"C:\Users\tim\Documents\datasets\just_quartets\paired_omr_correct\correct_quartets\sq_in_E-flat_major_Op.277__Fanny_Hensel.mxl"
error_stream_fpath = r"C:\Users\tim\Documents\datasets\just_quartets\paired_omr_correct\omr_quartets\sq_in_E-flat_major_Op.277__Fanny_Hensel.musicxml"


def parse_filter_stream(fpath):
    pa = m21.converter.parse(fpath)
    for el in pa.recurse(includeSelf=True):
        if (
            type(el) == m21.expressions.TextExpression
            or type(el) == m21.text.TextBox
            or type(el) == m21.spanner.Slur
            or isinstance(el, m21.instrument.Instrument)
        ):
            el.activeSite.remove(el)
        if hasattr(el, "lyric"):
            el.lyric = None
        if type(el) == m21.stream.base.Part:
            el.partName = None
            el.partAbbreviation = None
            el.coreElementsChanged()
    pa.coreElementsChanged()
    return pa


error_generator = ErrorGenerator(
    error_models_fpath=r"processed_datasets\paired_quartets_error_model_bymeasure_restate.joblib"
)
v = Vocabulary(load_from_file=r"processed_datasets\vocab_big.txt")


correct_stream = parse_filter_stream(correct_stream_fpath)
errored_stream = parse_filter_stream(error_stream_fpath)

agnostic_rec_correct = sta.m21_streams_to_agnostic([correct_stream])[0]
agnostic_rec_errored = sta.m21_streams_to_agnostic([errored_stream])[0]
vectorized_errored = v.words_to_vec([x.music_element for x in agnostic_rec_errored])
vectorized_correct = v.words_to_vec([x.music_element for x in agnostic_rec_correct])

err_resid, targets = error_generator.add_errors_to_seq(
    vectorized_correct, vectorized_errored, bands=0.15
)
targets = targets.astype("bool")

rows = []
out_string = ""
for i in range(300):
    cor = []
    measure_entries = [x for x in agnostic_rec_correct if x.measure_idx == i]
    for el in measure_entries:
        cor.append([el.music_element, el.measure_idx, el.event_idx])
    err = []
    measure_entries = [
        (j, x) for j, x in enumerate(agnostic_rec_errored) if x.measure_idx == i
    ]
    for j, el in measure_entries:
        err.append([el.music_element, el.measure_idx, el.event_idx, targets[j]])
    if len(err) < len(cor):
        for _ in range(len(cor) - len(err)):
            err.append(["-", "-", "-", "-"])
    elif len(cor) < len(err):
        for _ in range(len(err) - len(cor)):
            cor.append(["-", "-", "-"])
    with open(out_file, "a", newline="") as csvfile:
        wr = csv.writer(csvfile, delimiter=",")
        wr.writerow([f"measure {i}", "-", "-", "-", "-", "-", "-"])
        for c, e in zip(cor, err):
            wr.writerow(c + e)
