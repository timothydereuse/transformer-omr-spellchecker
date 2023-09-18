import torch
import training_helper_functions as tr_funcs
import data_management.semantic_to_agnostic as sta
from model_setup import PreparedLSTUTModel
import numpy as np
import music21 as m21
import model_params
import verovio
import os
import pickle

verovio_options = {
    "pageHeight": 1000,
    "pageWidth": 1750,
    "scale": 25,
    "header": "none",
    "footer": "none",
    "outputIndentTab": 0,
    "pageMarginBottom": 0,
    "pageMarginLeft": 25,
    "pageMarginRight": 0,
    "pageMarginTop": 0,
    "font": "Leland",
    "evenNoteSpacing": False,
    "systemMaxPerPage": 1,
    "spacingLinear": 0.25,
    "spacingNonLinear": 0.57,
    "adjustPageHeight": True,
    # "adjustPageWidth": True,
}


def get_all_sub_files_with_ext(output_folder, ext):
    path_pairs = []
    for folder_name in os.listdir(output_folder):
        if not os.path.isdir(os.path.join(output_folder, folder_name)):
            continue
        folder_path = os.path.join(output_folder, folder_name)
        file_names = os.listdir(folder_path)
        xml_files = [x for x in file_names if ext in x]
        if len(xml_files) == 0:
            continue
        path_pairs.append((folder_path, os.path.join(folder_path, xml_files[0])))
    return path_pairs


def assign_color_to_stream(this_stream, agnostic_rec, predictions, color_style):
    # given a music21 stream, a list of agnostic tokens, a list of predictions
    # on those tokens, and a color, assign that color to all objects in the
    # music21 stream where the prediction on its associated token is true.

    parts = list(this_stream.getElementsByClass(m21.stream.Part))
    all_measures = [list(x.getElementsByClass(m21.stream.Measure)) for x in parts]
    num_tokens = len(agnostic_rec)

    # for every note predicted incorrect, find the element of the m21 stream
    # that corresponds to it
    for record, prediction in zip(agnostic_rec, predictions):

        # change nothing if this token is predicted correct
        if not prediction:
            continue

        # print(
        #     record.part_idx,
        #     len(all_measures),
        #     record.measure_idx,
        #     len(all_measures[record.part_idx]),
        #     record.event_idx,
        #     len(all_measures[record.part_idx][record.measure_idx]),
        # )

        # get the single m21 element referred to by this token
        try:
            selected_element = all_measures[record.part_idx][record.measure_idx][
                record.event_idx
            ]
        except IndexError:
            continue
        token_type = record.music_element.split(".")[0]

        # how do we handle chords? if it's a chord, then record.chord_idx will be non-zero:
        # choose the note in that chord corresponding to the relevant chord idx
        if type(selected_element) == m21.chord.Chord:
            selected_element = selected_element.notes[record.chord_idx]

        # now for a bunch of if statements to handle each individual type of token
        # in its own way, because i don't know what else to do
        if type(selected_element) == m21.layout.SystemLayout:
            # if the token of interest refers to a layout element in m21, then that
            # token is part of a time signature, a courtesy clef, or the restatement
            # of a key signature. None of these things can be easily marked individually
            # in MusicXML or Music21, so we ignore them.
            continue
        elif (
            token_type == "accid"
            and type(selected_element) == m21.note.Note
            and selected_element.pitch.accidental
        ):
            selected_element.pitch.accidental.style.color = color_style
        elif token_type == "articulation" and type(selected_element) == m21.note.Note:
            for articulation in selected_element.articulations:
                articulation.style.color = color_style
        elif token_type == "rest" and type(selected_element) == m21.note.Rest:
            selected_element.style.color = color_style
        elif token_type == "note" and type(selected_element) == m21.note.Note:
            selected_element.style.color = color_style
        elif isinstance(selected_element, m21.clef.Clef):
            pass
        else:
            selected_element.style.color = color_style

    return this_stream


def run_agnostic_through_model(agnostic_rec, model, seq_length, vocab):
    # model.eval()
    recs = [x.music_element for x in agnostic_rec]
    vectorized = vocab.words_to_vec(recs).astype("long")
    inp = torch.tensor(vectorized)

    # expand input vector to be a multiple of the sequence length
    # and then reshape into correct form for prediction
    expand_amt = seq_length - (len(inp) % seq_length)
    expand_vec = torch.full((expand_amt,), vocab.SEQ_PAD)
    inp = torch.cat([inp, expand_vec])
    inp = inp.reshape(-1, seq_length)

    with torch.no_grad():
        pred = model(inp)

    # unwrap prediction
    unwrapped_pred = pred.reshape(-1)[:-expand_amt]

    return unwrapped_pred


def run_inference_and_align(
    errored_stream,
    model,
    v,
    sequence_length,
    correct_stream=None,
    error_generator=None,
):

    print("    converting stream to agnostic...")
    if not correct_stream:
        ground_truth_mode = False
    elif not error_generator:
        raise ValueError(
            "Must supply error_generator object when running in ground truth mode"
        )
    else:
        ground_truth_mode = True
        agnostic_rec_correct = sta.m21_streams_to_agnostic([correct_stream])[0]

    agnostic_rec_errored = sta.m21_streams_to_agnostic([errored_stream])[0]

    print("    running stream through model")
    predictions = run_agnostic_through_model(
        agnostic_rec_errored, model, sequence_length, v
    )

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    # threshold predictions of model
    sig_outputs = sigmoid(predictions)

    if not ground_truth_mode:
        return agnostic_rec_errored, sig_outputs, None

    # process further if we need to compare with ground truth
    print("    aligning with ground truth...")

    vectorized_errored = v.words_to_vec([x.music_element for x in agnostic_rec_errored])
    vectorized_correct = v.words_to_vec([x.music_element for x in agnostic_rec_correct])

    # get targets (ground truth)
    _, targets = error_generator.add_errors_to_seq(
        vectorized_correct,
        given_err_seq=vectorized_errored,
        bands=0.15,
        match_original_dim=False,
    )
    targets = targets.astype("bool")

    return agnostic_rec_errored, sig_outputs, targets


def evaluate_and_color_stream(
    errored_stream,
    agnostic_rec_errored,
    sig_outputs,
    threshold,
    targets=None,
    colors=None,
):

    if not colors:
        true_pos_color, false_pos_color, false_neg_color = (
            "#EAB704",
            "#D20712",
            "#1A63EA",
        )
    else:
        true_pos_color, false_pos_color, false_neg_color = colors

    if targets is None:
        thresh_pred = (sig_outputs > threshold).numpy().astype("bool")
        colored_stream = assign_color_to_stream(
            errored_stream,
            agnostic_rec_errored,
            thresh_pred,
            color_style=false_pos_color,
        )
    else:
        thresh_pred = (sig_outputs > threshold).numpy().astype("bool")
        true_positive = np.logical_and(targets, thresh_pred)
        false_positive = np.logical_and(np.logical_not(targets), thresh_pred)
        false_negative = np.logical_and(targets, np.logical_not(thresh_pred))

        colored_stream = assign_color_to_stream(
            errored_stream,
            agnostic_rec_errored,
            false_negative,
            color_style=false_neg_color,
        )
        colored_stream = assign_color_to_stream(
            colored_stream,
            agnostic_rec_errored,
            false_positive,
            color_style=false_pos_color,
        )
        colored_stream = assign_color_to_stream(
            colored_stream,
            agnostic_rec_errored,
            true_positive,
            color_style=true_pos_color,
        )

    return colored_stream


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


def get_paired_score_fnames(paired_quartets_root):
    # getting quartet paths and pairing them up
    correct_quart_path = os.path.join(paired_quartets_root, "correct_quartets")
    omr_quart_path = os.path.join(paired_quartets_root, "omr_quartets")
    score_fnames = os.listdir(correct_quart_path)
    omr_fnames = os.listdir(omr_quart_path)
    cor = set(["".join(x.split(".")[:-1]) for x in score_fnames])
    err = set(["".join(x.split(".")[:-1]) for x in omr_fnames])
    assert not (cor.difference(err)), "filenames not synced up cor -> err"
    assert not (err.difference(cor)), "filenames not synced up err -> cor"

    paired_score_fnames = zip(
        sorted(score_fnames, reverse=True), sorted(omr_fnames, reverse=True)
    )
    paired_score_fnames = list(paired_score_fnames)
    return paired_score_fnames


def end_to_end_inference(input_file, output_folder, prep_model, params, threshold, tk):

    print(f"parsing f{input_file}...")
    parsed_errored = parse_filter_stream(input_file)

    print(f"    running through model...")
    agnostic_rec_errored, sig_outputs, _ = run_inference_and_align(
        parsed_errored,
        prep_model.model,
        prep_model.v,
        sequence_length=params.seq_length,
    )

    print(f"    coloring score model...")
    colored_score = evaluate_and_color_stream(
        parsed_errored,
        agnostic_rec_errored,
        sig_outputs,
        threshold,
    )

    xml_fp = os.path.join(output_folder, "colored_score.musicxml")
    colored_score.write("musicxml", fp=xml_fp)

    print(f"rendering svg to {xml_fp}...")

    tk.loadFile(xml_fp)
    tk.redoLayout()
    pages = tk.getPageCount()

    for i in range(1, pages + 1):
        page_path = os.path.join(output_folder, f"page_{i}.svg")
        tk.renderToSVGFile(page_path, i)


if __name__ == "__main__":

    recompute_inference = False
    recompute_coloring = False
    model_path = r"trained_models\lstut_best_lstut_seqlen_4_(2023.09.08.17.53)_lstm512-1-tf112-6-64-2048.pt"
    paired_quartets_root = (
        r"C:\Users\tim\Documents\datasets\just_quartets\paired_omr_correct"
    )
    output_folder = r"C:\Users\tim\Documents\datasets\quartets_results_svg"

    # verovio setup
    tk = verovio.toolkit()
    tk.setOptions(verovio_options)
    verovio.enableLog(verovio.LOG_ERROR)

    # basic model setup
    saved_model_info = torch.load(model_path, map_location=torch.device("cpu"))
    threshold = saved_model_info["val_threshes"][0]
    params = model_params.Params("./param_sets/node_lstut.json", False, 4)
    prep_model = PreparedLSTUTModel(params, saved_model_info["model_state_dict"])

    fpath = r"C:\Users\tim\Documents\datasets\just_quartets\paired_omr_correct\correct_quartets\mendelssohn_5_op44iii_3.musicxml"
    outfile = r"C:\Users\tim\Documents\tex\dissertation\results\correctfelix"
    end_to_end_inference(fpath, outfile, prep_model, params, threshold, tk)
    assert False

    paired_score_fnames = get_paired_score_fnames(paired_quartets_root)
    # paired_score_fnames = paired_score_fnames[4:6]

    for cor_score, omr_score in paired_score_fnames:

        score_name = "".join(cor_score.split(".")[:-1])
        score_folder_path = os.path.join(output_folder, score_name)
        pkl_fp = os.path.join(score_folder_path, "inference_results.pkl")
        xml_fp = os.path.join(score_folder_path, "colored_score.musicxml")

        if not os.path.exists(score_folder_path):
            os.makedirs(score_folder_path)

        if (
            os.path.isfile(pkl_fp)
            and os.path.isfile(xml_fp)
            and not (recompute_coloring or recompute_inference)
        ):
            continue

        print(f"parsing {cor_score}...")
        cor_path = os.path.join(paired_quartets_root, "correct_quartets", cor_score)
        parsed_correct = parse_filter_stream(cor_path)
        err_fpath = os.path.join(paired_quartets_root, "omr_quartets", omr_score)
        parsed_errored = parse_filter_stream(err_fpath)

        if os.path.isfile(pkl_fp) or recompute_inference:
            print(f"    using precomputed results at {pkl_fp}")
            with open(pkl_fp, "rb") as f:
                res = pickle.load(f)
                agnostic_rec_errored, sig_outputs, targets = res
        else:
            agnostic_rec_errored, sig_outputs, targets = run_inference_and_align(
                parsed_errored,
                prep_model.model,
                prep_model.v,
                sequence_length=params.seq_length,
                correct_stream=parsed_correct,
                error_generator=prep_model.error_generator,
            )
            with open(pkl_fp, "wb") as f:
                pickle.dump((agnostic_rec_errored, sig_outputs, targets), f)

        if os.path.isfile(xml_fp) and not recompute_coloring:
            print(f"    using precomputed colored score at {xml_fp}")
            continue

        colored_score = evaluate_and_color_stream(
            parsed_errored,
            agnostic_rec_errored,
            sig_outputs,
            threshold,
            targets=targets,
        )

        try:
            colored_score.write("musicxml", fp=xml_fp)
        except:
            print(f"failed to export {xml_fp}, skipping")
            continue

    xml_paths = get_all_sub_files_with_ext(output_folder, "xml")

    for folder_path, xml_path in xml_paths[::-1]:
        print(f"rendering svg to {xml_path}...")

        tk.loadFile(xml_path)
        tk.redoLayout()
        pages = tk.getPageCount()

        for i in range(1, pages + 1):
            page_path = os.path.join(folder_path, f"page_{i}.svg")
            tk.renderToSVGFile(page_path, i)


assert False
cor_score, omr_score = paired_score_fnames[0]

print(f"parsing {cor_score}...")
cor_path = os.path.join(paired_quartets_root, "correct_quartets", cor_score)
correct_stream = parse_filter_stream(cor_path)
err_fpath = os.path.join(paired_quartets_root, "omr_quartets", omr_score)
errored_stream = parse_filter_stream(err_fpath)

v = prep_model.v
error_generator = prep_model.error_generator

agnostic_rec_correct = sta.m21_streams_to_agnostic([correct_stream])[0]
agnostic_rec_errored = sta.m21_streams_to_agnostic([errored_stream])[0]
vectorized_errored = v.words_to_vec([x.music_element for x in agnostic_rec_errored])
vectorized_correct = v.words_to_vec([x.music_element for x in agnostic_rec_correct])

# get targets (ground truth)
err_resid, targets = error_generator.add_errors_to_seq(
    vectorized_correct, vectorized_errored, bands=0.15
)
targets = targets.astype("bool")


import csv

# rows = []
# out_string = ""
# for i in range(300):
#     cor = []
#     measure_entries = [x for x in agnostic_rec_correct if x.measure_idx == i]
#     for el in measure_entries:
#         cor.append([el.music_element, el.measure_idx, el.event_idx])
#     err = []
#     measure_entries = [
#         (j, x) for j, x in enumerate(agnostic_rec_errored) if x.measure_idx == i
#     ]
#     for j, el in measure_entries:
#         err.append([el.music_element, el.measure_idx, el.event_idx, targets[j]])
#     if len(err) < len(cor):
#         for _ in range(len(cor) - len(err)):
#             err.append(["-", "-", "-", "-"])
#     elif len(cor) < len(err):
#         for _ in range(len(err) - len(cor)):
#             cor.append(["-", "-", "-"])
#     with open("compare_agnostic2.csv", "a", newline="") as csvfile:
#         wr = csv.writer(csvfile, delimiter=",")
#         wr.writerow([f"measure {i}", "-", "-", "-", "-", "-", "-"])
#         for c, e in zip(cor, err):
#             wr.writerow(c + e)
