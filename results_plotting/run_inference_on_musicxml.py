import torch
import training_helper_functions as tr_funcs
import data_management.semantic_to_agnostic as sta
from model_setup import PreparedLSTUTModel
import numpy as np
import music21 as m21
import model_params


def assign_color_to_stream(this_stream, agnostic_rec, predictions, color_style="red"):
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
        else:
            selected_element.style.color = color_style

    return this_stream


def run_agnostic_through_model(agnostic_rec, model, seq_length, vocab):
    model.eval()

    vectorized = vocab.words_to_vec(agnostic_rec).astype("long")
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


def run_inference_and_color_streams(
    errored_streams,
    model,
    v,
    threshold,
    correct_streams=None,
    colors=None,
    error_generator=None,
):

    if not colors:
        true_pos_color, false_pos_color, false_neg_color = ("red", "blue", "gray")
    else:
        true_pos_color, false_pos_color, false_neg_color = colors

    if not correct_streams:
        ground_truth_mode = False
    elif len(correct_streams) != len(errored_streams):
        raise ValueError(
            "Must have the same number of correct and corresponding errored files"
        )
    elif not error_generator:
        raise ValueError(
            "Must supply error_generator object when running in ground truth mode"
        )
    else:
        ground_truth_mode = True
        agnostic_records_correct = sta.m21_streams_to_agnostic(correct_streams)

    agnostic_records_errored = sta.m21_streams_to_agnostic(errored_streams)

    output_streams = []

    for i in range(len(errored_streams)):

        print(f"processing stream {i} of {len(errored_streams)}...")
        agnostic_rec = agnostic_records_errored[i]

        this_stream = parsed_errored[i]
        predictions = run_agnostic_through_model(
            agnostic_rec, model, params.seq_length, v
        )

        # threshold predictions of model
        thresh_pred = (predictions > threshold).numpy().astype("bool")

        if not ground_truth_mode:
            colored_stream = assign_color_to_stream(
                this_stream, agnostic_rec, thresh_pred, color_style=true_pos_color
            )
        else:
            # process further if we need to compare with ground truth

            vectorized_errored = v.words_to_vec(
                [x.music_element for x in agnostic_records_errored[i]]
            )
            vectorized_correct = v.words_to_vec(
                [x.music_element for x in agnostic_records_correct[i]]
            )

            # get targets (ground truth)
            _, targets = error_generator.add_errors_to_seq(
                vectorized_errored, vectorized_correct
            )
            targets = targets.astype("bool")

            true_positive = np.logical_and(targets, thresh_pred)
            false_positive = np.logical_and(targets, np.logical_not(thresh_pred))
            false_negative = np.logical_and(np.logical_not(targets), thresh_pred)

            colored_stream = assign_color_to_stream(
                this_stream, agnostic_rec, false_negative, color_style=false_neg_color
            )
            colored_stream = assign_color_to_stream(
                colored_stream,
                agnostic_rec,
                false_positive,
                color_style=false_pos_color,
            )
            colored_stream = assign_color_to_stream(
                colored_stream, agnostic_rec, true_positive, color_style=true_pos_color
            )

        output_streams.append(colored_stream)
    return output_streams


if __name__ == "__main__":
    model_path = r"C:\Users\tim\Documents\transformer-midi-error-correction\trained_models\lstut_best_lstut_seq_sweep_cyclicLR_1_(2023.07.09.09.19)_lstm256-1-tf212-4-64-2048.pt"
    saved_model_info = torch.load(model_path, map_location=torch.device("cpu"))

    threshold = saved_model_info["val_threshes"][1]

    params = model_params.Params("./param_sets/node_lstut.json", False, 1)
    device, num_gpus = tr_funcs.get_cuda_info()

    prep_model = PreparedLSTUTModel(params, saved_model_info["model_state_dict"])
    groups = tr_funcs.make_test_dataloaders(params, prep_model.dset_kwargs)

    errored_files = [
        r"C:\Users\tim\Documents\datasets\just_quartets\paired_omr_correct\omr_quartets\mendelssohn_1_op12_1.musicxml",
    ]

    correct_files = [
        r"C:\Users\tim\Documents\datasets\just_quartets\paired_omr_correct\correct_quartets\mendelssohn_1_op12_1.musicxml"
    ]

    parsed_correct = [m21.converter.parse(fpath) for fpath in correct_files]
    parsed_errored = [m21.converter.parse(fpath) for fpath in errored_files]

    out2 = run_inference_and_color_streams(
        parsed_errored, prep_model.model, prep_model.v, threshold
    )
    out1 = run_inference_and_color_streams(
        parsed_errored,
        prep_model.model,
        prep_model.v,
        threshold,
        correct_streams=parsed_correct,
        colors=None,
        error_generator=prep_model.error_generator,
    )

    out1[0].write("musicxml", fp="./test2.musicxml")
