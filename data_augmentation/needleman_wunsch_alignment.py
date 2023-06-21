from typing import Match, Optional, Union
import numpy as np
from functools import partial
from numba import njit
from numba.typed import List as numbaList

default_match_weights = [2, -1]
default_gap_penalties = [-2, -2, -1, -1]


@njit
def create_matrix(
    seq_a: numbaList[int],
    seq_b: numbaList[int],
    match_weights: numbaList[int],
    gap_rules: numbaList[int],
    bands: float,
) -> tuple[np.array, np.array, np.array, int]:

    gap_open_x, gap_open_y, gap_extend_x, gap_extend_y = gap_rules

    # y_mat and x_mat keep track of gaps in horizontal and vertical directions
    len_a = len(seq_a)
    len_b = len(seq_b)

    mat = np.zeros((len_a, len_b), dtype=np.int32)
    y_mat = np.zeros((len_a, len_b), dtype=np.int32)
    x_mat = np.zeros((len_a, len_b), dtype=np.int32)
    mat_ptr = np.zeros((len_a, len_b), dtype=np.int8)
    y_mat_ptr = np.zeros((len_a, len_b), dtype=np.int8)
    x_mat_ptr = np.zeros((len_a, len_b), dtype=np.int8)

    # establish boundary conditions
    for i in range(len_a):
        mat[i][0] = gap_extend_x * i
        x_mat[i][0] = -np.inf
        y_mat[i][0] = gap_extend_x * i
    for j in range(len_b):
        mat[0][j] = gap_extend_y * j
        x_mat[0][j] = gap_extend_y * j
        y_mat[0][j] = -np.inf

    biggest_seq_length = max(len_a, len_b)

    for i in range(1, len_a):
        # how far along are we in dimension 1?
        proportion = float(i) / len_a

        center = int(np.round(proportion * len_b))
        extend_length = int(np.ceil(biggest_seq_length * bands * 0.5))

        left_b = max(1, center - extend_length)
        right_b = min(len_b, center + extend_length)

        for j in range(left_b, right_b):

            # if np.all(seq_a[i-1] == seq_b[j-1]):
            if seq_a[i - 1] == seq_b[j - 1]:
                match_score = match_weights[0]
            else:
                # match_score = np.sum(seq_a[i-1] != seq_b[j-1]) * match_weights[1]
                match_score = match_weights[1]

            mat_vals = np.array(
                [mat[i - 1][j - 1], x_mat[i - 1][j - 1], y_mat[i - 1][j - 1]]
            )
            mat[i][j] = np.max(mat_vals) + match_score
            mat_ptr[i][j] = np.argmax(mat_vals)

            # update matrix for y gaps
            y_mat_vals = np.array(
                [
                    mat[i][j - 1] + gap_open_y + gap_extend_y,
                    x_mat[i][j - 1] + gap_open_y + gap_extend_y,
                    y_mat[i][j - 1] + gap_extend_y,
                ]
            )

            y_mat[i][j] = np.max(y_mat_vals)
            y_mat_ptr[i][j] = np.argmax(y_mat_vals)

            # update matrix for x gaps
            x_mat_vals = np.array(
                [
                    mat[i - 1][j] + gap_open_x + gap_extend_x,
                    x_mat[i - 1][j] + gap_extend_x,
                    y_mat[i - 1][j] + gap_open_x + gap_extend_x,
                ]
            )

            x_mat[i][j] = np.max(x_mat_vals)
            x_mat_ptr[i][j] = np.argmax(x_mat_vals)

    return mat_ptr, x_mat_ptr, y_mat_ptr, mat[-1][-1]


def perform_alignment(
    transcript: list[int],
    ocr: list[int],
    match_weights: Optional[tuple[int, int]] = None,
    gap_penalties: Union[tuple[int, int], tuple[int, int, int, int], None] = None,
    bands: Optional[float] = None,
    verbose: bool = False,
) -> tuple[list[Union[int, str]], list[Union[int, str]], list[str], list[tuple], int]:
    """
    @match_function must be a function that takes in two strings and returns a single integer:
        a positive integer for a "match," a negative integer for a "mismatch."

    @scoring_system must be array-like, of one of the following forms:
        [gap_open_x, gap_open_y, gap_extend_x, gap_extend_y]
        [gap_open, gap_extend]

    @ignore_case ensures that the default scoring method will treat uppercase and lowercase letters
        the same, by applying lower() before every comparison. this setting will be ignored if a
        callable function is passed into match_function.
    """

    if gap_penalties is None:
        gap_open_x, gap_open_y, gap_extend_x, gap_extend_y = default_gap_penalties
    elif len(gap_penalties) == 4:
        gap_open_x, gap_open_y, gap_extend_x, gap_extend_y = gap_penalties
    elif len(gap_penalties) == 2:
        gap_open_x, gap_open_y = (gap_penalties[0], gap_penalties[0])
        gap_extend_x, gap_extend_y = (gap_penalties[1], gap_penalties[1])
    else:
        raise ValueError(
            "gap_penalties argument {} invalid: must be a list of either 4 or 2 elements".format(
                gap_penalties
            )
        )

    if not len(transcript) > 0:
        print(ocr, transcript)
        raise ValueError("Transcript must have more than 0 elements.")
    if not len(ocr) > 0:
        print(ocr, transcript)
        raise ValueError("OCR sequence must have more than 0 elements.")

    transcript = transcript + [transcript[-1]]
    ocr = ocr + [ocr[-1]]

    # handle bands argument: if false, it's 1 - use all bands
    if not bands:
        bands_amt = 1
    elif type(bands) is float:
        bands_amt = bands
    else:
        raise ValueError(f"incorrect value for bands: {bands}.")

    if verbose:
        print(
            f"building matrix with seq. a of length {len(transcript)}, seq. b of length {len(ocr)} \n"
            f"match weights: {match_weights}, gap penalties: {gap_penalties}, bands amt: {bands_amt}"
        )

    # changing everything to numba's typed list, since python untyped lists will be deprecated
    transcript_numba = numbaList(transcript)
    ocr_numba = numbaList(ocr)
    match_weights = numbaList(match_weights)
    gap_penalties = numbaList(gap_penalties)
    mat_ptr, x_mat_ptr, y_mat_ptr, total_score = create_matrix(
        transcript_numba, ocr_numba, match_weights, gap_penalties, bands
    )

    # TRACEBACK
    # which matrix we're in tells us which direction to head back (diagonally, y, or x)
    # value of that matrix tells us which matrix to go to (mat, y_mat, or x_mat)
    # mat of 0 = match, 1 = x gap, 2 = y gap
    #
    # first
    tra_align = []
    ocr_align = []
    align_record = []
    pt_record = []
    xpt = len(transcript) - 1
    ypt = len(ocr) - 1
    mpt = mat_ptr[xpt][ypt]

    # start it off. we are forcibly aligning the final characters.
    tra_align += [transcript[xpt]]
    ocr_align += [ocr[ypt]]
    align_record += ["O"] if np.all(transcript[xpt] == ocr[ypt]) else ["~"]

    # start at bottom-right corner and work way up to top-left
    while xpt > 0 and ypt > 0:

        pt_record.append((xpt, ypt, mpt))

        # case if the current cell is reachable from the diagonal
        if mpt == 0:
            tra_align.append(transcript[xpt - 1])
            ocr_align.append(ocr[ypt - 1])
            # added_text = str(transcript[xpt - 1]) + ' ' + str(ocr[ypt - 1])

            # determine if this diagonal step was a match or a mismatch
            align_record.append(
                "O" if np.all(transcript[xpt - 1] == ocr[ypt - 1]) else "~"
            )

            mpt = mat_ptr[xpt][ypt]
            xpt -= 1
            ypt -= 1

        # case if current cell is reachable horizontally
        elif mpt == 1:
            tra_align.append(transcript[xpt - 1])
            ocr_align.append("_")

            align_record.append("-")
            mpt = x_mat_ptr[xpt][ypt]
            xpt -= 1

        # case if current cell is reachable vertically
        elif mpt == 2:
            tra_align.append("_")
            ocr_align.append(ocr[ypt - 1])
            # added_text = '_ ' + str(ocr[ypt - 1])

            align_record.append("+")
            mpt = y_mat_ptr[xpt][ypt]
            ypt -= 1

        # for debugging
        # print('mpt: {} xpt: {} ypt: {} added_text: [{}]'.format(mpt, xpt, ypt, added_text))

    # we want to have ended on the very top-left cell (xpt == 0, ypt == 0). if this is not so
    # we need to add the remaining terms from the incomplete sequence.

    while ypt > 0:
        tra_align.append("_")
        ocr_align.append(ocr[ypt - 1])
        align_record.append("-")
        ypt -= 1

    while xpt > 0:
        ocr_align.append("_")
        tra_align.append(transcript[xpt - 1])
        align_record.append("+")
        xpt -= 1

    # reverse all records, since we obtained them by traversing the matrices from the bottom-right
    tra_align = tra_align[-1:0:-1]
    ocr_align = ocr_align[-1:0:-1]
    align_record = align_record[-1:0:-1]
    pt_record = pt_record[-1:0:-1]

    if verbose:
        for n in range(len(tra_align)):
            line = "{} {} {}"
            print(line.format(tra_align[n], ocr_align[n], align_record[n]))

    return tra_align, ocr_align, align_record, pt_record, total_score


if __name__ == "__main__":

    seq1 = "The Needleman–Wunsch algorithm is an algorithm used in bioinformatics to align protein or nucleotides sequences. It was an early application of dynamic programming to compare biological sequences. The algorithm was developed by Saul B. Needleman and Christian D. Wunsch and published in 1970."
    seq2 = "The Needleman–Wunsch algorithm is an algorithm used to align protein or nucleotide sequences. It was one of the first applications of dynamic programming to the comparison of biological sequences. The algorithm was developed by Saul B. Needleman and Christian D. Wunsch and published in 1970."
    match_weights = [4, -2]
    gap_penalties = [-4, -4, -1, -1]

    seq1 = list([ord(x) for x in seq1])
    seq2 = list([ord(x) for x in seq2])

    a, b, align_record, pt, score = perform_alignment(
        seq1, seq2, match_weights, gap_penalties, bands=0.7, verbose=True
    )

    sa = ""
    sb = ""

    for n in range(len(a)):
        # spacing = str(max(len(a[n]), len(b[n])))
        spacing = "1"
        sa += ("{:" + spacing + "s}").format(chr(a[n]) if type(a[n]) is int else "_")
        sb += ("{:" + spacing + "s}").format(chr(b[n]) if type(b[n]) is int else "_")

    print(sa)
    print(sb)

    # changing everything to numba's typed list, since python untyped lists will be deprecated

    # i = 3
    # print(correct_fnames[i], correct_dset[i].shape, error_dset[i].shape)
    # a, b, align_record, pt, score = perform_alignment(
    #     list(correct_dset[i]),
    #     list(error_dset[i]),
    #     match_weights,
    #     gap_penalties,
    #     bands=0.2,
    #     verbose=False,
    # )
    # print("".join(align_record))

    # transcript_numba = numbaList(list(correct_dset[i]))
    # ocr_numba = numbaList((error_dset[i]))
    # match_weights = numbaList(match_weights)
    # gap_penalties = numbaList(gap_penalties)
    # mat_ptr, x_mat_ptr, y_mat_ptr, mat = create_matrix(
    #     transcript_numba, ocr_numba, match_weights, gap_penalties, bands=0.05
    # )

    # pt = np.concatenate([np.array(pt), [pt[-1]]])
    # lines = []
    # for j, r in enumerate(align_record):
    #     entry_a = "_" if a[j] == "_" else v.vtw[a[j]]
    #     entry_b = "_" if b[j] == "_" else v.vtw[b[j]]
    #     line = f"{entry_a:<35} {entry_b:<35} {r} {pt[j][0]} {pt[j][1]} {pt[j][2]}\n"
    #     lines.append(line)

    # with open(f"test_bands_1_piece{i}.txt", "w") as f:
    #     f.writelines(lines)

    # plt.plot(pt[:, 0], pt[:, 1])
    # plt.plot(np.arange(0, np.max(pt)))
    # plt.show()
    # added_text = str(transcript[xpt - 1]) + ' _'
