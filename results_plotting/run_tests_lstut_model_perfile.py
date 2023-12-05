import training_helper_functions as tr_funcs
from model_setup import PreparedLSTUTModel
from data_management.vocabulary import Vocabulary
import model_params
import torch
import csv
import numpy as np


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, inp):
        super(DummyDataset).__init__()
        self.data = inp

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, i):
        return [[self.data[0][i][0], self.data[0][i][1]], self.data[1][i]]


if __name__ == "__main__":
    model_path = r"trained_models\lstut_best_lstut_seqlen_4_(2023.09.08.17.53)_lstm512-1-tf112-6-64-2048.pt"
    saved_model_info = torch.load(model_path, map_location=torch.device("cpu"))

    params = model_params.Params("./param_sets/node_lstut.json", False, 4)
    device, num_gpus = tr_funcs.get_cuda_info()

    params.test_sets = [{"base": "test/omr", "real_data": True}]

    prep_model = PreparedLSTUTModel(params, saved_model_info["model_state_dict"])
    groups = tr_funcs.make_test_dataloaders(params, prep_model.dset_kwargs)
    g = groups[0]  # get OMR group

    v = Vocabulary(load_from_file=r"processed_datasets\vocab_big.txt")

    stats_lines = [
        [
            "piece",
            "num_measures",
            "num tokens",
            "num errors",
            "num measures with errors",
            "std(tokens_per_measure)",
            "std errors per measure",
            "num_detections",
            "precision at 0.9",
            "avg precision",
            "norm recall",
        ]
    ]

    barlines = v.words_to_vec(
        [
            "barline.regular",
            "barline.double",
            "barline.final",
            "barline.heavy-heavy",
            "barline.heavy-light",
            "barline.dotted",
        ]
    )

    for x in g.dset.iter_file():

        file_dset = DummyDataset(x)

        batch_np = np.concatenate(x[0][:, 0, :].numpy())
        targets_np = np.concatenate(x[0][:, 1, :].numpy())

        num_pad_tokens = (batch_np == 1).sum()
        measures_locations = np.argwhere(np.isin(batch_np, barlines)).ravel()
        num_measures = measures_locations.shape[0]
        print(num_measures)

        dloader = torch.utils.data.DataLoader(file_dset, batch_size=params.batch_size)

        res_stats, tst_exs, test_results = tr_funcs.test_end_group(
            dloader, prep_model.run_epoch_kwargs, params.target_recalls, verbose=True
        )

        res_string = tr_funcs.get_nice_results_string(g.name, res_stats)
        print(res_string)

        tokens_per_measure = np.diff(measures_locations)
        errs_in_measure = []
        for i in range(num_measures - 1):
            st = measures_locations[i]
            end = measures_locations[i + 1]
            num_errs = sum(targets_np[st:end])
            errs_in_measure.append(num_errs)

        stats_line = [
            x[1][0][2],
            num_measures,
            test_results.targets.shape[0] - num_pad_tokens,
            (test_results.targets == 1).sum(),
            sum(np.array(errs_in_measure) > 0),
            np.std(errs_in_measure),
            np.std(tokens_per_measure),
            res_stats["prop_positive_predictions"][0.9]
            * (test_results.targets.shape[0] - num_pad_tokens),
            res_stats["precision"][0.9],
            res_stats["average_precision"],
            res_stats["normalized_recall"],
        ]
        stats_lines.append(stats_line)
        print(stats_line)

    with open("results_csv/perfilecsv_test.csv", "a", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        for line in stats_lines:
            writer.writerow(line)

# groups[0].dset.seq_length = 512
# total_tokens = 0
# total_errors = 0
# total_correct = 0
# for x in groups[0].dset:
#     total_tokens += x[0][0].shape[0]
#     total_errors += np.count_nonzero(x[0][1])
#     total_correct += np.count_nonzero(x[0][1] == 0)
# print(total_tokens, total_correct, total_errors)
