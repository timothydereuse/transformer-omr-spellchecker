import training_helper_functions as tr_funcs
from model_setup import PreparedLSTUTModel
import model_params
import torch

if __name__ == "__main__":
    model_path = r"trained_models\lstut_best_lstut_seqlen_8_(2023.09.10.03.59)_lstm512-1-tf112-6-64-2048_UT.pt"
    saved_model_info = torch.load(model_path, map_location=torch.device("cpu"))

    params = model_params.Params("./param_sets/node_lstut.json", False, 8)
    device, num_gpus = tr_funcs.get_cuda_info()

    prep_model = PreparedLSTUTModel(params, saved_model_info["model_state_dict"])
    groups = tr_funcs.make_test_dataloaders(params, prep_model.dset_kwargs)

    groups = [groups[0]]
    for g in groups:
        res_stats, tst_exs, test_results = tr_funcs.test_end_group(
            g.dloader, prep_model.run_epoch_kwargs, params.target_recalls, verbose=True
        )

        res_string = tr_funcs.get_nice_results_string(g.name, res_stats)
        print(res_string)

        precision, recalls, threshes = test_results.make_pr_curve()

        import pandas as pd
        import numpy as np

        df = pd.DataFrame(
            data={
                "precision": precision,
                "recall": recalls,
                "threshes": np.concatenate([threshes, [threshes[-1]]]),
            }
        )

        df.to_csv("./results_csv/UT_PR_curve.csv")

# groups[0].dset.seq_length = 512
# total_tokens = 0
# total_errors = 0
# total_correct = 0
# for x in groups[0].dset:
#     total_tokens += x[0][0].shape[0]
#     total_errors += np.count_nonzero(x[0][1])
#     total_correct += np.count_nonzero(x[0][1] == 0)
# print(total_tokens, total_correct, total_errors)
