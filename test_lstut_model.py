import training_helper_functions as tr_funcs
from model_setup import PreparedLSTUTModel
import model_params
import torch

if __name__ == "__main__":
    model_path = "trained_models\lstut_best_LSTUT_TRIAL_0_(2023.01.10.17.06)_1-1-1-11-1-32-32.pt"
    saved_model_info = torch.load(model_path)

    params = model_params.Params('./param_sets/trial_lstut.json', False, 0)
    device, num_gpus = tr_funcs.get_cuda_info()

    prep_model = PreparedLSTUTModel(params, saved_model_info['model_state_dict'])
    groups = tr_funcs.make_test_dataloaders(params, prep_model.dset_kwargs)

    for g in groups:
        res_stats, tst_exs, test_results = tr_funcs.test_end_group(
            g.dloader,
            g.with_targets,
            prep_model.run_epoch_kwargs,
            params.target_recalls
            )

        res_string = tr_funcs.get_nice_results_string(g.name, res_stats)
        print(res_string)