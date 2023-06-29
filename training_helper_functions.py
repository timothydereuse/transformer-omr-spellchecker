import torch
import logging
import numpy as np
import plot_outputs as po
import results_and_metrics as ttm
import agnostic_omr_dataloader as dl
from torch.utils.data import DataLoader
from collections import namedtuple
import GPUtil


def get_cuda_info():
    num_gpus = torch.cuda.device_count()
    if torch.cuda.is_available():
        logging.info(f"found {num_gpus} gpus")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device, num_gpus


def log_gpu_info():
    total, used, free, load = (0, 0, 0, 0)
    for g in GPUtil.getGPUs():
        total += g.memoryTotal
        used += g.memoryUsed
        free += g.memoryFree
        load += g.load
    return (total, used, free, load)


def make_test_dataloaders(params, kwargs_dict):
    # make dloaders for all test datasets identified in parameters file

    all_dset_groups = []
    EndGroup = namedtuple("TestGroup", "dset dloader name real_data")

    for test_set in params.test_sets:
        new_kwargs = dict(kwargs_dict)

        if test_set["real_data"]:
            new_kwargs["dset_fname"] = params.dset_path
        else:
            new_kwargs["dset_fname"] = params.aug_dset_path

        test_dset = dl.AgnosticOMRDataset(base=test_set["base"], **new_kwargs)
        dloader_omr = DataLoader(test_dset, params.batch_size, pin_memory=True)
        all_dset_groups.append(
            EndGroup(test_dset, dloader_omr, test_set["base"], test_set["real_data"])
        )

    return all_dset_groups


def get_merged_dataloaders(*itrs):
    for dload_result in zip(itrs):
        merged1 = []
        merged2 = []
        for single in dload_result:
            merged1.append(single[0])
            merged2.extend(single[1])
            # merge them in
        merged_batch = torch.concatenate(merged1, axis=0)
        yield merged_batch, merged2


def itr_merge(itrs):
    for itr in itrs:
        for v in itr:
            yield v


def run_epoch(
    model,
    dloader,
    criterion,
    example_generator,
    optimizer=None,
    scheduler=None,
    device="cpu",
    train=True,
    log_each_batch=False,
    clip_grad_norm=0.5,
    test_results=None,
    autoregressive=False,
):
    """
    Performs a training or validation or testing epoch.
    @model: the model to use.
    @dloader: the dataloader to fetch data from.
    @optimizer: the optimizer to use, if training. Set to None for inference.
    @scheduler: LR scheduler
    @criterion: the loss function to use, for training and validation.
    @device: the device on which to put the model.
    @train: if true, performs a gradient update on the model's weights. if false, treated as
            a validation run.
    @log_each_batch: if true, logs information about each batch's loss / time elapsed.
    @if a TestResults object, then logs all targets and outputs encountered during training
        into that TestResults object so that the results can be evaluated later.
    @autoregressive: if true, feeds the target into the model along with the input, for
        autoregressive teacher forcing.
    """
    num_seqs_used = 0
    total_loss = 0.0

    if type(dloader) is list:
        merged_dloader = itr_merge(dloader)
    else:
        merged_dloader = dloader

    for i, dloader_output in enumerate(merged_dloader):
        batch = dloader_output[0]
        batch_metadata = dloader_output[1]

        if type(batch) == list and len(batch) == 2:
            inp, target = batch
        elif type(batch) == torch.Tensor:
            batch = batch.float().cpu()
            inp, target = example_generator.add_errors_to_batch(batch)
        else:
            raise ValueError(f"type {type(batch)} and len {len(batch)} wrong")

        inp = inp.to(device).type(torch.long)
        target = target.to(device)

        if train:
            optimizer.zero_grad()

        if autoregressive:
            output = model(inp, target).squeeze(-1)
        else:
            output = model(inp).squeeze(-1)

        loss = criterion(output, target)

        if test_results:
            test_results.update(output, target)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            if (
                scheduler.state_dict()["last_epoch"]
                < scheduler.state_dict()["total_steps"]
            ):
                scheduler.step()

        batch_loss = loss.sum().item()
        total_loss += batch_loss
        num_seqs_used += target.numel()

        if log_each_batch:
            log_loss = batch_loss / target.numel()
            print(f"    batch {i}, loss {log_loss:2.7e}")

        if i == 0:
            example_dict = {
                "orig": batch,
                "input": inp,
                "target": target,
                "output": output,
                "batch_names": batch_metadata[2],
                "batch_offsets": batch_metadata[1],
                "batch_file_inds": batch_metadata[0],
            }

    mean_loss = total_loss / max(1, num_seqs_used)

    return mean_loss, example_dict


def test_end_group(end_dloader, run_epoch_kwargs, target_recalls):

    # make test_results with dummy threshes object, to fill in later
    test_results = ttm.TestResults(threshes=[], target_recalls=target_recalls)

    # run single epoch over test set
    with torch.no_grad():
        tst_loss, tst_exs = run_epoch(
            dloader=end_dloader,
            train=False,
            log_each_batch=False,
            test_results=test_results,
            **run_epoch_kwargs,
        )

    # get actual thresholds for given recalls
    sig_tst_outputs = test_results.sigmoid_outputs()
    tst_threshes = ttm.find_thresh_for_given_recalls(
        sig_tst_outputs, test_results.targets, target_recalls
    )
    mcc, f1_thresh = ttm.multilabel_thresholding(sig_tst_outputs, test_results.targets)
    tst_threshes.append(f1_thresh)
    test_results.threshes = tst_threshes
    res_stats = test_results.calculate_stats()

    for k in res_stats:
        for k2 in res_stats[k]:
            res_stats[k][k2] = np.round(res_stats[k][k2], 4)
    res_stats["average_precision"] = test_results.average_precision()
    res_stats["normalized_recall"] = ttm.normalized_recall(
        test_results.outputs, test_results.targets
    )
    res_stats["threshes"] = tst_threshes
    res_stats["max_mcc"] = mcc

    return res_stats, tst_exs, test_results


def get_nice_results_string(end_name, res_stats):

    result_string = (
        ""
        f"{end_name} STATS:\n"
        f'{end_name} threshes:                    {res_stats["threshes"]}\n'
        f'{end_name}_precision:                   {res_stats["precision"]} | \n'
        f'{end_name}_recall:                      {res_stats["recall"]} | \n'
        f'{end_name}_true negative:               {res_stats["true negative rate"]} \n'
        f'{end_name}_prop_positive_predictions:   {res_stats["prop_positive_predictions"]} \n'
        f'{end_name}_prop_positive_targets:       {res_stats["prop_positive_targets"]} \n'
        f'{end_name}_max_mcc:                     {res_stats["max_mcc"]} \n'
        f'{end_name}_average_precision:           {res_stats["average_precision"]} \n'
        f'{end_name}_normalized_recall:           {res_stats["normalized_recall"]} \n'
    )

    return result_string


if __name__ == "__main__":

    from data_augmentation import error_gen_logistic_regression as err_gen
    import agnostic_omr_dataloader as dl
    from torch.utils.data import DataLoader
    from models.LSTUT_model import LSTUT
    import data_management.vocabulary as vocab

    if not any(
        [type(x) is logging.StreamHandler for x in logging.getLogger().handlers]
    ):
        logging.getLogger().addHandler(logging.StreamHandler())

    print("making vocabulary and dataset")
    v = vocab.Vocabulary(load_from_file="./data_management/vocab.txt")
    dset = dl.AgnosticOMRDataset(
        base=None,
        dset_fname="./processed_datasets/quartets_felix_omr_agnostic.h5",
        seq_length=50,
        minibatch_div=0.02,
        vocabulary=v,
    )
    dset_test = dl.AgnosticOMRDataset(
        base=None,
        dset_fname="./processed_datasets/supervised_omr_targets.h5",
        seq_length=50,
        minibatch_div=0.1,
        vocabulary=v,
    )

    print("making error generator")
    error_generator = err_gen.ErrorGenerator(
        smoothing=1,
        simple=False,
        simple_error_rate=0.05,
        models_fpath=("./data_augmentation/quartet_omr_error_models.joblib"),
    )

    print("testing dataloader")
    dload = DataLoader(dset, batch_size=3)
    dload_test = DataLoader(dset_test, batch_size=3)
    for i, batch in enumerate(dload):
        batch = batch[0].float()
        inp, target = error_generator.add_errors_to_batch(batch)
        print(inp.shape, batch.shape)
        if i > 2:
            break

    lstut_settings = {
        "seq_length": dset.seq_length,
        "d_model": 256,
        "output_feats": 1,
        "lstm_layers": 2,
        "tf_layers": 1,
        "tf_heads": 1,
        "tf_depth": 2,
        "hidden_dim": 32,
        "ff_dim": 32,
        "dropout": 0.1,
        "vocab_size": v.num_words,
    }

    print("defining model")
    model = LSTUT(**lstut_settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    print("running epoch")
    loss, exs = run_epoch(
        model, dload, optimizer, criterion, error_generator, log_each_batch=True
    )

    print("running test epoch")
    loss, exs = run_epoch(
        model, dload_test, optimizer, criterion, error_generator, log_each_batch=True
    )
