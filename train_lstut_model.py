import time, logging, argparse, copy
import numpy as np
import torch, wandb
import wandb_logging
import agnostic_omr_dataloader as dl
import results_and_metrics as ttm
import run_tests_lstut_model as tlm
import training_helper_functions as tr_funcs
from model_setup import PreparedLSTUTModel
from torch.utils.data import DataLoader
import plot_outputs as po
import model_params

################################################
# PARSE COMMAND-LINE ARGS
################################################

parser = argparse.ArgumentParser(
    description="Training and testing script for the transformer-omr-spellchecker project. "
    "Must reference a .json parameters file (in the /param_sets folder. "
    "Requires pre-processed .h5 files containing symbolic music files in agnostic format; "
    "Some of these .h5 files are included with the transformer-omr-spellchecker repository on GitHub. "
    "Use the script run_all_data_preparation to make these files from scratch, or from another dataset."
)
parser.add_argument(
    "parameters", default="default_params.json", help="Parameter file in .json format."
)
parser.add_argument(
    "-m",
    "--mod_number",
    type=int,
    default=0,
    help="Index of specific modification to apply to given parameter set.",
)
parser.add_argument(
    "-w",
    "--wandb",
    type=ascii,
    action="store",
    default=None,
    help="Name of wandb project to log results to. "
    "If none supplied, results are printed to stdout (and log file if -l is used).",
)
parser.add_argument(
    "-l",
    "--logging",
    action="store_true",
    help="Whether or not to log training results to file.",
)
parser.add_argument(
    "-d",
    "--dryrun",
    action="store_true",
    help="Halts execution immediately before training begins.",
)
args = vars(parser.parse_args())

################################################
# SETTING UP DATASETS AND MODEL FOR TRAINING
################################################

params = model_params.Params(args["parameters"], args["logging"], args["mod_number"])
dry_run = args["dryrun"]
run_name = params.params_id_str + " " + params.mod_string

if (not dry_run) and args["wandb"]:
    wandb.init(
        project=args["wandb"].strip("'"),
        config=params.params_dict,
        entity="timothydereuse",
        tags=params.run_tags,
    )
    wandb.run.name = run_name

print("defining datasets...")
device, num_gpus = tr_funcs.get_cuda_info()

prep_model = PreparedLSTUTModel(params)

aug_dset_tr = dl.AgnosticOMRDataset(
    base="train", dset_fname=params.aug_dset_path, **prep_model.dset_kwargs
)
aug_dset_vl = dl.AgnosticOMRDataset(
    base="validate",
    dset_fname=params.aug_dset_path,
    **prep_model.dset_kwargs,
)

dset_tr = dl.AgnosticOMRDataset(
    base="train/omr", dset_fname=params.dset_path, **prep_model.dset_kwargs
)
dset_vl = dl.AgnosticOMRDataset(
    base="validate/omr",
    dset_fname=params.dset_path,
    **prep_model.dset_kwargs,
)

dloader = DataLoader(dset_tr, params.batch_size, pin_memory=True)
dloader_val = DataLoader(dset_vl, params.batch_size, pin_memory=True)
aug_dloader = DataLoader(aug_dset_tr, params.batch_size, pin_memory=True)
aug_dloader_val = DataLoader(aug_dset_vl, params.batch_size, pin_memory=True)

if dry_run:
    import sys

    sys.exit("Dry run successful. Exiting.")

#########################
# TRAIN MODEL
#########################

print("beginning training")
start_time = time.time()
val_losses = []
train_losses = []
best_model = None
now_finetuning = not params.finetuning
started_finetuning = 0
tr_funcs.log_gpu_info()

for epoch in range(params.num_epochs):
    epoch_start_time = time.time()

    # decide what dataloader to use for this training epoch:
    # are we finetuning? or combining all data?
    if not params.finetuning:
        this_epoch_dloader = [dloader, aug_dloader]
        this_epoch_val_dloader = [dloader_val, aug_dloader_val]
    elif params.finetuning and not now_finetuning:
        this_epoch_dloader = aug_dloader
        this_epoch_val_dloader = aug_dloader_val
    elif params.finetuning and now_finetuning:
        this_epoch_dloader = dloader
        this_epoch_val_dloader = dloader_val

    # perform training epoch
    prep_model.model.train()
    train_loss, tr_exs = tr_funcs.run_epoch(
        dloader=this_epoch_dloader,
        train=True,
        log_each_batch=False,
        **prep_model.run_epoch_kwargs,
    )
    _, gpu_used, gpu_free, _ = tr_funcs.log_gpu_info()

    # test on validation set
    prep_model.model.eval()
    num_entries = 0
    val_loss = 0.0
    with torch.no_grad():
        val_loss, val_exs = tr_funcs.run_epoch(
            dloader=this_epoch_val_dloader,
            train=False,
            log_each_batch=False,
            **prep_model.run_epoch_kwargs,
        )

    val_losses.append(val_loss)
    train_losses.append(train_loss)
    prep_model.scheduler.step()

    # get thresholds that maximize f1 and match required recall scores
    sig_val_output = torch.sigmoid(val_exs["output"])
    sig_train_output = torch.sigmoid(tr_exs["output"])
    tr_mcc, tr_thresh = ttm.multilabel_thresholding(sig_train_output, tr_exs["target"])
    val_mcc = ttm.matthews_correlation(
        sig_val_output.cpu(), val_exs["target"].cpu(), tr_thresh
    )
    val_norm_recall = ttm.normalized_recall(
        sig_val_output.cpu(), val_exs["target"].cpu()
    )
    val_threshes = ttm.find_thresh_for_given_recalls(
        sig_val_output.cpu(), val_exs["target"].cpu(), params.target_recalls
    )

    epoch_end_time = time.time()
    print(
        f"epoch {epoch:3d} | "
        f"sys/sec_per_epoch         {(epoch_end_time - epoch_start_time):3.5e} | "
        f"tr/loss      {train_loss:1.6e} | "
        f"val/loss        {val_loss:1.6e} | "
        f"tr/thresh       {tr_thresh:1.5f} | "
        f"tr/mcc          {tr_mcc:1.6f} | "
        f"val/mcc         {val_mcc:1.6f} | "
        f"val/norm_recall {val_norm_recall:1.6f} | "
        f"sys/gpu_free        {gpu_free:1.6f} | "
        f"sys/gpu_used        {gpu_used:1.6f} | "
    )

    if args["wandb"]:
        wandb.log(
            {
                "sys/epoch_s": (epoch_end_time - epoch_start_time),
                "tr/loss": train_loss,
                "val/loss": val_loss,
                "tr/thresh": tr_thresh,
                "tr/mcc": tr_mcc,
                "val/mcc": val_mcc,
                "val/norm_recall": val_norm_recall,
                "sys/gpu_free": gpu_free,
                "sys/gpu_used": gpu_used,
            }
        )

    # keep snapshot of best model
    cur_model = {
        "epoch": epoch,
        "model_state_dict": prep_model.model.state_dict(),
        "optimizer_state_dict": prep_model.optimizer.state_dict(),
        "scheduler_state_dict": prep_model.scheduler.state_dict(),
        "val_losses": val_losses,
        "val_threshes": val_threshes,
    }
    if (len(val_losses) > 1) and (val_losses[-1] < min(val_losses[:-1])):
        best_model = copy.deepcopy(cur_model)
        m_name = f"./trained_models/lstut_best_{params.params_id_str}.pt"
        torch.save(best_model, m_name)

    time_since_best = epoch - val_losses.index(min(val_losses))
    elapsed = time.time() - start_time
    # is it time... to fine tune?

    ft_1 = not now_finetuning and (time_since_best > params.early_stopping_patience)
    ft_2 = not now_finetuning and elapsed > (params.aug_max_time_minutes * 60)

    if ft_1 or ft_2:
        print(f"done training on augmented data: epoch {epoch}. beginning to finetune")
        now_finetuning = True
        started_finetuning = epoch
        prep_model.model.module.freeze_tf()
    elif (
        now_finetuning
        and (time_since_best > params.early_stopping_patience)
        and (epoch - started_finetuning > time_since_best)
    ):
        # early stopping
        print(
            f"stopping early at epoch {epoch} because validation score stopped increasing"
        )
        break
    elif now_finetuning and elapsed > (params.max_time_minutes * 60):
        # stopping based on time limit defined in params file
        print(f"stopping early at epoch {epoch} because of time limit")
        break

end_time = time.time()
print(
    f"Training over at epoch at epoch {epoch}.\n"
    f"Total training time: {end_time - start_time} s."
)

# save a final model checkpoint
# if max_epochs reached, or early stopping condition reached, save best model
best_epoch = best_model["epoch"]
m_name = f"./trained_models/lstut_best_{params.params_id_str}.pt"
torch.save(best_model, m_name)

#########################
# TESTING TRAINED MODEL
#########################

if args["wandb"]:
    wandb.run.summary["total_training_time"] = end_time - start_time
end_groups = tr_funcs.make_test_dataloaders(params, prep_model.dset_kwargs)

for end_group in end_groups:

    res_stats, tst_exs, test_results = tr_funcs.test_end_group(
        end_group.dloader,
        prep_model.run_epoch_kwargs,
        params.target_recalls,
    )

    res_string = tr_funcs.get_nice_results_string(end_group.name, res_stats)
    print(res_string)

    if args["wandb"]:
        wandb_logging.add_stats_to_wandb(
            res_stats, params.target_recalls, end_group.name
        )
        wandb_logging.save_examples_to_wandb(
            res_stats,
            tst_exs,
            prep_model.v,
            params.target_recalls,
            end_group.name,
            params.num_examples_to_save,
        )

wandb.finish()
