import wandb
import numpy as np
import plot_outputs as po


def add_stats_to_wandb(res_stats, target_recalls, end_name):
    pairs = []
    for i, thresh in enumerate(target_recalls):
        endthresh = f"{end_name}.{thresh}"
        for met in [
            "precision",
            "true negative rate",
            "prop_positive_predictions",
            "prop_positive_targets",
        ]:
            pairs.append((f"final.{endthresh}.{met}", res_stats[met][thresh]))

    pairs.append(
        (f"final.{end_name}.average_precision", res_stats["average_precision"])
    )
    pairs.append(
        (f"final.{end_name}.normalized_recall", res_stats["normalized_recall"])
    )

    for k, v in pairs:
        wandb.log({k: v})


def save_examples_to_wandb(
    res_stats, tst_exs, v, target_recalls, end_name, num_examples_to_save
):
    wandb_dict = {}
    num_examples_to_save = min(num_examples_to_save, len(tst_exs["output"]))

    for j, thresh in enumerate(res_stats["threshes"]):

        # the 0 represents thresh optimized for f1 score instead
        target_recalls = target_recalls + ["F1"]

        inds_to_save = np.random.choice(
            len(tst_exs["output"]), num_examples_to_save, replace=False
        )
        for ind_to_save in inds_to_save:

            batch_name = f"{tst_exs['batch_names'][ind_to_save][-50:]} {tst_exs['batch_offsets'][ind_to_save]}"
            lines = po.plot_agnostic_results(
                tst_exs, v, thresh, return_arrays=True, ind=ind_to_save
            )
            table = wandb.Table(
                data=lines, columns=["ORIG", "INPUT", "TARGET", "OUTPUT", "RAW"]
            )
            # wandb_dict[f"{end_name}/table/{target_recalls[j]}/{batch_name}"] = table
            wandb.log({f"{end_name}/table/{target_recalls[j]}/{batch_name}": table})

    # wandb.run.summary[f"final_examples"] = wandb_dict
