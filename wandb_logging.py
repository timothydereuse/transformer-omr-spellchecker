import wandb
import numpy as np
import plot_outputs as po

def add_stats_to_wandb(res_stats, target_recalls, end_name):
    for i, thresh in enumerate(target_recalls):
        wandb.run.summary[f"{end_name}_{thresh}_precision"] = res_stats["precision"][thresh]
        wandb.run.summary[f"{end_name}_{thresh}_true_negative"] = res_stats["true negative rate"][thresh]
        wandb.run.summary[f"{end_name}_{thresh}_prop_positive_predictions"] = res_stats["prop_positive_predictions"][thresh]
        wandb.run.summary[f"{end_name}_{thresh}_prop_positive_targets"] = res_stats["prop_positive_targets"][thresh]
    wandb.run.summary[f"{end_name}_average_precision"] = res_stats['average_precision']
    wandb.run.summary[f"{end_name}_normalized_recall"] = res_stats['normalized_recall']


def save_examples_to_wandb(res_stats, tst_exs, v, target_recalls, end_name, num_examples_to_save):
    wandb_dict = {}
    num_examples_to_save = min(num_examples_to_save, len(tst_exs['output']))

    for j, thresh in enumerate(res_stats['threshes']):
        
        # the 0 represents thresh optimized for f1 score instead
        target_recalls = target_recalls + ['F1']

        inds_to_save = np.random.choice(len(tst_exs['output']), num_examples_to_save, replace=False)
        for ind_to_save in (inds_to_save):
            
            batch_name = f"{tst_exs['batch_names'][ind_to_save]} {tst_exs['batch_offsets'][ind_to_save]}"
            lines = po.plot_agnostic_results(tst_exs, v, thresh, return_arrays=True, ind=ind_to_save)
            table = wandb.Table(data=lines, columns=['ORIG', 'INPUT', 'TARGET', 'OUTPUT', 'RAW'])
            wandb_dict[f'{end_name}_{target_recalls[j]}_{batch_name}'] = table
        
    wandb.run.summary[f'final_examples'] = wandb_dict