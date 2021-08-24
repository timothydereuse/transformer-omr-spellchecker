import torch
import logging
import numpy as np
import plot_outputs as po

def get_cuda_info():
    num_gpus = torch.cuda.device_count()
    if torch.cuda.is_available():
        logging.info(f'found {num_gpus} gpus')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device, num_gpus

def log_gpu_info():
    for i in range(torch.cuda.device_count()):
        t = torch.cuda.get_device_properties(i)
        c = torch.cuda.memory_cached(i) / (2 ** 10)
        a = torch.cuda.memory_allocated(i) / (2 ** 10)
        logging.info(f'device: {t}, memory cached: {c:5.2f}, memory allocated: {a:5.2f}')


# def make_point_set_target(batch, dloader, device='cpu', make_examples_settings={}):
#     errored_input, error_locs = mse.error_indices(batch, **make_examples_settings)
#     # dset = dloader.dataset
#     # errored_input = dset.normalize_batch(errored_input).to(device)
#     # error_locs = dset.normalize_batch(error_locs).to(device)
#     return errored_input, error_locs


def run_epoch(model, dloader, optimizer, criterion, example_generator, device='cpu',
              train=True, log_each_batch=False, clip_grad_norm=0.5, test_results=None, autoregressive=False):
    '''
    Performs a training or validation epoch.
    @model: the model to use.
    @dloader: the dataloader to fetch data from.
    @optimizer: the optimizer to use, if training.
    @criterion: the loss function to use.
    @device: the device on which to perform training.
    @train: if true, performs a gradient update on the model's weights. if false, treated as
            a validation run.
    @log_each_batch: if true, logs information about each batch's loss / time elapsed.
    @autoregressive: if true, feeds the target into the model along with the input, for
        autoregressive teacher forcing.
    '''
    num_seqs_used = 0
    total_loss = 0.

    for i, batch in enumerate(dloader):

        batch = batch.float().cpu()
        inp, target = example_generator.add_errors_to_batch(batch, simple=True, parallel=3)

        # batch = batch.to(device)
        inp = torch.tensor(inp, device=device).type(torch.long)
        target = torch.tensor(target, device=device)

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

        batch_loss = loss.sum().item()
        total_loss += batch_loss
        num_seqs_used += target.numel()

        if log_each_batch:
            log_loss = (batch_loss / target.numel())
            logging.info(f'    batch {i}, loss {log_loss:2.7e}')

    mean_loss = total_loss / num_seqs_used
    example_dict = {'orig': batch, 'input': inp, 'target': target, 'output': output}
    return mean_loss, example_dict


if __name__ == '__main__':

    import toy_datasets as td
    from torch.utils.data import DataLoader
    from models.LSTUT_model import LSTUT

    dset_args = {
        'num_feats': 1,
        'num_seqs': 300,
        'seq_length': 128,
        'seq_period': 128 // 5,
        'phase_vary': 0.01,
        'freq_vary': 0.01
    }
    dset = td.SequenceCopyDataset(**dset_args)

    dload = DataLoader(dset, batch_size=50)
    for i, batch in enumerate(dload):
        batch = batch.float()
        inp, target = make_point_set_target(batch, dload, device='cpu')
        if i > 2:
            break

    lstut_settings = {
            "num_feats": 1,
            "output_feats": 1,
            "lstm_layers": 2,
            "n_layers": 1,
            "n_heads": 1,
            "tf_depth": 2,
            "hidden_dim": 32,
            "ff_dim": 32,
            "dropout": 0.1
        }

    model = LSTUT(**lstut_settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    loss, exs = run_epoch(model, dload, optimizer, criterion, log_each_batch=True)

    # fig, axs = po.plot_set(exs, dset, 4)
