import torch
import logging
import make_supervised_examples as mse
import model_params as params
from chamferdist import ChamferDistance


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


def make_point_set_target(batch, dloader, device):
    dset = dloader.dataset
    errored_input, error_locs = mse.error_indices(batch, num_indices=5, max_num_error_pts=40)
    errored_input = dset.normalize_batch(errored_input).to(device)
    error_locs = dset.normalize_batch(error_locs).to(device)
    return errored_input, error_locs


def run_epoch(model, dloader, optimizer, criterion, device='cpu',
              train=True, log_each_batch=False, autoregressive=False):
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
        inp, target = make_point_set_target(batch, dloader, device)

        # batch = batch.to(device)
        inp = inp.to(device)
        target = target.to(device)

        if train:
            optimizer.zero_grad()

        if autoregressive:
            output = model(inp, target)
        else:
            output = model(inp)

        # for chamfer distance, in this context, target should be first
        loss = criterion(output, target)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_gradient_norm)
            optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_seqs_used += target.numel()

        if log_each_batch:
            logging.info(f'    batch {i}, loss {batch_loss / target.numel():3.7f}')

    mean_loss = total_loss / num_seqs_used
    example_dict = {'input': inp, 'target': target, 'output': output}
    return mean_loss, example_dict


if __name__ == '__main__':

    import data_loaders as dl
    from torch.utils.data import DataLoader
    from models.set_transformer_model import SetTransformer

    dset = dl.MidiNoteTupleDataset(
        dset_fname=params.dset_path,
        seq_length=512,
        base='train',
        padding_amt=params.padding_amt,
        trial_run=0.03)

    dload = DataLoader(dset, batch_size=50)
    for i, batch in enumerate(dload):
        batch = batch.float()
        inp, target = make_point_set_target(batch, dload, device='cpu')
        if i > 2:
            break

    set_transformer_settings = {
        'num_feats': 4,
        'num_output_points': 40,
        'n_layers_prepooling': 2,
        'n_layers_postpooling': 2,
        'n_heads': 1,
        'hidden_dim': 32,
        'ff_dim': 32,
        'dropout': 0.1
    }

    model = SetTransformer(**set_transformer_settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = ChamferDistance()

    loss, exs = run_epoch(model, dload, optimizer, criterion, log_each_batch=True)

    # fig, axs = po.plot_set(exs, dset, 4)
