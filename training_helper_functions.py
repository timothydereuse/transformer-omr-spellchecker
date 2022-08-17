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
              train=True, log_each_batch=False, clip_grad_norm=0.5, test_results=None,
              autoregressive=False, batch_includes_training_data=False):
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

    for i, dloader_output in enumerate(dloader):
        batch = dloader_output[0]
        batch_metadata = dloader_output[1]

        if batch_includes_training_data and len(batch) == 2:
            inp, target = batch
        else:
            batch = batch.float().cpu()
            inp, target = example_generator.add_errors_to_batch(batch)

        # batch = batch.to(device)
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
            test_results.update(torch.sigmoid(output), target)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        batch_loss = loss.sum().item()
        total_loss += batch_loss
        num_seqs_used += target.numel()

        if log_each_batch:
            log_loss = (batch_loss / target.numel())
            print(f'    batch {i}, loss {log_loss:2.7e}')

    mean_loss = total_loss / max(1, num_seqs_used)
    example_dict = {
        'orig': batch if not batch_includes_training_data else inp,
        'input': inp,
        'target': target,
        'output': output, 
        'batch_names': batch_metadata[2],
        'batch_offsets': batch_metadata[1],
        'batch_file_inds': batch_metadata[0]
        }
    return mean_loss, example_dict


if __name__ == '__main__':

    from data_augmentation import error_gen_logistic_regression as err_gen
    import agnostic_omr_dataloader as dl
    from torch.utils.data import DataLoader
    from models.LSTUT_model import LSTUT
    import data_management.vocabulary as vocab

    if not any([type(x) is logging.StreamHandler for x in logging.getLogger().handlers]):
        logging.getLogger().addHandler(logging.StreamHandler())

    print('making vocabulary and dataset')
    v = vocab.Vocabulary(load_from_file='./data_management/vocab.txt')
    dset = dl.AgnosticOMRDataset(
        base=None,
        dset_fname="./processed_datasets/quartets_felix_omr_agnostic.h5",
        seq_length=50,
        dataset_proportion=0.02,
        vocabulary=v,
    )
    dset_test = dl.AgnosticOMRDataset(
        base=None,
        dset_fname="./processed_datasets/supervised_omr_targets.h5",
        seq_length=50,
        dataset_proportion=0.1,
        vocabulary=v,
    )

    print('making error generator')
    error_generator = err_gen.ErrorGenerator(
        smoothing=1,
        simple=False,
        simple_error_rate=0.05,
        models_fpath=('./data_augmentation/quartet_omr_error_models.joblib')
    )

    print('testing dataloader')
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
            "num_feats": 1,
            "output_feats": 1,
            "lstm_layers": 2,
            "tf_layers": 1,
            "tf_heads": 1,
            "tf_depth": 2,
            "hidden_dim": 32,
            "ff_dim": 32,
            "dropout": 0.1,
            "vocab_size": v.num_words
        }

    print('defining model')
    model = LSTUT(**lstut_settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    print('running epoch')
    loss, exs = run_epoch(model, dload, optimizer, criterion, error_generator, log_each_batch=True)

    print('running test epoch')
    loss, exs = run_epoch(model, dload_test, optimizer, criterion, error_generator, log_each_batch=True)

