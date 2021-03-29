import torch
import logging
import make_supervised_examples as mse
import model_params as params


def log_gpu_info():
    for i in range(torch.cuda.device_count()):
        t = torch.cuda.get_device_properties(i)
        c = torch.cuda.memory_cached(i) / (2 ** 10)
        a = torch.cuda.memory_allocated(i) / (2 ** 10)
        logging.info(f'device: {t}, memory cached: {c:5.2f}, memory allocated: {a:5.2f}')


def make_point_set_target(batch, dloader, device):
    dset = dloader.dataset
    errored_input, error_locs = mse.error_indices(batch)
    errored_input = torch.tensor(dset.normalize_batch(errored_input)).to(device)
    for i in range(len(error_locs)):
        error_locs[i] = torch.tensor(dset.normalize_batch(error_locs[i])).to(device)
    return errored_input, error_locs


def run_epoch(model, dloader, optimizer, criterion, device, train=True, log_each_batch=False, autoregressive=False):
    num_seqs_used = 0
    total_loss = 0.

    for i, batch in enumerate(dloader):

        batch = batch.float().cpu()
        inp, target = make_point_set_target(batch, dloader, device)

        batch = batch.to(device)
        inp = inp.to(device)
        target = target.to(device)

        if train:
            optimizer.zero_grad()

        if autoregressive:
            output = model(inp, target)
        else:
            output = model(inp)

        loss = criterion(
            output.view(output.shape[0], -1),
            target.view(target.shape[0], -1)
            )

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
    return mean_loss, (inp, target, output)
