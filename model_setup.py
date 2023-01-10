import torch.nn as nn
import torch
import training_helper_functions as tr_funcs
import models.LSTUT_model as lstut
import data_management.vocabulary as vocab
import data_augmentation.error_gen_logistic_regression as err_gen


class PreparedLSTUTModel():
    # struct-like object that takes in a params object and generates an LSTUT model
    # for inference or for training. assembles parameter sets for running epochs
    # and instantiating datasets. holds vocabulary, error generator, loss,
    # optimizer, scheduler, model itself.

    def __init__(self, params):
        self.params = params

        self.v = vocab.Vocabulary(load_from_file=params.saved_vocabulary)

        self.error_generator = err_gen.ErrorGenerator(
            simple=params.simple_errors,
            smoothing=params.error_gen_smoothing,
            simple_error_rate=params.simple_error_rate,
            parallel=params.errors_parallel,
            models_fpath=params.error_model
        )

        self.lstut_settings = params.lstut_settings
        self.lstut_settings['vocab_size'] = self.v.num_words
        self.lstut_settings['seq_length'] = params.seq_length

        self.device, num_gpus = tr_funcs.get_cuda_info()
        self.lstut_model = lstut.LSTUT(**self.lstut_settings).to(self.device)
        self.model = nn.DataParallel(self.lstut_model, device_ids=list(range(num_gpus)))
        self.model = self.model.float()
        self.model_size = sum(p.numel() for p in self.model.parameters())
        print(f'created model with n_params={self.model_size}')

        self.dset_kwargs = {
            'dset_fname': params.dset_path,
            'seq_length': params.seq_length,
            'padding_amt': params.padding_amt,
            'dataset_proportion': params.dataset_proportion,
            'vocabulary': self.v
        }

        self.class_ratio = (params.error_gen_smoothing)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(self.class_ratio))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=self.optimizer, **params.scheduler_settings)

        self.run_epoch_kwargs = {
            'model': self.model,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'device': self.device,
            'example_generator': self.error_generator,
        }