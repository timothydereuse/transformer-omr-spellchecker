import json
import logging, datetime
from unittest import skip


class Params(object):
    '''
    contains all hyperparameters and logging information for a single training run.
    '''

    def __init__(self, base_file='params_default.json', log_training=False, mod_num=0):
        with open(base_file, 'r') as f:
            self.params_dict = json.load(f)

        for k in self.params_dict.keys():
            self.__dict__[k] = self.params_dict[k]

        self.log_training = log_training
        self.model_summary = (
            '{num_feats}-{output_feats}-{lstm_layers}-{tf_layers}'
            '{tf_heads}-{tf_depth}-{hidden_dim}-{ff_dim}').format(**self.lstut_settings)

        start_training_time = datetime.datetime.now().strftime("(%Y.%m.%d.%H.%M)")
        self.start_training_time = start_training_time
        self.params_id_str = f'{self.params_name}_{mod_num}_{start_training_time}_{self.model_summary}'
        self.log_fname = f'./logs/training_{self.params_id_str}.log'
        self.results_fname = f'./logs/test_results_{self.params_id_str}.log'
        if self.log_training:
            logging.basicConfig(filename=self.log_fname, filemode='w', level=logging.INFO,
                                format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
            if not any([type(x) is logging.StreamHandler for x in logging.getLogger().handlers]):
                logging.getLogger().addHandler(logging.StreamHandler())
        self.mod_num = mod_num
        self.mod_string = ''

        if mod_num > 0 and not hasattr(self, 'param_sweep'):
            raise ValueError(f'Given parameter file {base_file} has no modifications defined, '
                                'but the Params class was passed mod number {mod_num}.')
        elif mod_num > 0:
            self.apply_mod(self.param_sweep[mod_num - 1])

    def apply_mod(self, mod):
        sk = sorted(list(mod.keys()))
        self.mod_string = ' '.join([f'{k}-{mod[k]}' for k in sk])

        for k in mod.keys():
            name = k
            val = mod[k]
            if not name:
                return
            elif '.' not in name:
                assert name in self.__dict__.keys(), "modification name not found in params!"
                self.__dict__[name] = val
                self.params_dict[name] = val
            elif '.' in name:
                pt1, pt2 = name.split('.')
                assert pt1 in self.__dict__.keys(), "modification name not found in params!"
                assert pt2 in self.__dict__[pt1].keys(), "modification name not found in params!"
                self.__dict__[pt1][pt2] = val
                self.params_dict[pt1][pt2] = val


# -- constants that need to be here so that they can be referenced, but shouldn't be changed
flags = {
    'sos': [-1],
    'eos': [-2],
    'mask': [-3],
    'pad': [-4]}

notetuple_flags = {
    'sos': [0, 0, 0, 10],
    'eos': [0, 0, 0, 20],
    'mask': [0, 0, 0, 30],
    'pad': [0, 0, 0, 0]
}
