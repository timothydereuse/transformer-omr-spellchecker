import json
import logging, datetime


class Params(object):
    '''
    contains all hyperparameters and logging information for a single training run.
    '''

    def __init__(self, base_file='params/default_params.json', log_training=False, mod_num=0):
        with open(base_file, 'r') as f:
            params = json.load(f)

        for k in params.keys():
            self.__dict__[k] = params[k]

        self.modifications = []
        if mod_num > 0:
            if not hasattr(self, 'parameter_searching'):
                raise ValueError(f'Given parameter file {base_file} has no modifications defined, '
                                 'but the Params class was passed mod number {mod_num}.')
            for k in self.parameter_searching:
                for i in self.parameter_searching[k]:
                    self.modifications.append((k, i))
                self.apply_mod(mod_num - 1)

        self.model_summary = (
            '{num_feats}-{num_output_points}-{n_layers}-'
            '{n_heads}-{tf_depth}-{hidden_dim}-{ff_dim}').format(**self.set_transformer_settings)

        start_training_time = datetime.datetime.now().strftime("(%Y.%m.%d.%H.%M)")
        params_id_str = f'{self.params_name}_{mod_num}_{start_training_time}_{self.model_summary}'
        self.log_fname = f'./logs/training_{params_id_str}.log'
        self.results_fname = f'./logs/test_results_{params_id_str}.log'
        if log_training:
            logging.basicConfig(filename=self.log_fname, filemode='w', level=logging.INFO,
                                format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
            if not any([type(x) is logging.StreamHandler for x in logging.getLogger().handlers]):
                logging.getLogger().addHandler(logging.StreamHandler())

    def apply_mod(self, num):
        name, val = self.modifications[num]
        if not name:
            return
        elif '.' not in name:
            self.__dict__[name] = val
        elif '.' in name:
            pt1, pt2 = name.split('.')
            self.__dict__[pt1][pt2] = val


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
