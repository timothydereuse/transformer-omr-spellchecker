if __name__ == "__main__":

    from data_augmentation.error_gen_logistic_regression import ErrorGenerator
    from agnostic_omr_dataloader import AgnosticOMRDataset
    from torch.utils.data import DataLoader
    from data_management.vocabulary import Vocabulary

    dset_path = r'./processed_datasets/quartets_felix_omr_agnostic.h5'
    v = Vocabulary(load_from_file='./data_management/vocab.txt')

    seq_len = 50
    proportion = 0.2
    dset = AgnosticOMRDataset(dset_path, seq_len, v)

    dload = DataLoader(dset, batch_size=5)
    batches = []
    for i, x in enumerate(dload):
        print(i, x.shape)
        batches.append(x)
        if i > 2:
            break

    print('creating error generator')
    e = ErrorGenerator(ngram=5, smoothing=0.7, parallel=4, models_fpath='./data_augmentation/quartet_omr_error_models.joblib')

    synth_error = e.get_synthetic_error_sequence(x[0].numpy())
    simple_error = e.get_simple_synthetic_error_sequence(x[0].numpy())
    print('adding errors to entire batch...')
    for i in range(5):
        print(i)
        X, Y = e.add_errors_to_batch(x.numpy())
        print(X.shape, Y.shape)