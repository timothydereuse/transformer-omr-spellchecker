import torch
import agnostic_omr_dataloader as dl
from data_management.vocabulary import Vocabulary
from data_augmentation.error_gen_logistic_regression import ErrorGenerator
import knn_classifier.perform_knn as perform_knn
import pygad
import numpy as np
import logging, datetime

pieces_to_try = 10
parallel = 8
embedding_name = r'./knn_classifier/agnostic_embedding_vectors_byline.npy'

if __name__ == "__main__":

    start_training_time = datetime.datetime.now().strftime("(%Y.%m.%d.%H.%M)")
    logname = f'./logs/ga_train_{start_training_time}.log'
    logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

    error_generator = ErrorGenerator(
        models_fpath='./processed_datasets/quartet_omr_error_models_byline.joblib',
        smoothing=1,
        simple=True,
        simple_error_rate=(2/1)
        )
    error_generator.simple_probs = [0, 0.5, 0.5]

    v = Vocabulary(load_from_file='./data_management/vocab.txt')

    dset_kwargs = {
        'dset_fname': 'processed_datasets/all_string_quartets_agnostic_byline.h5',
        'seq_length': 100, # N.B. this does not matter because of reshape(-1) below
        'padding_amt': 1,
        'minibatch_div': 1,
        'vocabulary': v,
        'shuffle_files': True
    }

    dset_tr = dl.AgnosticOMRDataset(base='train', **dset_kwargs)
    embedding_vectors = np.load(embedding_name)

    # scores = perform_knn.test_knn_detection(dset_tr, error_generator, embedding_vectors, pieces_to_try=2)

    argument_order = ['window_size', 'n_nearest_neighbors', 'embedding_dim_reduce',
        'knn_bypass_thresh', 'metric_order', 'use_big_dset']
    gene_space = [
        [3, 5, 7, 9, 11, 13, 15], # window size
        [30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140, 150, 170, 200, 250], # n nearest neighbors
        list(range(3, 25)), # embedding reduction dims
        list(range(2, 50)), # knn bypass thresh
        [1, 2], # metric_order
        [0, 1], # use big dset
    ]

    def fitness_func(solution, solution_idx):
        kwargs = {}
        for i, a in enumerate(argument_order):
            kwargs[a] = int(solution[i])

        try:
            scores = perform_knn.test_knn_detection(
                dset_tr,
                error_generator,
                embedding_vectors,
                pieces_to_try=pieces_to_try,
                smoothing=500,
                **kwargs
            )
        except RuntimeError:
            scores = [0, 0]

        fitness = np.mean(scores)
        print(solution, solution_idx, fitness)

        return fitness
    
    def callback_gen(ga_instance):
        print('callback func')
        logging.info(f"Generation : {ga_instance.generations_completed}")
        logging.info(f"Fitness of the best solution : {ga_instance.best_solution()[1]}")
        logging.info(f"Best solution : {ga_instance.best_solution()[0]}")

    # Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
    ga_instance = pygad.GA(num_generations=200,
                        fitness_func=fitness_func,
                        num_parents_mating=8,
                        sol_per_pop=16,
                        num_genes=len(gene_space),
                        gene_space=gene_space,
                        mutation_type="adaptive",
                        mutation_num_genes=(3, 1),
                        stop_criteria='saturate_3',
                        on_generation=callback_gen,
                        parallel_processing=['thread', parallel])

    # Running the GA to optimize the parameters of the function.
    ga_instance.run()
    ga_instance.save('saved_ga_instance')
    
