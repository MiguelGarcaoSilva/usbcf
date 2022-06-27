from surprise import BaselineOnly, KNNWithMeans, SVD, SVDpp, NMF, CoClustering
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import dump
from surprise import dataset
from usbcf import USBCF_nomem, USBCFCombineBicSols_nomem
from bbcf import BBCF, BBCF_nomem
from tqdm import tqdm
from surprise.model_selection import PredefinedKFold, KFold
from sklearn.model_selection import ParameterGrid
import multiprocessing
import numpy as np
import pandas as pd
import random
import copy
import os
import datetime
import logging
import sys
import glob
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler(sys.stdout))
_logger.setLevel(level=logging.INFO)



def load_mvlens_1M_dataset():
    # data_1M = Dataset.load_builtin("ml-1m")

    data_1M = Dataset.load_from_file(os.environ["SURPRISE_DATA_FOLDER"] +
                                      "ml-1m/ml-1m/ratings.dat",
                                      reader=Reader('ml-1m'))
    return data_1M



def run_training_test(name, algo, fold, train, test, output_path):

    path_partition = os.path.join(output_path,
                                  "iteration"+str(fold+1))

    modelpath = path_partition + "/" + name + ".bpk"
    if os.path.isfile(modelpath):
        return None
    else:
        algo = copy.deepcopy(algo)
        start = datetime.datetime.now()
        model = algo.fit(train)
        end = datetime.datetime.now()
        delta = end-start
        fit_time = int(delta.total_seconds() * 1000)
        start = datetime.datetime.now()
        predictions = model.test(test)
        end = datetime.datetime.now()
        delta = end-start
        pred_time = int(delta.total_seconds() * 1000)

        result_stats = [name, fold+1]
        result_stats.extend([fit_time, pred_time])

        if hasattr(model, 'stats_bics_sol'):
            if hasattr(model, 'users_sizes'):
                result_stats.extend([model.stats_bics_sol, model.users_sizes,
                                     model.items_sizes])
            else:
                users_sizes = list()
                items_sizes = list()
                for user, user_model in model.user_fitted_model.items():
                    if user_model is not None:
                        users_sizes.append(user_model.trainset.n_users)
                        items_sizes.append(user_model.trainset.n_items)
                    else:
                        users_sizes.append(0)
                        items_sizes.append(0)
                result_stats.extend([model.stats_bics_sol,
                                     users_sizes,
                                     items_sizes])

        else:
            result_stats.extend([" ", " ", " "])
        dump.dump(modelpath, predictions=predictions, algo=model,
                  verbose=1)
        algo = None
        model = None
        return result_stats


def run_evaluation(algo_dict, data):

    if type(data) is dataset.DatasetUserFolds:
        kf = PredefinedKFold()
        output_path = "../../Output/Models-surprise/ml-1m/"
    else:
        kf = KFold(n_splits=5)
        output_path = "../../Output/Models-surprise/ml-1m/"

    acc_results = pd.DataFrame(columns=["model", "fold", "coverage", "cov_user",
                                        "cov_item", "mae", "rmse"])
    for name, algorithm in algo_dict.items():
        for fold, (train, test) in enumerate(kf.split(data)):
            path_partition = os.path.join(output_path,
                                          "iteration"+str(fold+1))
            if not os.path.exists(path_partition):
                os.makedirs(path_partition)

            modelpath = path_partition + "/" + name + ".bpk"

            if os.path.isfile(modelpath):
                print("Loading", name, "partition", fold)
                preds, _ = dump.load(modelpath)

                all_test_users = set([pred[0] for pred in preds])
                all_test_items = set([pred[1] for pred in preds])
                filtered_preds = filter(
                    lambda x: not x[4]["was_impossible"], preds)
                preds_size = len(preds)
                preds = None
                real_preds = list(filtered_preds)
                users_considered = set([pred[0] for pred in real_preds])
                items_considered = set([pred[1] for pred in real_preds])
                cov = len(real_preds)/preds_size
                users_cov = len(users_considered)/len(all_test_users)
                items_cov = len(items_considered)/len(all_test_items)
                if len(real_preds) != 0:
                    real_mae_score = accuracy.mae(real_preds, verbose=False)
                    real_rmse_score = accuracy.rmse(real_preds, verbose=False)
                else:
                    real_mae_score = 0
                    real_rmse_score = 0
                acc_results.loc[len(acc_results)] = [name, fold+1, str(cov),
                                                     str(users_cov),
                                                     str(items_cov),
                                                     str(real_mae_score),
                                                     str(real_rmse_score)]
                acc_results.to_csv(output_path + '/acc_results.csv',
                                   header=not os.path.exists(output_path + '/acc_results.csv'),
                                   index=False)


def train_parallel(algo_dict, data):
    final_df = pd.DataFrame(columns=["model", "fold", "fit_time", "pred_time",
                                     "stats_bic1", "stats_bic2", "stats_bic3"])
    if type(data) is dataset.DatasetUserFolds:
        kf = PredefinedKFold()
        output_path = "../../Output/Models-surprise/ml-1m/"
    else:
        kf = KFold(n_splits=5)
        output_path = "../../Output/Models-surprise/ml-1m/"
    for fold, (train, test) in enumerate(kf.split(data)):
        # get number of cpus available to job
        try:
            ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except KeyError:
            ncpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(ncpus) as pool, tqdm(total=len(algo_dict.items())) as pbar:
            result_objects = [pool.apply_async(run_training_test,
                                               args=([name, algorithm, fold, train,
                                                      test, output_path]),
                                               callback=lambda _: pbar.update(1))
                              for name, algorithm in algo_dict.items()]
    
            stats_results = [r.get() for r in result_objects]
        stats_results = [x for x in stats_results if x is not None]
        df = pd.DataFrame.from_records(stats_results, columns=["model", "fold",
                                                               "fit_time",
                                                               "pred_time",
                                                               "stats_bic1",
                                                               "stats_bic2",
                                                               "stats_bic3"])
        final_df = final_df.append(df)

    stats_path = output_path + "stats_test.csv"
    final_df.to_csv(stats_path, header=not os.path.exists(stats_path),
                    mode='a', index=False)


def train_isolated(algo_dict, data):

    if type(data) is dataset.DatasetUserFolds:
        kf = PredefinedKFold()
        output_path = "../../Output/Models-surprise/ml-1m/"
    else:
        kf = KFold(n_splits=5)
        output_path = "../../Output/Models-surprise/ml-1m/"
    for fold, (train, test) in enumerate(kf.split(data)):
        for name, algorithm in tqdm(algo_dict.items()):
            stats_result = run_training_test(name, algorithm, fold, train,
                                             test, output_path)
            if stats_result is not None:
                df = pd.DataFrame.from_records([stats_result],
                                               columns=["model", "fold",
                                                        "fit_time",
                                                        "pred_time",
                                                        "stats_bic1",
                                                        "stats_bic2",
                                                        "stats_bic3"])
                stats_path = output_path + "stats_test.csv"
                df.to_csv(stats_path, header=not os.path.exists(stats_path),
                          mode='a', index=False)


def main():

    # set dataset folder
    os.environ["SURPRISE_DATA_FOLDER"] = '../../Datasets/'

    # set RNG
    np.random.seed(99)
    random.seed(99)

    data_1M = load_mvlens_1M_dataset()
    
    algo_dict = dict()

    
    param_grid_bbcf = {"number_of_nearest_bics": [1000,2000,5000,10000], "nnbrs": [20],
                        "min_num_biclusters": [100000], "min_cols": [3,5,7,10,15,20],
                        "consistency": [1], "max_overlap": [0.99]}
    grid = ParameterGrid(param_grid_bbcf)
    for params in grid:
        algo_bbcf = BBCF_nomem(params['number_of_nearest_bics'], params['nnbrs'],
                          params['min_num_biclusters'], params['min_cols'],
                          params['consistency'], params['max_overlap'])
        algo_dict[str(algo_bbcf)] = algo_bbcf

    # param_grid_usbcf = {"threshold_sim": [0.25,0.5], "nnbrs": [20],
    #                     "min_num_biclusters": [100000], "min_cols": [3, 5, 7, 10, 15, 20],
    #                     "consistency": [1], "max_overlap": [0.99]}
    # grid = ParameterGrid(param_grid_usbcf)
    # for params in grid:
    #     algo_usbcf = USBCF_nomem(params['threshold_sim'], params['nnbrs'],
    #                         params['min_num_biclusters'], params['min_cols'],
    #                         params['consistency'], params['max_overlap'])
    #     algo_dict[str(algo_usbcf)] = algo_usbcf

    # param_grid_usbcfcomb = {"threshold_sim": [0.25], "nnbrs": [20],
    #                         "min_num_biclusters": [100000], "min_cols": [[3,5,7,10,15,20]],
    #                         "consistency": [1], "max_overlap": [0.99]}
    # grid = ParameterGrid(param_grid_usbcfcomb)
    # for params in grid:
    #     algo_usbcfcomb = USBCFCombineBicSols_nomem(params['threshold_sim'], params['nnbrs'],
    #                                           params['min_num_biclusters'], params['min_cols'],
    #                                           params['consistency'], params['max_overlap'])
    #     algo_dict[str(algo_usbcfcomb)] = algo_usbcfcomb



    algo_coclust = CoClustering()
    algo_dict["Coclust"] = algo_coclust

    for i in range(0, 5):
        path_partition = os.path.join("../../Output/Models-surprise/ml-1m/",
                                      "iteration"+str(i+1))
        if not os.path.exists(path_partition):
            os.makedirs(path_partition)

    _logger.info("Running trainning and predictions")

    train_isolated(algo_dict, data_1M)
    _logger.info("Evaluating predictions")

    run_evaluation(algo_dict, data_1M)


if __name__ == '__main__':
    main()