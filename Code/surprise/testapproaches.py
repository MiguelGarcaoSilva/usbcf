from surprise import BaselineOnly, KNNWithMeans, SVD, SVDpp, NMF, CoClustering
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import dump
from surprise import dataset
from usbcf import USBCF, USBCFCombineBicSols
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


def load_mvlens_100k_dataset():
    # path to dataset folder
    files_dir = os.path.expanduser('../../Datasets/ml-100k/')

    train_file = files_dir + 'u%d.base'
    test_file = files_dir + 'u%d.test'
    folds_files = [(train_file % i, test_file % i) for i in range(1, 6)]

    data_100k = Dataset.load_from_folds(folds_files, reader=Reader('ml-100k'))

    return data_100k


def load_mvlens_1M_dataset():
    # data_1M = Dataset.load_builtin("ml-1m")

    data_1M = Dataset.load_from_file(os.environ["SURPRISE_DATA_FOLDER"] +
                                     "ml-1m/ml-1m/ratings.dat",
                                     reader=Reader('ml-1m'))
    return data_1M


def load_jester_dataset():
    # data_jester =  Dataset.load_builtin(name=u'jester')
    data_files = glob.glob(os.path.join(os.environ["SURPRISE_DATA_FOLDER"]
                                        + "jester/", "*.xls"))
    df = pd.concat(map(lambda file: pd.read_excel(file, header=None),
                   sorted(data_files)), ignore_index=True).iloc[:,  1:]
    df = df.rename_axis('user').reset_index()
    df = df.melt(id_vars=["user"])
    df.columns = ['user', 'item', 'rating']
    df = df[df.rating != 99]
    df["user"] = df["user"].apply(str)
    df["item"] = df["item"].apply(str)
    df.to_csv("jester_full.csv", sep="\t")
    data_jester = Dataset.load_from_df(df,
                                       reader=Reader(rating_scale=(-10, 10)))
    return data_jester


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
        output_path = "../../Output/Models-surprise/ml-100k/"
    else:
        kf = KFold(n_splits=5)
        output_path = "../../Output/Models-surprise/jester/"

    acc_results = pd.DataFrame(columns=["model", "fold", "cov",
                                        "mae", "rmse"])
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
                preds_size = len(preds)
                filtered_preds = filter(
                    lambda x: not x[4]["was_impossible"], preds)
                preds = None
                real_preds = list(filtered_preds)
                cov = len(real_preds)/preds_size
                if len(real_preds) != 0:
                    real_mae_score = accuracy.mae(real_preds, verbose=False)
                    real_rmse_score = accuracy.rmse(real_preds, verbose=False)
                else:
                    real_mae_score = 0
                    real_rmse_score = 0
                acc_results.loc[len(acc_results)] = [name, fold+1, str(cov),
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
        output_path = "../../Output/Models-surprise/ml-100k/"
    else:
        kf = KFold(n_splits=5)
        output_path = "../../Output/Models-surprise/jester/"
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
        output_path = "../../Output/Models-surprise/ml-100k/"
    else:
        kf = KFold(n_splits=5)
        output_path = "../../Output/Models-surprise/jester/"
    for fold, (train, test) in enumerate(kf.split(data)):
        for name, algorithm in algo_dict.items():
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

    #data_100k = load_mvlens_100k_dataset()

    data_jester = load_jester_dataset()

    # data_1M = load_mvlens_1M_dataset()

    algo_dict = dict()

    # algo_baseline = BaselineOnly()
    # algo_dict["baseline"] = algo_baseline
    # sim_options_ub = {'name': 'cosine',
    #                   'user_based': True,
    #                   'min_support': 1
    #                   }
    # sim_options_ib = {'name': 'cosine',
    #                   'user_based': False,
    #                   'min_support': 1
    #                   }
    # param_grid = {'k': [10,20,30,40,50,60,70,80,90,100]}
    # grid = ParameterGrid(param_grid)
    # for params in grid:
    #     algo_knn_ub = KNNWithMeans(k=params['k'], min_k=1,
    #                                 sim_options=sim_options_ub)
    #     algo_dict["UserBased-k="+str(params['k'])] = algo_knn_ub
    #     algo_knn_ib = KNNWithMeans(k=params['k'], min_k=1,
    #                                 sim_options=sim_options_ib)
    #     algo_dict["ItemBased-k="+str(params['k'])] = algo_knn_ib

    param_grid_bbcf = {"number_of_nearest_bics": [50,100,150,200,250,300], "nnbrs": [20],
                        "min_num_biclusters": [100000], "min_cols": [3,5,7,10,15,20],
                        "consistency": [1], "max_overlap": [1]}
    grid = ParameterGrid(param_grid_bbcf)
    for params in grid:
        algo_bbcf = BBCF_nomem(params['number_of_nearest_bics'], params['nnbrs'],
                          params['min_num_biclusters'], params['min_cols'],
                          params['consistency'], params['max_overlap'])
        algo_dict[str(algo_bbcf)] = algo_bbcf

    # param_grid_usbcf = {"threshold_sim": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], "nnbrs": [20],
    #                     "min_num_biclusters": [100000], "min_cols": [3, 5, 7, 10, 15, 20],
    #                     "consistency": [1], "max_overlap": [1]}
    # grid = ParameterGrid(param_grid_usbcf)
    # for params in grid:
    #     algo_usbcf = USBCF(params['threshold_sim'], params['nnbrs'],
    #                        params['min_num_biclusters'], params['min_cols'],
    #                        params['consistency'], params['max_overlap'])
    #     algo_dict[str(algo_usbcf)] = algo_usbcf

    # param_grid_usbcfcomb = {"threshold_sim": [0.2], "nnbrs": [20],
    #                         "min_num_biclusters": [100000], "min_cols": [[3, 5, 7, 10, 15, 20]],
    #                         "consistency": [1], "max_overlap": [1]}
    # grid = ParameterGrid(param_grid_usbcfcomb)
    # for params in grid:
    #     algo_usbcfcomb = USBCFCombineBicSols(params['threshold_sim'], params['nnbrs'],
    #                                          params['min_num_biclusters'], params['min_cols'],
    #                                          params['consistency'], params['max_overlap'])
    #     algo_dict[str(algo_usbcfcomb)] = algo_usbcfcomb

    # param_grid_svd = {'n_factors': [10,20,30,50,100,200,400],
    #                   "n_epochs": [10, 20, 50, 100],
    #                   "reg_all": [0.005, 0.01, 0.02, 0.05, 0.1]}
    # grid = ParameterGrid(param_grid_svd)
    # for params in grid:
    #     algo_svd = SVD(n_factors=params["n_factors"],
    #                     n_epochs=params["n_epochs"], reg_all=params["reg_all"])
    #     algo_dict["algo_svd-"+str(params["n_factors"])+"-"
    #               + str(params["n_epochs"])+"-"
    #               + str(params["reg_all"])] = algo_svd
    #     algo_svdpp = SVDpp(n_factors=params["n_factors"],
    #                         n_epochs=params["n_epochs"],
    #                         reg_all=params["reg_all"])
    #     algo_dict["algo_svd++-"+str(params["n_factors"])
    #               + "-"+str(params["n_epochs"])+"-"
    #               + str(params["reg_all"])] = algo_svdpp
    #     algo_nmf = NMF(n_factors=params["n_factors"],
    #                     n_epochs=params["n_epochs"], reg_pu=params["reg_all"],
    #                     reg_qi=params["reg_all"])
    #     algo_dict["algo_nmf-"+str(params["n_factors"])
    #               + "-"+str(params["n_epochs"])+"-"
    #               + str(params["reg_all"])] = algo_nmf

    algo_coclust = CoClustering()
    algo_dict["Coclust"] = algo_coclust

    for i in range(0, 5):
        path_partition = os.path.join("../../Output/Models-surprise/ml-100k/",
                                      "iteration"+str(i+1))
        if not os.path.exists(path_partition):
            os.makedirs(path_partition)
        path_partition = os.path.join("../../Output/Models-surprise/jester/",
                                      "iteration"+str(i+1))
        if not os.path.exists(path_partition):
            os.makedirs(path_partition)

    _logger.info("Running trainning and predictions")

    #train_isolated(algo_dict, data_100k)
    train_isolated(algo_dict, data_jester)
    _logger.info("Evaluating predictions")

    run_evaluation(algo_dict, data_jester)


if __name__ == '__main__':
    main()