from lenskit.algorithms import basic, als, svd, funksvd, user_knn, \
    item_knn, Recommender
from lenskit.metrics.predict import rmse, mae
from lenskit import util, batch, topn
from bbcf import BBCF, BBCF_noweight
from usbcf import USBCF_MSR, USBCF_CMR

import binpickle
import logging
import sys
import os
import pandas as pd
from sklearn.model_selection import ParameterGrid

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# util.log_to_notebook()

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler(sys.stdout))
_logger.setLevel(level=logging.INFO)


def make_models_and_predictions(algo_dict_preds_recs, algo_dict_recs_only):
    all_preds = []
    all_recs = []
    test_data = []
    for i in range(1, 6):
        _logger.info("STARTING PARTITION %s", str(i))
        train = pd.read_csv("../../Datasets/ml-100k/u%d.base" % (i, ), sep="\t",
                            names=["user", "item", "rating", "timestamp"],
                            index_col=None)
        test = pd.read_csv("../../Datasets/ml-100k/u%d.test" % (i,), sep="\t",
                           names=["user", "item", "rating", "timestamp"],
                           index_col=None)
        test_data.append(test)
        path_partition = os.path.join(
            "../Models-lenskit/ml-100k/", "iteration"+str(i))
        if not os.path.exists(path_partition):
            os.makedirs(path_partition)
        for name, algorithm in algo_dict_preds_recs.items():
            pred, recs = run_models(name, algorithm, False, train, test, i,
                                    path_partition)
            all_preds.append(pred)
            all_recs.append(recs)
        for name, algorithm in algo_dict_recs_only.items():
            pred, recs = run_models(name, algorithm, True, train, test, i,
                                    path_partition)
            all_recs.append(recs)
    all_preds = pd.concat(all_preds, ignore_index=True)
    all_recs = pd.concat(all_recs, ignore_index=True)
    return all_preds, all_recs, test_data


def run_models(aname, algo, rec_only, train, test, partition, path_partition):
    modelpath = path_partition + "/" + aname + ".bpk"
    if os.path.isfile(modelpath):
        model = binpickle.load(modelpath)
        if not rec_only:
            if hasattr(model.predictor, 'stats_biclustering_solution'):
                with open(path_partition+'/../stats_test.txt', 'a') as f:
                    f.write(aname + "," + str(partition) + ","
                            + str(model.predictor.stats_biclustering_solution)
                            + "," + str(model.predictor.stats_nearest_bics)
                            + "\n")
            pred = batch.predict(model, test)
            recs = batch.recommend(model, test.user.unique(), 20)
            _logger.debug(pred)
            _logger.debug(recs)
            pred['algorithm'], pred['partition'] = aname, partition
            recs['algorithm'], recs['partition'] = aname, partition
            return pred, recs
        else:
            recs = batch.recommend(model, test.user.unique(), 20)
            _logger.debug(recs)
            recs['algorithm'], recs['partition'] = aname, partition
            return None, recs
    else:
        fittable = Recommender.adapt(algo)
        model_persist_obj = batch.train_isolated(fittable, train)
        model = model_persist_obj.get()
        _logger.info("Model %s fitted, saving as binpikle", modelpath)
        binpickle.dump(model, modelpath)
        if not rec_only:
            if hasattr(model.predictor, 'stats_biclustering_solution'):
                with open(path_partition+'/../stats_test.txt', 'a') as f:
                    f.write(aname + "," + str(partition) + ","
                            + str(model.predictor.stats_biclustering_solution)
                            + "," + str(model.predictor.stats_nearest_bics)
                            + "\n")
            pred = batch.predict(model, test)
            recs = batch.recommend(model, test.user.unique(), 20)
            _logger.debug(pred)
            model_persist_obj.close()
            pred['algorithm'], pred['partition'] = aname, partition
            recs['algorithm'], recs['partition'] = aname, partition
            return pred, recs
        else:
            recs = batch.recommend(model, test.user.unique(), 20)
            _logger.debug(recs)
            model_persist_obj.close()
            recs['algorithm'], recs['partition'] = aname, partition
            return None, recs


def eval_coverage(preds):
    results_cov = dict()
    df_result = pd.DataFrame(columns=["algorithm", "nans_iteration_list",
                                      "coverage"])
    grouping = preds.groupby(["partition", "algorithm"])
    for group, preds_group in grouping:
        nans = preds_group.prediction.isna().sum()
        rows = len(preds_group.prediction)
        results_cov.setdefault(group[1], []).append((nans, rows))
    for algorithm in results_cov:
        nans_algo_list = results_cov[algorithm]
        total_nans_algo = 0
        total_rows_algo = 0
        for partition in nans_algo_list:
            total_nans_algo += partition[0]
            total_rows_algo += partition[1]
        df_result = df_result.append({"algorithm": algorithm,
                                      "nans_iteration_list": nans_algo_list,
                                      "coverage": 1
                                      - (total_nans_algo/total_rows_algo)},
                                     ignore_index=True)
    return df_result


def run_evaluation_predictions(all_preds, group_cols):
    results = pd.DataFrame(columns=group_cols + ["rmse", "mae"])
    for cols_values, group in all_preds.groupby(group_cols):
        cols_values = [cols_values] if isinstance(cols_values, int) \
            else cols_values
        results = results.append(dict(dict(zip(group_cols, cols_values)),
                                      **{"rmse": rmse(group.prediction,
                                                      group.rating,
                                                      missing="ignore"),
                                         "mae": mae(group.prediction,
                                                    group.rating,
                                                    missing="ignore")}),
                                 ignore_index=True)
    return results


def run_evaluation_recommendations(all_recs, test_data, group_cols):
    test_data = pd.concat(test_data, ignore_index=True)
    rla = topn.RecListAnalysis(group_cols=group_cols)
    rla.add_metric(topn.recall)
    rla.add_metric(topn.precision)
    rla.add_metric(topn.ndcg)
    results = rla.compute(all_recs, test_data)
    results['f1'] = results.apply(lambda row: (2*row.precision*row.recall) /
                                  (row.precision+row.recall), axis=1)
    return results


def main():

    random_seed = util.init_rng(99)

    algo_dict_preds_recs = dict()
    algo_dict_recs_only = dict()
    # algo_random = basic.Random(rng_spec=random_seed)
    # algo_dict_recs_only["Basic-random"] = algo_random
    # algo_popular = basic.Popular()
    # algo_dict_recs_only["Basic-popular"] = algo_popular
    algo_basic = basic.Bias()
    algo_dict_preds_recs["Basic-bias"] = algo_basic

    param_grid_bbcf = {"number_of_nearest_bics": [50], "nnbrs": [20],
                       "min_num_biclusters": [100000], "min_cols": [2],
                       "consistency": [1], "max_overlap": [1]}

    grid = ParameterGrid(param_grid_bbcf)
    for params in grid:
        algo = BBCF(params['number_of_nearest_bics'], params['nnbrs'],
                    params['min_num_biclusters'], params['min_cols'],
                    params['consistency'], params['max_overlap'])
        algo_dict_preds_recs[str(algo)] = algo
        # algo = USBCF_MSR(params['number_of_nearest_bics'],
        #                  params['nnbrs'],
        #                  params['min_num_biclusters'],
        #                  params['min_cols'], params['consistency'],
        #                  params['max_overlap'])
        # algo_dict_preds_recs[str(algo)] = algo

    path_exp = "../Results-lenskit/ml-100k.exp/"
    _logger.info("Creating models and making predictions... %s", path_exp)
    all_preds_exp, all_recs_exp, test_data = make_models_and_predictions(
        algo_dict_preds_recs, algo_dict_recs_only)
    if not os.path.exists(os.path.join(path_exp, "preds_recs")):
        os.makedirs(os.path.join(path_exp, "preds_recs"))
    all_preds_exp.to_csv(path_exp + "preds_recs/allpreds.csv")
    all_recs_exp.to_csv(path_exp + "preds_recs/allrecs.csv")

    _logger.info("Running Evaluation predictions... %s", path_exp)
    results_preds_eval_perpartition = run_evaluation_predictions(
        all_preds_exp, ["partition", "algorithm"])

    if not os.path.exists(os.path.join(path_exp, "results")):
        os.makedirs(os.path.join(path_exp, "results"))

    results_preds_eval_perpartition.to_csv(path_exp
                                           + "results/preds_eval_perpartition_byalgopartition.csv")
    results_preds_eval_perpartition.groupby("algorithm")[["rmse", "mae"]].mean(
    ).to_csv(path_exp+"results/preds_eval_perpartition_byalgo_mean.csv")
    results_preds_eval_perpartition.groupby("algorithm")[["rmse", "mae"]].std(
    ).to_csv(path_exp+"results/preds_eval_perpartition_byalgo_std.csv")

    cov_results_rows = eval_coverage(all_preds_exp)
    cov_results_rows.to_csv(path_exp+'results/coverage.txt')

    results_recs_eval = run_evaluation_recommendations(
        all_recs_exp, test_data, None)

    results_recs_eval.to_csv(path_exp+"results/recs_algopartitionuser.csv")
    results_recs_eval.groupby(["algorithm", "partition"])[["precision", "recall", "f1", "ndcg"]].mean().to_csv(path_exp
                                                                                                               + "results/recs_byalgopartition.csv")
    results_recs_eval.groupby(["algorithm", "partition"])[["precision", "recall", "f1", "ndcg"]].mean(
    ).groupby("algorithm").mean().to_csv(path_exp+"results/recs_byalgo_mean.csv")
    results_recs_eval.groupby(["algorithm", "partition"])[["precision", "recall", "f1", "ndcg"]].mean(
    ).groupby("algorithm").std().to_csv(path_exp+"results/recs_byalgo_std.csv")


if __name__ == '__main__':
    main()
