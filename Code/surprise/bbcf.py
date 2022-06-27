from surprise import AlgoBase, KNNWithMeans
from biclustering.qubic import QUBIC2, QUBIC
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from surprise import PredictionImpossible, Prediction
from surprise import Dataset, Reader
from tqdm import tqdm

import numpy as np
import pandas as pd
import heapq
import sys
import logging
import multiprocessing
import copy
import math
import statistics
import pickle
import os


_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler(sys.stdout))
_logger.setLevel(level=logging.INFO)


class BBCF(AlgoBase):

    def __init__(self, number_of_nearest_bics=5, nnbrs=20, num_biclusters=100,
                 min_cols=2, consistency=1, max_overlap=1):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

        # bbcf
        self.nnbics = number_of_nearest_bics

        # biclustering
        self.bic_algo = QUBIC2(num_biclusters=num_biclusters,
                               discreteFlag=True, minCols=min_cols,
                               consistency=consistency,
                               max_overlap_level=max_overlap)
        # ibknn
        sim_options = {'name': 'pearson',
                       'user_based': False,
                       'min_support': 1
                       }
        self.cf_algo = KNNWithMeans(k=nnbrs, sim_options=sim_options,
                                    verbose=False)

        self.rating_matrix_csr = None

        # users model dict
        self.user_fitted_model = dict()

        # extra stats
        self.stats_bics_sol = list()

    def fit(self, trainset):
        _logger.info('Fitting %s', str(self))

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        row_ind, col_ind, vals = [], [], []

        for (u, i, r) in self.trainset.all_ratings():
            row_ind.append(u)
            col_ind.append(i)
            if r == 0:
                r = 99
            vals.append(r)

        self.rating_matrix_csr = csr_matrix((vals, (row_ind, col_ind)),
                                            shape=(self.trainset.n_users,
                                                   self.trainset.n_items))

        rating_matrix_dense = np.zeros([max(row_ind)+1, max(col_ind)+1])
        rating_matrix_dense[row_ind, col_ind] = vals
        rating_matrix_dense[rating_matrix_dense == 0] = np.nan
        rating_matrix_dense[rating_matrix_dense == 99] = 0
        # Preprocessing for biclustering

        # discretization
        if not all([isinstance(value, int) or value.is_integer()
                    for value in vals]):
            mask = (rating_matrix_dense >= 0)
            rating_matrix_dense_rounded = np.empty_like(rating_matrix_dense)
            rating_matrix_dense_rounded[mask] = np.floor(
                rating_matrix_dense[mask] + 0.5)
            rating_matrix_dense_rounded[~mask] = np.ceil(
                rating_matrix_dense[~mask] - 0.5)
            rating_matrix_dense = rating_matrix_dense_rounded

        # workaround so that np.nan can be represented by "0" if needed
        rating_matrix_dense += 100
        rating_matrix_dense = np.nan_to_num(
            rating_matrix_dense, nan=0)

        _logger.info('running biclustering')
                        
        str_trainset = hash(tuple([(u,i,r) for u,i,r in trainset.all_ratings()]))
        bic_sol_path = "../../Output/Models-surprise/bicsols/"+ str(str_trainset) + "/" 
        if not os.path.exists(bic_sol_path):
                os.makedirs(bic_sol_path)
        bic_sol_path = bic_sol_path + str(self.bic_algo)+ ".pkl"
        if os.path.isfile(bic_sol_path):
            _logger.info('using precomputed biclustering solution')
            with open(bic_sol_path,"rb") as f:
                biclustering_solution = pickle.load(f)
        else:
            _logger.info('computing biclustering solution')
            # Generate bics
            bic_sol = self.bic_algo.run(rating_matrix_dense)
            # P-Value calculation
            # biclustering_solution.run_constant_freq_column(rating_matrix_dense_rounded,
            #                                                list(range(self.trainset.rating_scale[0],
            #                                                           self.trainset.rating_scale[1]+1)),
            #                                                True)
            biclustering_solution = bic_sol.biclusters
            with open(bic_sol_path,"wb") as f:
                pickle.dump(bic_sol.biclusters, f)
        if len(biclustering_solution) < 1:
            _logger.info('biclustering failed to find biclusters')
            return None

        rows_sizes, cols_sizes = list(), list()
        for bic in biclustering_solution:
            rows_sizes.append(len(bic.rows))
            cols_sizes.append(len(bic.cols))
        self.stats_bics_sol.append([len(biclustering_solution),
                                    statistics.mean(rows_sizes),
                                    statistics.pstdev(rows_sizes),
                                    statistics.mean(cols_sizes),
                                    statistics.pstdev(cols_sizes)])

        _logger.info('biclustering completed: %s', self.stats_bics_sol)

        # Calculate similarities and train models
        _logger.info("calculating similarities and models")
        # get number of cpus available to job
        try:
            ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except KeyError:
            ncpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(ncpus) as pool, tqdm(total=self.trainset.n_users) as pbar:
            result_objects = [pool.apply_async(self.obtain_user_bics_sims,
                                               args=([inner_uid, biclustering_solution]),
                                               callback=lambda _: pbar.update(1))
                              for inner_uid in self.trainset.all_users()]
            self.user_fitted_model = dict([r.get() for r in result_objects])
        return self

    def estimate(self, inner_uid, inner_iid):
        # user or item not in the system
        if not (self.trainset.knows_user(inner_uid)
                and self.trainset.knows_item(inner_iid)):
            raise PredictionImpossible('User and/or item is unknown.')

        # find user model
        raw_uid = self.trainset.to_raw_uid(inner_uid)
        raw_iid = self.trainset.to_raw_iid(inner_iid)
        # use user-specific model to predict
        prediction = self.user_fitted_model[inner_uid].predict(
            raw_uid, raw_iid)
        if prediction[4]["was_impossible"]:
            raise PredictionImpossible(
                'User and/or item is unknown (for the cf algorithm).')

        return prediction[3], prediction[4]

    def obtain_user_bics_sims(self, inner_uid, bics_sol):
        user_sims = []
        # item rated by active user
        user_ratings = self.rating_matrix_csr.getrow(
            inner_uid).toarray().ravel()
        user_items_indexes = np.flatnonzero(user_ratings).tolist()
        for bic in bics_sol:
            users_indexes_bic = bic.rows
            items_indexes_bic = bic.cols
            # sim computation
            sim_u_b = len(list(set(user_items_indexes) & set(
                items_indexes_bic))) / len(set(items_indexes_bic))
            weight_u_b = sim_u_b * len(users_indexes_bic)
            user_sims.append(weight_u_b)

        nearest_bics_indexes = heapq.nlargest(self.nnbics,
                                              range(len(user_sims)),
                                              key=lambda k: user_sims[k])

        # User-specific U-I matrix todo alterar
        bic_result_rows = set()
        bic_result_cols = set()
        for bic_index in nearest_bics_indexes:
            bic_result_rows.update(bics_sol[bic_index].rows)
            bic_result_cols.update(bics_sol[bic_index].cols)


        _logger.debug("New user %d matrix dims: %d %d",
                      inner_uid, len(bic_result_rows), len(bic_result_cols))

        df = self.bicluster_to_df([sorted(bic_result_rows),
                                   sorted(bic_result_cols)])

        user_model = self.fit_user_model(inner_uid, df)

        return (inner_uid, user_model)

    def fit_user_model(self, inner_uid, user_k_nearest_bic):
        df = user_k_nearest_bic
        data = Dataset.load_from_df(df[['user', 'item', 'rating']],
                                    reader=Reader(rating_scale=self.trainset.rating_scale))
        trainset = data.build_full_trainset()
        algo = copy.deepcopy(self.cf_algo)
        user_model = algo.fit(trainset)
        _logger.debug("Trainned model for inner user %d", inner_uid)
        return user_model

    def bicluster_to_df(self, bicluster):
        rows_bicluster = bicluster[0]
        cols_bicluster = bicluster[1]
        list_bic = []
        for row_idx in rows_bicluster:
            for col_idx in cols_bicluster:
                user = self.trainset.to_raw_uid(row_idx)
                item = self.trainset.to_raw_iid(col_idx)
                rating = dict(self.trainset.ur[row_idx]).get(col_idx, 0)
                if rating != 0:
                    list_bic.extend(
                        [{"user": user, "item": item, "rating": rating}])
        df_bicluster = pd.DataFrame(list_bic)
        return df_bicluster

    def __str__(self):
        return 'BBCF({},{})'.format(self.nnbics, self.bic_algo)


class BBCF_nomem(AlgoBase):

    trainset = None
    biclustering_solution = None
    rating_matrix_csr = None

    def __init__(self, number_of_nearest_bics=5, nnbrs=20, num_biclusters=1000,
                 min_cols=2, consistency=1, max_overlap=1):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

        # bbcf
        self.nnbics = number_of_nearest_bics

        # biclustering
        self.bic_algo = QUBIC2(num_biclusters=num_biclusters,
                                discreteFlag=True, minCols=min_cols,
                                consistency=consistency,
                                max_overlap_level=max_overlap)

        # ibknn
        self.sim_options = {'name': 'pearson',
                            'user_based': False,
                            'min_support': 1
                            }
        self.nnbrs = nnbrs
        self.cf_algo = KNNWithMeans(k=self.nnbrs, sim_options=self.sim_options,
                                verbose=False)

        # extra stats
        self.stats_bics_sol = list()
        self.users_sizes = list()
        self.items_sizes = list()

    def fit(self, trainset):
        _logger.info('Fitting %s', str(self))

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)
        self.trainset = None
        BBCF_nomem.trainset = trainset

        row_ind, col_ind, vals = [], [], []

        for (u, i, r) in BBCF_nomem.trainset.all_ratings():
            row_ind.append(u)
            col_ind.append(i)
            if r == 0:
                r = 99
            vals.append(r)

        BBCF_nomem.rating_matrix_csr = csr_matrix((vals, (row_ind, col_ind)),
                                                  shape=(BBCF_nomem.trainset.n_users,
                                                  BBCF_nomem.trainset.n_items))

        rating_matrix_dense = np.zeros([max(row_ind)+1, max(col_ind)+1])
        rating_matrix_dense[row_ind, col_ind] = vals
        rating_matrix_dense[rating_matrix_dense == 0] = np.nan
        rating_matrix_dense[rating_matrix_dense == 99] = 0
        # Preprocessing for biclustering

        # discretization
        if not all([isinstance(value, int) or value.is_integer()
                    for value in vals]):
            mask = (rating_matrix_dense >= 0)
            rating_matrix_dense_rounded = np.empty_like(rating_matrix_dense)
            rating_matrix_dense_rounded[mask] = np.floor(
                rating_matrix_dense[mask] + 0.5)
            rating_matrix_dense_rounded[~mask] = np.ceil(
                rating_matrix_dense[~mask] - 0.5)
            rating_matrix_dense = rating_matrix_dense_rounded

        # workaround so that np.nan can be represented by "0" if needed
        rating_matrix_dense += 100
        rating_matrix_dense = np.nan_to_num(
            rating_matrix_dense, nan=0)

        _logger.info('running biclustering')
        str_trainset = hash(tuple([(u,i,r) for u,i,r in trainset.all_ratings()]))
        bic_sol_path = "../../Output/Models-surprise/bicsols/"+ str(str_trainset) + "/" 
        if not os.path.exists(bic_sol_path):
                os.makedirs(bic_sol_path)
        bic_sol_path = bic_sol_path + str(self.bic_algo)+ ".pkl"
        if os.path.isfile(bic_sol_path):
            _logger.info('using precomputed biclustering solution')
            with open(bic_sol_path,"rb") as f:
                BBCF_nomem.biclustering_solution = pickle.load(f)
        else:
            _logger.info('computing biclustering solution')
            # Generate bics
            bic_sol = self.bic_algo.run(rating_matrix_dense)
            # P-Value calculation
            # biclustering_solution.run_constant_freq_column(rating_matrix_dense_rounded,
            #                                                list(range(self.trainset.rating_scale[0],
            #                                                           self.trainset.rating_scale[1]+1)),
            #                                                True)
            BBCF_nomem.biclustering_solution = bic_sol.biclusters
            with open(bic_sol_path,"wb") as f:
                pickle.dump(bic_sol.biclusters, f)

        if len(BBCF_nomem.biclustering_solution) < 1:
            _logger.info('biclustering failed to find biclusters')
            return None

        rows_sizes, cols_sizes = list(), list()
        for bic in BBCF_nomem.biclustering_solution:
            rows_sizes.append(len(bic.rows))
            cols_sizes.append(len(bic.cols))
        self.stats_bics_sol.append([len(BBCF_nomem.biclustering_solution),
                                    statistics.mean(rows_sizes),
                                    statistics.pstdev(rows_sizes),
                                    statistics.mean(cols_sizes),
                                    statistics.pstdev(cols_sizes)])

        _logger.info('biclustering completed: %s', self.stats_bics_sol)
        return self

    def obtain_user_bics_sims(self, inner_uid):
        user_sims = []
        # item rated by active user
        user_ratings = BBCF_nomem.rating_matrix_csr.getrow(
            inner_uid).toarray().ravel()
        user_items_indexes = np.flatnonzero(user_ratings).tolist()
        for bic in BBCF_nomem.biclustering_solution:
            users_indexes_bic = bic.rows
            items_indexes_bic = bic.cols
            # sim computation
            sim_u_b = len(list(set(user_items_indexes) & set(
                items_indexes_bic))) / len(set(items_indexes_bic))
            weight_u_b = sim_u_b * len(users_indexes_bic)
            user_sims.append(weight_u_b)

        nearest_bics_indexes = heapq.nlargest(self.nnbics,
                                              range(len(user_sims)),
                                              key=lambda k: user_sims[k])

        # User-specific U-I matrix todo alterar
        bic_result_rows = set()
        bic_result_cols = set()
        for bic_index in nearest_bics_indexes:
            bic_result_rows.update(BBCF_nomem.biclustering_solution[bic_index].rows)
            bic_result_cols.update(BBCF_nomem.biclustering_solution[bic_index].cols)

        _logger.debug("New user %d matrix dims: %d %d",
                      inner_uid, len(bic_result_rows), len(bic_result_cols))

        df = self.bicluster_to_df([sorted(bic_result_rows),
                                   sorted(bic_result_cols)])
        user_model = self.fit_user_model(inner_uid, df)
        
        return (inner_uid, user_model, len(bic_result_rows), len(bic_result_cols))

    def fit_user_model(self, inner_uid, user_k_nearest_bic):
        df = user_k_nearest_bic
        data = Dataset.load_from_df(df[['user', 'item', 'rating']],
                                    reader=Reader(
                                        rating_scale=BBCF_nomem.trainset.rating_scale))
        trainset = data.build_full_trainset()
        algo = copy.deepcopy(self.cf_algo)
        user_model = algo.fit(trainset)
        return user_model

    def bicluster_to_df(self, bicluster):
        rows_bicluster = bicluster[0]
        cols_bicluster = bicluster[1]
        list_bic = []
        for row_idx in rows_bicluster:
            for col_idx in cols_bicluster:
                user = BBCF_nomem.trainset.to_raw_uid(row_idx)
                item = BBCF_nomem.trainset.to_raw_iid(col_idx)
                rating = dict(BBCF_nomem.trainset.ur[row_idx]).get(col_idx, 0)
                if rating != 0:
                    list_bic.extend(
                        [{"user": user, "item": item, "rating": rating}])
        df_bicluster = pd.DataFrame(list_bic)
        return df_bicluster

    # override to be faster
    # def test(self, testset, verbose=False):
    #     """Test the algorithm on given testset, i.e. estimate all the ratings
    #     in the given testset.
    #     Args:
    #         testset: A test set, as returned by a :ref:`cross-validation
    #             itertor<use_cross_validation_iterators>` or by the
    #             :meth:`build_testset() <surprise.Trainset.build_testset>`
    #             method.
    #         verbose(bool): Whether to print details for each predictions.
    #             Default is False.
    #     Returns:
    #         A list of :class:`Prediction\
    #         <surprise.prediction_algorithms.predictions.Prediction>` objects
    #         that contains all the estimated ratings.
    #     """
    #     testset = pd.DataFrame(testset, columns = ["uid","iid","r_ui_trans"])
    #     predictions = []
    #     for name, group in testset.groupby(["uid"]):
    #         #modelo do user
    #         iuid = BBCF_nomem.trainset.to_inner_uid(name)
    #         (_, usermodel) = self.obtain_user_bics_sims(iuid)
    #         user_predictions = [self.predict(uid,
    #                                     iid,
    #                                     r_ui_trans,
    #                                     usermodel,
    #                                     verbose=verbose)
    #                         for (uid, iid, r_ui_trans) in group.values.tolist()]
    #         predictions.extend(user_predictions)
    #         usermodel = None
    #     return predictions

    # override to be faster with multiprocessing
    def test(self, testset, verbose=False):
        """Test the algorithm on given testset, i.e. estimate all the ratings
        in the given testset.
        Args:
            testset: A test set, as returned by a :ref:`cross-validation
                itertor<use_cross_validation_iterators>` or by the
                :meth:`build_testset() <surprise.Trainset.build_testset>`
                method.
            verbose(bool): Whether to print details for each predictions.
                Default is False.
        Returns:
            A list of :class:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` objects
            that contains all the estimated ratings.
        """
        # get number of cpus available to job
        try:
            ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except KeyError:
            ncpus = multiprocessing.cpu_count()
        testset = pd.DataFrame(testset, columns=["uid", "iid", "r_ui_trans"])
        self.predictions = []
        groupping = testset.groupby(["uid"])
        with multiprocessing.Pool(ncpus) as pool, tqdm(total=len(groupping)) as pbar:
            result_objects = [pool.apply_async(self.predict_for_user_tests,
                                               args=([uid, group, verbose]),
                                               callback=lambda _: pbar.update(1))
                              for uid, group in groupping]
            stats_results = [r.get() for r in result_objects]
        for preds, sizes in stats_results:
            self.predictions.extend(preds)
            self.users_sizes.append(sizes[0])
            self.items_sizes.append(sizes[1])
        return self.predictions


    def predict_for_user_tests(self, uid, group, verbose):
        # modelo do user
        iuid = BBCF_nomem.trainset.to_inner_uid(uid)
        (_, usermodel, user_size, item_size) = self.obtain_user_bics_sims(iuid)

        user_predictions = [self.predict(uid,
                                         iid,
                                         r_ui_trans,
                                         usermodel)
                            for (uid, iid, r_ui_trans) in group.values.tolist()]
        return user_predictions, (user_size, item_size)

    def predict(self, uid, iid, r_ui=None, user_model=None, clip=True,
                verbose=False):
        # Convert raw ids to inner ids
        try:
            iuid = BBCF_nomem.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = BBCF_nomem.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:
            est = self.estimate(iuid, iiid, user_model)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = BBCF_nomem.trainset.global_mean
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = BBCF_nomem.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def estimate(self, inner_uid, inner_iid, user_model):
        # user or item not in the system
        if not (BBCF_nomem.trainset.knows_user(inner_uid)
                and BBCF_nomem.trainset.knows_item(inner_iid)):
            raise PredictionImpossible('User and/or item is unknown.')

        # create user model
        # (_, user_model) = self.obtain_user_bics_sims(inner_uid)
        raw_uid = BBCF_nomem.trainset.to_raw_uid(inner_uid)
        raw_iid = BBCF_nomem.trainset.to_raw_iid(inner_iid)
        # use user-specific model to predict
        prediction = user_model.predict(raw_uid, raw_iid)

        if prediction[4]["was_impossible"]:
            raise PredictionImpossible(
                'User and/or item is unknown (for the cf algorithm).')

        return prediction[3], prediction[4]

    def __str__(self):
        return 'BBCF_nomem({},{},KNNMeans,{},{})'.format(self.nnbics,self.nnbrs,
                                                self.sim_options, self.bic_algo)
