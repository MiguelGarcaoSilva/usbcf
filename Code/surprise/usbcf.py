from surprise import AlgoBase, KNNWithMeans
from biclustering.qubic import QUBIC2
from biclustering.bicluster import Biclustering
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from surprise import PredictionImpossible, Prediction
from surprise import Dataset, Reader
from tqdm import tqdm

import numpy as np
import pandas as pd
import sys
import logging
import multiprocessing
import copy
import statistics
import pickle
import os

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler(sys.stdout))
_logger.setLevel(level=logging.INFO)


class USBCF(AlgoBase):

    def __init__(self, threshold_sim=0.5, nnbrs=20, num_biclusters=100,
                 min_cols=2, consistency=1, max_overlap=1):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

        # bbcf
        self.threshold_sim = threshold_sim

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

        # Generate bics - not saved as attribute to save space
        biclustering_solution = self.bic_algo.run(rating_matrix_dense)
        # P-Value calculation
        # biclustering_solution.run_constant_freq_column(rating_matrix_dense,
        #                                                list(range(self.trainset.rating_scale[0],
        #                                                           self.trainset.rating_scale[1]+1)),
        #                                                True)
        biclustering_solution = biclustering_solution.biclusters

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
        pool = multiprocessing.Pool(ncpus-1)
        result_objects = [pool.apply_async(self.obtain_user_bics_sims,
                                           args=([inner_uid, biclustering_solution]))
                          for inner_uid in self.trainset.all_users()]

        self.user_fitted_model = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

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
        if self.user_fitted_model[inner_uid] != None:
            prediction = self.user_fitted_model[inner_uid].predict(
                raw_uid, raw_iid)
        else:
            details = {}
            details['was_impossible'] = True
            details['reason'] = "No data to build algorithm"
            prediction = Prediction(raw_uid, raw_iid, None, None, details)
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
            users_indexes_bic = sorted(bic.rows)
            items_indexes_bic = sorted(bic.cols)

            items_indexes_interception = sorted(
                set(user_items_indexes) & set(items_indexes_bic))

            if len(items_indexes_interception) == 0:
                user_sims.append(0)
                continue

            # Missingness similarity
            sim_match = len(
                items_indexes_interception) / len(set(items_indexes_bic))

            # Rating deviation similarity
            matrix_bic = self.rating_matrix_csr[np.ix_(users_indexes_bic,
                                                       items_indexes_bic)].todense()

            row_user_interception = [user_ratings[index] for index in
                                     items_indexes_bic]

            new_matrix = matrix_bic[:, np.nonzero(row_user_interception)[0]]

            rating_dev = mean_squared_error(np.asarray(new_matrix[0]).ravel(),
                                            user_ratings[items_indexes_interception],
                                            squared=False)

            rating_deviation = rating_dev/(self.trainset.rating_scale[1]
                                           - self.trainset.rating_scale[0])
            sim_fit = 1-rating_deviation

            # # sim_significance
            # if bic.pvalue != 0:
            #     pvalue_exp = abs(math.log10(bic.pvalue))
            #     sig_rate = 1.05
            #     sim_significance = (pow(sig_rate, - pvalue_exp) -
            #                         pow(sig_rate, 0)) / (pow(sig_rate, -100) -
            #                                              pow(sig_rate, 0))
            # else:
            #     sim_significance = 1
            # _logger.debug("sim_significance", sim_significance)

            sim = sim_match * sim_fit

            user_sims.append(sim)

        nearest_bics_indexes = [i for i, sim in enumerate(user_sims)
                                if sim > self.threshold_sim]
        if len(nearest_bics_indexes) == 0:
            return (inner_uid, None)

        # User-specific U-I matrix
        bic_result_rows = set([inner_uid])
        bic_result_cols = set()
        for bic_index in nearest_bics_indexes:
            bic_result_rows.update(bics_sol[bic_index].rows)
            bic_result_cols.update(bics_sol[bic_index].cols)

        if inner_uid % 100 == 0:
            _logger.info("New user %d matrix dims: %d %d",
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
        if inner_uid % 100 == 0:
            _logger.info("Trainned model for inner user %d", inner_uid)
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
        return 'USBCF({},{})'.format(self.threshold_sim, self.bic_algo)


class USBCF_nomem(AlgoBase):

    trainset = None
    biclustering_solution = None
    rating_matrix_csr = None

    def __init__(self, threshold_sim=0.5, nnbrs=20, num_biclusters=100,
                 min_cols=2, consistency=1, max_overlap=1):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

        # bbcf
        self.threshold_sim = threshold_sim

        # biclustering
        self.min_cols = min_cols
        self.bic_algo = QUBIC2(num_biclusters=num_biclusters,
                               discreteFlag=True, minCols=self.min_cols,
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
        USBCF_nomem.trainset = trainset

        row_ind, col_ind, vals = [], [], []

        for (u, i, r) in USBCF_nomem.trainset.all_ratings():
            row_ind.append(u)
            col_ind.append(i)
            if r == 0:
                r = 99
            vals.append(r)

        USBCF_nomem.rating_matrix_csr = csr_matrix((vals, (row_ind, col_ind)),
                                                   shape=(USBCF_nomem.trainset.n_users,
                                                          USBCF_nomem.trainset.n_items))

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
        str_trainset = hash(tuple([(u, i, r)
                            for u, i, r in trainset.all_ratings()]))
        bic_sol_path = "../../Output/Models-surprise/bicsols/" + \
            str(str_trainset) + "/"
        if not os.path.exists(bic_sol_path):
            os.makedirs(bic_sol_path)
        bic_sol_path = bic_sol_path + str(self.bic_algo) + ".pkl"
        if os.path.isfile(bic_sol_path):
            _logger.info('using precomputed biclustering solution')
            with open(bic_sol_path, "rb") as f:
                USBCF_nomem.biclustering_solution = pickle.load(f)
        else:
            _logger.info('computing biclustering solution')
            # Generate bics
            bic_sol = self.bic_algo.run(rating_matrix_dense)
            # P-Value calculation
            # biclustering_solution.run_constant_freq_column(rating_matrix_dense_rounded,
            #                                                list(range(self.trainset.rating_scale[0],
            #                                                           self.trainset.rating_scale[1]+1)),
            #                                                True)
            USBCF_nomem.biclustering_solution = bic_sol.biclusters
            with open(bic_sol_path, "wb") as f:
                pickle.dump(bic_sol.biclusters, f)

        if len(USBCF_nomem.biclustering_solution) < 1:
            _logger.info('biclustering failed to find biclusters')
            return None

        rows_sizes, cols_sizes = list(), list()
        for bic in USBCF_nomem.biclustering_solution:
            rows_sizes.append(len(bic.rows))
            cols_sizes.append(len(bic.cols))
        self.stats_bics_sol.append([len(USBCF_nomem.biclustering_solution),
                                    statistics.mean(rows_sizes),
                                    statistics.pstdev(rows_sizes),
                                    statistics.mean(cols_sizes),
                                    statistics.pstdev(cols_sizes)])

        _logger.info('biclustering completed: %s', self.stats_bics_sol)
        return self

    def estimate(self, inner_uid, inner_iid, user_model):
        # user or item not in the system
        if not (USBCF_nomem.trainset.knows_user(inner_uid)
                and USBCF_nomem.trainset.knows_item(inner_iid)):
            raise PredictionImpossible('User and/or item is unknown.')

        # find user model
        raw_uid = USBCF_nomem.trainset.to_raw_uid(inner_uid)
        raw_iid = USBCF_nomem.trainset.to_raw_iid(inner_iid)

        # use user-specific model to predict
        if user_model is not None:
            prediction = user_model.predict(raw_uid, raw_iid)
        else:
            details = {}
            details['was_impossible'] = True
            details['reason'] = "No data to build algorithm"
            prediction = Prediction(raw_uid, raw_iid, None, None, details)
        if prediction[4]["was_impossible"]:
            raise PredictionImpossible(
                'User and/or item is unknown (for the cf algorithm).')

        return prediction[3], prediction[4]

    def obtain_user_bics_sims(self, inner_uid):
        user_sims = []
        # item rated by active user
        user_ratings = USBCF_nomem.rating_matrix_csr.getrow(
            inner_uid).toarray().ravel()
        user_items_indexes = np.flatnonzero(user_ratings).tolist()

        for bic in USBCF_nomem.biclustering_solution:
            users_indexes_bic = sorted(bic.rows)
            items_indexes_bic = sorted(bic.cols)

            items_indexes_interception = sorted(
                set(user_items_indexes).intersection(set(items_indexes_bic)))

            if len(items_indexes_interception) == 0:
                user_sims.append(0)
                continue

            # Missingness similarity
            sim_match = len(
                items_indexes_interception) / len(set(items_indexes_bic))

            # Rating deviation similarity
            matrix_bic = USBCF_nomem.rating_matrix_csr[np.ix_(users_indexes_bic,
                                                       items_indexes_bic)].todense()

            row_user_interception = [user_ratings[index] for index in
                                     items_indexes_bic]

            new_matrix = matrix_bic[:, np.nonzero(row_user_interception)[0]]

            rating_dev = mean_squared_error(np.asarray(new_matrix[0]).ravel(),
                                            user_ratings[items_indexes_interception],
                                            squared=False)

            rating_deviation = rating_dev/(USBCF_nomem.trainset.rating_scale[1]
                                           - USBCF_nomem.trainset.rating_scale[0])
            sim_fit = 1-rating_deviation

            sim = sim_match * sim_fit

            user_sims.append(sim)

        nearest_bics_indexes = [i for i, sim in enumerate(user_sims)
                                if sim > self.threshold_sim]

        if not nearest_bics_indexes:
            return (inner_uid, None, 0, 0)

        # User-specific U-I matrix
        bic_result_rows = set([inner_uid])
        bic_result_cols = set()
        for bic_index in nearest_bics_indexes:
            bic_result_rows.update(
                USBCF_nomem.biclustering_solution[bic_index].rows)
            bic_result_cols.update(
                USBCF_nomem.biclustering_solution[bic_index].cols)


        df = self.bicluster_to_df([sorted(bic_result_rows),
                                   sorted(bic_result_cols)])

        user_model = self.fit_user_model(inner_uid, df)

        return (inner_uid, user_model, len(bic_result_rows), len(bic_result_cols))

    def fit_user_model(self, inner_uid, user_k_nearest_bic):
        df = user_k_nearest_bic
        data = Dataset.load_from_df(df[['user', 'item', 'rating']],
                                    reader=Reader(
                                        rating_scale=USBCF_nomem.trainset.rating_scale))
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
                user = USBCF_nomem.trainset.to_raw_uid(row_idx)
                item = USBCF_nomem.trainset.to_raw_iid(col_idx)
                rating = dict(USBCF_nomem.trainset.ur[row_idx]).get(col_idx, 0)
                if rating != 0:
                    list_bic.extend(
                        [{"user": user, "item": item, "rating": rating}])
        df_bicluster = pd.DataFrame(list_bic)
        return df_bicluster

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
        groupping = testset.groupby(["uid"])
        predictions = []
        with multiprocessing.Pool(ncpus - 1) as pool, tqdm(total=len(groupping)) as pbar:
            result_objects = [pool.apply_async(self.predict_for_user_tests,
                                               args=([uid, group, verbose]),
                                               callback=lambda _: pbar.update(1))
                              for uid, group in groupping]
            stats_results = [r.get() for r in result_objects]
        for preds, sizes in stats_results:
            predictions.extend(preds)
            self.users_sizes.append(sizes[0])
            self.items_sizes.append(sizes[1])
        return predictions

    def predict_for_user_tests(self, uid, group, verbose):
        # modelo do user
        iuid = USBCF_nomem.trainset.to_inner_uid(uid)
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
            iuid = USBCF_nomem.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = USBCF_nomem.trainset.to_inner_iid(iid)
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
            est = USBCF_nomem.trainset.global_mean
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = USBCF_nomem.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def __str__(self):
        return 'USBCF_nomem({},{},KNNMeans,{},{})'.format(self.threshold_sim,
                                                          self.nnbrs,
                                                          self.sim_options,
                                                          self.min_cols)



class USBCFCombineBicSols(AlgoBase):

    def __init__(self, threshold_sim=0.5, nnbrs=20, num_biclusters=100,
                 min_cols=[3, 5, 7, 10, 15, 20], consistency=1, max_overlap=1):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

        # bbcf
        self.threshold_sim = threshold_sim

        # biclustering
        self.num_biclusters = num_biclusters
        self.consistency = consistency
        self.min_cols = min_cols
        self.max_overlap = max_overlap

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
            vals.append(r)

        self.rating_matrix_csr = csr_matrix((vals, (row_ind, col_ind)),
                                            shape=(self.trainset.n_users,
                                                   self.trainset.n_items))
        # Preprocessing for biclustering
        rating_matrix_dense = self.rating_matrix_csr.todense()
        rating_matrix_dense.astype(int)

        _logger.info('running biclustering')

        # Generate bics - not saved as attribute to save space
        biclustering_solutions = []
        for i in self.min_cols:
            bic_algo = QUBIC2(num_biclusters=self.num_biclusters,
                              discreteFlag=True, minCols=i,
                              consistency=self.consistency,
                              max_overlap_level=self.max_overlap)
            biclustering_solutions.append(
                bic_algo.run(rating_matrix_dense).biclusters)

        temp_bicsol = [j for i in biclustering_solutions for j in i]

        biclustering_solution = []
        for bic in temp_bicsol:
            flag_max = True
            for bic_final in biclustering_solution:
                if (bic.overlap(bic_final)) == 1 and (bic.area <= bic_final.area):
                    flag_max = False
                    break
            if flag_max:
                biclustering_solution.append(bic)

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
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
        result_objects = [pool.apply_async(self.obtain_user_bics_sims,
                                           args=([inner_uid, biclustering_solution]))
                          for inner_uid in self.trainset.all_users()]

        self.user_fitted_model = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

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
        if self.user_fitted_model[inner_uid] != None:
            prediction = self.user_fitted_model[inner_uid].predict(
                raw_uid, raw_iid)
        else:
            details = {}
            details['was_impossible'] = True
            details['reason'] = "No data to build algorithm"
            prediction = Prediction(raw_uid, raw_iid, None, None, details)
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
            users_indexes_bic = sorted(bic.rows)
            items_indexes_bic = sorted(bic.cols)

            items_indexes_interception = sorted(
                set(user_items_indexes) & set(items_indexes_bic))

            if len(items_indexes_interception) == 0:
                user_sims.append(0)
                continue

            # Missingness similarity
            sim_match = len(
                items_indexes_interception) / len(set(items_indexes_bic))

            

            # Rating deviation similarity
            matrix_bic = self.rating_matrix_csr[np.ix_(users_indexes_bic,
                                                       items_indexes_bic)].todense()

            row_user_interception = [user_ratings[index] for index in
                                     items_indexes_bic]

            new_matrix = matrix_bic[:, np.nonzero(row_user_interception)[0]]

            rating_dev = mean_squared_error(np.asarray(new_matrix[0]).ravel(),
                                            user_ratings[items_indexes_interception],
                                            squared=False)

            rating_deviation = rating_dev/(self.trainset.rating_scale[1]
                                           - self.trainset.rating_scale[0])
            sim_fit = 1-rating_deviation
            _logger.debug("sim_fit:", sim_fit)

            # # sim_significance
            # if bic.pvalue != 0:
            #     pvalue_exp = abs(math.log10(bic.pvalue))
            #     sig_rate = 1.05
            #     sim_significance = (pow(sig_rate, - pvalue_exp) -
            #                         pow(sig_rate, 0)) / (pow(sig_rate, -100) -
            #                                              pow(sig_rate, 0))
            # else:
            #     sim_significance = 1
            # _logger.debug("sim_significance", sim_significance)

            sim = sim_match * sim_fit

            
            user_sims.append(sim)

        nearest_bics_indexes = [i for i, sim in enumerate(user_sims)
                                if sim > self.threshold_sim]
        if len(nearest_bics_indexes) == 0:
            return (inner_uid, None)

        # User-specific U-I matrix
        bic_result_rows = [inner_uid]
        bic_result_cols = []
        for bic_index in nearest_bics_indexes:
            bic_result_rows.extend(bics_sol[bic_index].rows)
            bic_result_cols.extend(bics_sol[bic_index].cols)

        bic_result_rows = set(bic_result_rows)
        bic_result_cols = set(bic_result_cols)
        if inner_uid % 100 == 0:
            _logger.info("New user %d matrix dims: %d %d",
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
        if inner_uid % 100 == 0:
            _logger.info("Trainned model for inner user %d", inner_uid)
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
        return 'USBCFCombineBicSols({},{})'.format(self.threshold_sim, self.min_cols)


class USBCFCombineBicSols_nomem(AlgoBase):

    trainset = None
    biclustering_solution = []
    rating_matrix_csr = None

    def __init__(self, threshold_sim=0.5, nnbrs=20, num_biclusters=100,
                 min_cols=[3, 5, 7, 10, 15, 20], consistency=1, max_overlap=1):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

        # bbcf
        self.threshold_sim = threshold_sim

        # biclustering
        self.num_biclusters = num_biclusters
        self.consistency = consistency
        self.min_cols = min_cols
        self.max_overlap = max_overlap

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
        USBCFCombineBicSols_nomem.trainset = trainset

        row_ind, col_ind, vals = [], [], []

        for (u, i, r) in USBCFCombineBicSols_nomem.trainset.all_ratings():
            row_ind.append(u)
            col_ind.append(i)
            if r == 0:
                r = 99
            vals.append(r)

        USBCFCombineBicSols_nomem.rating_matrix_csr = csr_matrix((vals, (row_ind, col_ind)),
                                                                 shape=(USBCFCombineBicSols_nomem.trainset.n_users,
                                                                        USBCFCombineBicSols_nomem.trainset.n_items))

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

        str_trainset = hash(tuple([(u, i, r) for u, i, r
                                   in trainset.all_ratings()]))
        biclustering_solutions = []
        for i in self.min_cols:
            bic_algo = QUBIC2(num_biclusters=self.num_biclusters,
                              discreteFlag=True, minCols=i,
                              consistency=self.consistency,
                              max_overlap_level=self.max_overlap)
            str_trainset = hash(tuple([(u, i, r) for u, i, r
                                       in trainset.all_ratings()]))
            bic_sol_path = "../../Output/Models-surprise/bicsols/" + \
                str(str_trainset) + "/"
            if not os.path.exists(bic_sol_path):
                os.makedirs(bic_sol_path)
            bic_sol_path = bic_sol_path + str(bic_algo) + ".pkl"
            if os.path.isfile(bic_sol_path):
                _logger.info('using precomputed biclustering solution')
                with open(bic_sol_path, "rb") as f:
                    biclustering_solutions.append(pickle.load(f))
            else:
                _logger.info('computing biclustering solution')
                # Generate bics
                bic_sol = bic_algo.run(rating_matrix_dense)
                # P-Value calculation
                bic_sol.run_constant_freq_column(rating_matrix_dense_rounded,
                                                                list(range(self.trainset.rating_scale[0],
                                                                          self.trainset.rating_scale[1]+1)),
                                                                True)
                biclustering_solutions.append(bic_sol.biclusters)
                with open(bic_sol_path, "wb") as f:
                    pickle.dump(bic_sol.biclusters, f)

        print("Filtering non maximal biclusters")
        #filter unique biclusters
        bicsol = Biclustering([j for i in biclustering_solutions for j in i])
        print("A lista tinha:", len(bicsol.biclusters))
        bicsol.remove_duplicates()
        print("Depois de remover duplicados tem:", len(bicsol.biclusters))
        bicsol.sort_by_area(descending=False)

        temp_biclusters = bicsol.biclusters
        USBCFCombineBicSols_nomem.biclustering_solution = temp_biclusters
        # get number of cpus available to job
        try:
            ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except KeyError:
            ncpus = multiprocessing.cpu_count()
            
        with multiprocessing.Pool(ncpus - 1) as pool, tqdm(total=len(temp_biclusters)) as pbar:

            result_objects = [pool.apply_async(self.get_max_bics,
                                               args=(
                                                   [index, bic]),
                                               callback=lambda _: pbar.update(1))
                              for index, bic in enumerate(temp_biclusters)]
            stats_results = [r.get() for r in result_objects if r.get()]

        # print("Antes de remover nao significativos tem:", len(bicsol.biclusters))
        # bicsol.remove_bypvalue(0.001)
        # print("Depois de remover nao significativos tem:", len(bicsol.biclusters))

        USBCFCombineBicSols_nomem.biclustering_solution = stats_results
        rows_sizes, cols_sizes = list(), list()

        for bic in USBCFCombineBicSols_nomem.biclustering_solution:
            rows_sizes.append(len(bic.rows))
            cols_sizes.append(len(bic.cols))
        self.stats_bics_sol.append([len(USBCFCombineBicSols_nomem.biclustering_solution),
                                    statistics.mean(rows_sizes),
                                    statistics.pstdev(rows_sizes),
                                    statistics.mean(cols_sizes),
                                    statistics.pstdev(cols_sizes)])

        if len(USBCFCombineBicSols_nomem.biclustering_solution) < 1:
            _logger.info('biclustering failed to find biclusters')
            return None

        _logger.info('biclustering completed: %s', self.stats_bics_sol)

        return self

    def get_max_bics(self, index, bic):
        # Percorre a lista de biclusters
        flag_max = True

        for bic2 in USBCFCombineBicSols_nomem.biclustering_solution[(index + 1):]:
            # Bic esta contido em bic2
            if (bic.contained_in(bic2) == 1):
                flag_max = False
                break

        if flag_max:
            return bic
        return None

    def estimate(self, inner_uid, inner_iid, user_model):
        # user or item not in the system
        if not (USBCFCombineBicSols_nomem.trainset.knows_user(inner_uid)
                and USBCFCombineBicSols_nomem.trainset.knows_item(inner_iid)):
            raise PredictionImpossible('User and/or item is unknown.')

        # find user model
        raw_uid = USBCFCombineBicSols_nomem.trainset.to_raw_uid(inner_uid)
        raw_iid = USBCFCombineBicSols_nomem.trainset.to_raw_iid(inner_iid)
        # use user-specific model to predict
        if user_model != None:
            prediction = user_model.predict(raw_uid, raw_iid)
        else:
            details = {}
            details['was_impossible'] = True
            details['reason'] = "No data to build algorithm"
            prediction = Prediction(raw_uid, raw_iid, None, None, details)
        if prediction[4]["was_impossible"]:
            raise PredictionImpossible(
                'User and/or item is unknown (for the cf algorithm).')

        return prediction[3], prediction[4]

    def obtain_user_bics_sims(self, inner_uid):
        user_sims = []
        # item rated by active user
        user_ratings = USBCFCombineBicSols_nomem.rating_matrix_csr.getrow(
            inner_uid).toarray().ravel()
        user_items_indexes = np.flatnonzero(user_ratings).tolist()

        for bic in USBCFCombineBicSols_nomem.biclustering_solution:
            users_indexes_bic = sorted(bic.rows)
            items_indexes_bic = sorted(bic.cols)

            items_indexes_interception = sorted(
                set(user_items_indexes) & set(items_indexes_bic))

            if len(items_indexes_interception) == 0:
                user_sims.append(0)
                continue

            # Missingness similarity
            sim_match = len(
                items_indexes_interception) / len(set(items_indexes_bic))


            # Rating deviation similarity
            matrix_bic = USBCFCombineBicSols_nomem.rating_matrix_csr[np.ix_(users_indexes_bic,
                                                                            items_indexes_bic)].todense()

            row_user_interception = [user_ratings[index] for index in
                                     items_indexes_bic]

            new_matrix = matrix_bic[:, np.nonzero(row_user_interception)[0]]

            rating_dev = mean_squared_error(np.asarray(new_matrix[0]).ravel(),
                                            user_ratings[items_indexes_interception],
                                            squared=False)

            rating_deviation = rating_dev/(USBCFCombineBicSols_nomem.trainset.rating_scale[1]
                                           - USBCFCombineBicSols_nomem.trainset.rating_scale[0])
            sim_fit = 1-rating_deviation

            sim = sim_match * sim_fit

            user_sims.append(sim)

        nearest_bics_indexes = [i for i, sim in enumerate(user_sims)
                                if sim > self.threshold_sim]
        if len(nearest_bics_indexes) == 0:
            return (inner_uid, None, 0, 0)

        # User-specific U-I matrix
        bic_result_rows = set([inner_uid])
        bic_result_cols = set()
        for bic_index in nearest_bics_indexes:
            bic_result_rows.update(
                USBCFCombineBicSols_nomem.biclustering_solution[bic_index].rows)
            bic_result_cols.update(
                USBCFCombineBicSols_nomem.biclustering_solution[bic_index].cols)

        _logger.debug("New user %d matrix dims: %d %d",
                      inner_uid, len(bic_result_rows), len(bic_result_cols))

        df = self.bicluster_to_df([sorted(bic_result_rows),
                                   sorted(bic_result_cols)])

        user_model = self.fit_user_model(inner_uid, df)

        return (inner_uid, user_model, len(bic_result_rows), len(bic_result_cols))

    def fit_user_model(self, inner_uid, user_k_nearest_bic):
        df = user_k_nearest_bic
        data = Dataset.load_from_df(df[['user', 'item', 'rating']],
                                    reader=Reader(rating_scale=USBCFCombineBicSols_nomem.trainset.rating_scale))
        trainset = data.build_full_trainset()
        algo = copy.deepcopy(self.cf_algo)
        user_model = algo.fit(trainset)
        if inner_uid % 100 == 0:
            _logger.debug("Trainned model for inner user %d", inner_uid)
        return user_model

    def bicluster_to_df(self, bicluster):
        rows_bicluster = bicluster[0]
        cols_bicluster = bicluster[1]
        list_bic = []
        for row_idx in rows_bicluster:
            for col_idx in cols_bicluster:
                user = USBCFCombineBicSols_nomem.trainset.to_raw_uid(row_idx)
                item = USBCFCombineBicSols_nomem.trainset.to_raw_iid(col_idx)
                rating = dict(USBCFCombineBicSols_nomem.trainset.ur[row_idx]).get(
                    col_idx, 0)
                if rating != 0:
                    list_bic.extend(
                        [{"user": user, "item": item, "rating": rating}])
        df_bicluster = pd.DataFrame(list_bic)
        return df_bicluster

    # override to be faster with multiprocessing
    def test(self, testset, verbose=False):
        """Test the algorithm on given testset, i.e. estimate all the ratings
        in the given testset.
        """
        # get number of cpus available to job
        try:
            ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except KeyError:
            ncpus = multiprocessing.cpu_count()
        testset = pd.DataFrame(testset, columns=["uid", "iid", "r_ui_trans"])
        predictions = []
        groupping = testset.groupby(["uid"])
        with multiprocessing.Pool(ncpus - 1) as pool, tqdm(total=len(groupping)) as pbar:
            result_objects = [pool.apply_async(self.predict_for_user_tests,
                                               args=([uid, group, verbose]),
                                               callback=lambda _: pbar.update(1))
                              for uid, group in groupping]
            stats_results = [r.get() for r in result_objects]
        for preds, sizes in stats_results:
            predictions.extend(preds)
            self.users_sizes.append(sizes[0])
            self.items_sizes.append(sizes[1])
        return predictions
    

    def predict_for_user_tests(self, uid, group, verbose):
        # modelo do user
        iuid = USBCFCombineBicSols_nomem.trainset.to_inner_uid(uid)
        (_, usermodel, user_size, item_size) = self.obtain_user_bics_sims(iuid)

        user_predictions = [self.estimate(uid,
                                          iid,
                                          r_ui_trans,
                                          verbose=verbose)
                            for (uid, iid, r_ui_trans) in group.values.tolist()]
        return user_predictions, (user_size, item_size)


    def predict_for_user_tests(self, uid, group, verbose):
        # modelo do user
        iuid = USBCFCombineBicSols_nomem.trainset.to_inner_uid(uid)
        (_, usermodel, user_size, item_size) = self.obtain_user_bics_sims(iuid)

        user_predictions = [self.predict(uid,
                                          iid,
                                          r_ui_trans,
                                          usermodel,
                                          verbose=verbose)
                            for (uid, iid, r_ui_trans) in group.values.tolist()]
        return user_predictions, (user_size, item_size)

    def predict(self, uid, iid, r_ui=None, user_model=None, clip=True,
                verbose=False):
        # Convert raw ids to inner ids
        try:
            iuid = USBCFCombineBicSols_nomem.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = USBCFCombineBicSols_nomem.trainset.to_inner_iid(iid)
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
            est = USBCFCombineBicSols_nomem.trainset.global_mean
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = USBCFCombineBicSols_nomem.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def __str__(self):
        return 'USBCFCombineBicSols_nomem({},{},KNNMeans,{},{})'.format(self.threshold_sim, self.nnbrs,
                                                                        self.sim_options, self.min_cols)
