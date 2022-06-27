import pandas as pd
import numpy as np
from biclustering.qubic import QUBIC, QUBIC2
from lenskit.algorithms import Predictor, item_knn
from lenskit import util, matrix
import sys
import logging
import multiprocessing
import statistics
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler(sys.stdout))
_logger.setLevel(level=logging.INFO)


class USBCF_MSR(Predictor):
    """
    Algoritmo baseado no BBCF mas com nova similaridade
    """
    _timer = None

    def __init__(self, number_of_nearest_bics=5, nnbrs=10, num_biclusters=100,
                 min_cols=2, consistency=1, max_overlap=1):
        # bbcf
        self.nnbics = number_of_nearest_bics
        # biclustering
        self.min_num_biclusters = num_biclusters
        self.min_cols = min_cols
        self.consistency = consistency
        self.max_overlap = max_overlap
        # ibknn
        self.algo = item_knn.ItemItem(nnbrs, min_nbrs=1, min_sim=0.0000001,
                                      center=True)

        # mapping from users/items IDs to row/col numbers.
        self.users_map = None
        self.items_map = None
        self.rating_matrix_csr = None
        self.user_fitted_model = dict()

        # extra stats
        self.stats_biclustering_solution = list()
        self.stats_nearest_bics = list()

    def fit(self, ratings, **kwargs):
        _logger.info('Fitting %s', str(self))
        self._timer = util.Stopwatch()

        self.rating_matrix_csr, self.users_map,  self.items_map = matrix.sparse_ratings(
            ratings)

        # preprocessamento
        rating_matrix_dense = self.rating_matrix_csr.to_scipy().todense()
        rating_matrix_dense.astype(int)

        bic_algo = QUBIC2(num_biclusters=self.min_num_biclusters,
                         discreteFlag=True, minCols=self.min_cols,
                         consistency=self.consistency,
                         max_overlap_level=self.max_overlap)
        _logger.info('running biclustering')

        # generate bics - not saved as attribute to save space
        biclustering_solution = bic_algo.run(rating_matrix_dense).biclusters

        rows_sizes, cols_sizes = list(), list()
        for bic in biclustering_solution:
            rows_sizes.append(len(bic.rows))
            cols_sizes.append(len(bic.cols))
        self.stats_biclustering_solution.append([len(biclustering_solution),
                                                 statistics.mean(rows_sizes),
                                                 statistics.pstdev(rows_sizes),
                                                 statistics.mean(cols_sizes),
                                                 statistics.pstdev(cols_sizes)])

        # calculate similarities
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        result_objects = [pool.apply_async(self.obtain_user_bics_similarities,
                                           args=([idx, biclustering_solution]))
                          for idx, _ in enumerate(self.users_map)]
        user_k_nearest_bic = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        rows_sizes, cols_sizes = list(), list()
        for user in user_k_nearest_bic:
            rows_sizes.append(len(user_k_nearest_bic[user].user.unique()))
            cols_sizes.append(len(user_k_nearest_bic[user].item.unique()))
        self.stats_nearest_bics.append([statistics.mean(rows_sizes),
                                        statistics.pstdev(rows_sizes),
                                        statistics.mean(cols_sizes),
                                        statistics.pstdev(cols_sizes)])

        _logger.info(' [%s] found k nearest biclusters - %s',
                     self._timer, str(self))

        # fit user models
        ncpus = int(multiprocessing.cpu_count()/2) - \
            1 if int(multiprocessing.cpu_count()/2)-1 > 1 else 1
        pool = multiprocessing.Pool(ncpus)
        result_objects = [pool.apply_async(self.fit_user_model, args=(
            [idx, user_k_nearest_bic])) for idx, _ in enumerate(self.users_map)]

        self.user_fitted_model = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        _logger.info(' [%s] trained model - %s', self._timer, str(self))

        del self._timer

    def predict_for_user(self, user, items, ratings=None):
        if user % 100 == 0:
            _logger.debug('predicting %s items for user %s', len(items), user)
        # user nao esta no sistema
        if user not in self.users_map:
            _logger.debug('user %s missing, returning empty predictions', user)
            return pd.Series(np.nan, index=items)

        # get user index
        idx_user = self.users_map.get_loc(user)

        # encontra o modelo do user
        rating_pred = self.user_fitted_model[idx_user].predict_for_user(
            user, items)

        _logger.debug('user %s: predicted for %d of %d items',
                      user, rating_pred.notna().sum(), len(items))
        _logger.debug(rating_pred)
        return rating_pred

    def obtain_user_bics_similarities(self, idx_user, biclustering_solution):
        user_sims = []
        # item rated by active user
        user_items_indexes = np.nonzero(
            self.rating_matrix_csr.row(idx_user))[0].tolist()

        for bic in biclustering_solution:
            users_indexes_bic = sorted(bic.rows)
            items_indexes_bic = sorted(bic.cols)

            items_indexes_interception = sorted(
                set(user_items_indexes) & set(items_indexes_bic))

            # similaridade- intercepcao
            sim_u_b_intercept = len(
                items_indexes_interception) / len(set(items_indexes_bic))

            # sem items em comum
            if sim_u_b_intercept == 0:
                user_sims.append((0, float('inf')))
                continue

            matrix_bic = np.asarray(self.rating_matrix_csr.to_scipy(
            )[np.ix_(users_indexes_bic, items_indexes_bic)].todense())

            # msr-inicial
            msr_bic = self.msr_missings_adaptation(matrix_bic)

            # caso user ja la esteja, msr = ao do bic
            if idx_user in users_indexes_bic:
                user_sims.append((sim_u_b_intercept, 0))
                continue

            # msr-final
            row_user_interception = [self.rating_matrix_csr.row(
                idx_user)[index] for index in items_indexes_bic]
            matrix_user = np.vstack([matrix_bic, row_user_interception])
            msr_bicanduser = self.msr_missings_adaptation(matrix_user)

            # versao so msrfinal
            sim_pattern = msr_bicanduser-msr_bic
            user_sims.append((sim_u_b_intercept, sim_pattern))

        # find nearest bics for the user
        nearest_bics_indexes_interception = sorted(range(len(user_sims)),
                                                   key=lambda k: user_sims[k][0],
                                                   reverse=True)
        nearest_bics_indexes_pattern = sorted(range(len(user_sims)),
                                              key=lambda k: user_sims[k][1],
                                              reverse=False)

        result = dict()
        for pos1_bic, bic in enumerate(nearest_bics_indexes_interception):
            result[bic] = pos1_bic + nearest_bics_indexes_pattern.index(bic)

        user_k_nearest_bics_indexes = [k for k, v in sorted(
            result.items(), key=lambda item: item[1])][:self.nnbics]

        #ideia de numeros de bics dinamicos, depois testar
        # user_k_nearest_bics_indexes = []
        # items_covered = set()
        # for k,v in sorted(result.items(), key=lambda item: item[1]):
        #     if ((len(items_covered) / len(self.items_map)) < 0.8) and (len(user_k_nearest_bics_indexes) < self.nnbics):
        #         user_k_nearest_bics_indexes.append(k)
        #         items_covered.update(biclustering_solution[k].cols)
        #     else:
        #         print("Para o user", idx_user, "usou", len(user_k_nearest_bics_indexes), "bics")
        #         break

        #comeca com o user, assumindo que faz sentido o user estar no bicluster
        bic_result_rows = set([idx_user])
        bic_result_cols = set()
        for bic_index in user_k_nearest_bics_indexes:
            bic_result_rows = bic_result_rows.union(
                set(biclustering_solution[bic_index].rows))
            bic_result_cols = bic_result_cols.union(
                set(biclustering_solution[bic_index].cols))

        print("Criou a matrix de dados thread do user", idx_user,
              " - dims do bic setup:", len(bic_result_rows),
              len(bic_result_cols))
        return (idx_user, self.bicluster_to_df([sorted(bic_result_rows),
                                                sorted(bic_result_cols)]))

    def fit_user_model(self, idx_user, user_k_nearest_bic):
        user_model = util.clone(self.algo).fit(user_k_nearest_bic[idx_user])
        if idx_user % 100 == 0:
            _logger.info("Fez o modelo do user %d", idx_user)
        return (idx_user, user_model)


    def msr_missings_adaptation(self, matrix):
        matrix_copy = np.copy(np.where(matrix != 0, matrix, np.nan))
        rows_nan_means = np.nanmean(matrix_copy, 1)
        cols_nan_means = np.nanmean(matrix_copy, 0)
        matrix_nan_means = np.nanmean(matrix_copy)
        residue_sum = 0
        for idx_row, row in enumerate(matrix_copy):
            for idx_col, elem in enumerate(row):
                if not np.isnan(elem):
                    residue_sum += (elem - rows_nan_means[idx_row] -
                                    cols_nan_means[idx_col] + matrix_nan_means)**2
        return(1/(np.count_nonzero(~np.isnan(matrix_copy))) * residue_sum)

    def bicluster_to_df(self, bicluster):
        rows_bicluster = bicluster[0]
        cols_bicluster = bicluster[1]
        df_bicluster = pd.DataFrame(columns=["user", "item", "rating"])
        for row_idx in rows_bicluster:
            for col_idx in cols_bicluster:
                user = self.users_map[row_idx]
                item = self.items_map[col_idx]
                rating = self.rating_matrix_csr.to_scipy()[row_idx, col_idx]
                if rating != 0:
                    df_bicluster = df_bicluster.append(
                        {"user": user, "item": item, "rating": rating},
                        ignore_index=True)
        return df_bicluster.astype({"user": int, "item": int, "rating": float})

    def __str__(self):
        return 'USBCF_MSR({},{},{},{},{})'.format(self.nnbics, self.min_num_biclusters,
                                                  self.min_cols, self.consistency,
                                                  self.max_overlap)


class USBCF_CMR(Predictor):
    """
    Algoritmo baseado no BBCF mas com nova similaridade
    """
    _timer = None

    def __init__(self, number_of_nearest_bics=5, nnbrs=10, num_biclusters=30,
                 min_cols=2, consistency=1, max_overlap=0.99):
        # bbcf
        self.nnbics = number_of_nearest_bics
        # biclustering
        self.min_num_biclusters = num_biclusters
        self.min_cols = min_cols
        self.consistency = consistency
        self.max_overlap = max_overlap
        # ibknn
        self.algo = item_knn.ItemItem(
            nnbrs, min_nbrs=1, min_sim=0.0000001, center=True)

        # mapping from users/items IDs to row/col numbers.
        self.users_map = None
        self.items_map = None
        self.rating_matrix_csr = None
        self.user_fitted_model = dict()

        # extra stats
        self.stats_biclustering_solution = list()
        self.stats_nearest_bics = list()

    def fit(self, ratings, **kwargs):
        _logger.info('starting MyAlgorithm_NewSIM-train')
        self._timer = util.Stopwatch()

        self.rating_matrix_csr, self.users_map,  self.items_map = matrix.sparse_ratings(
            ratings)

        # preprocessamento
        rating_matrix_dense = self.rating_matrix_csr.to_scipy().todense()
        qubic = None
        qubic = QUBIC(num_biclusters=self.min_num_biclusters, discrete=False,
                      minCols=self.min_cols,
                      consistency=self.consistency,
                      max_overlap_level=self.max_overlap)

        # generate bics - not saved as attribute to save space
        biclustering_solution = qubic.run(rating_matrix_dense).biclusters

        rows_sizes, cols_sizes = list(), list()
        for bic in biclustering_solution:
            rows_sizes.append(len(bic.rows))
            cols_sizes.append(len(bic.cols))
        self.stats_biclustering_solution.append([len(biclustering_solution),
                                                 statistics.mean(rows_sizes),
                                                 statistics.pstdev(rows_sizes),
                                                 statistics.mean(cols_sizes),
                                                 statistics.pstdev(cols_sizes)])
        _logger.info(' [%s] biclustering run ', self._timer)

        # calculate similarities
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        result_objects = [pool.apply_async(self.obtain_user_bics_similarities, args=(
            [idx, biclustering_solution])) for idx, _ in enumerate(self.users_map)]
        user_k_nearest_bic = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        rows_sizes, cols_sizes = list(), list()
        for user in user_k_nearest_bic:
            rows_sizes.append(len(user_k_nearest_bic[user].user.unique()))
            cols_sizes.append(len(user_k_nearest_bic[user].item.unique()))
        self.stats_nearest_bics.append([statistics.mean(rows_sizes),
                                        statistics.pstdev(rows_sizes),
                                        statistics.mean(cols_sizes),
                                        statistics.pstdev(cols_sizes)])

        _logger.info(' [%s] found k nearest biclusters - %s',
                     self._timer, str(self))

        # fit user models
        ncpus = int(multiprocessing.cpu_count()/2) - \
            1 if int(multiprocessing.cpu_count()/2)-1 > 1 else 1
        pool = multiprocessing.Pool(ncpus)
        result_objects = [pool.apply_async(self.fit_user_model, args=(
            [idx, user_k_nearest_bic])) for idx, _ in enumerate(self.users_map)]

        self.user_fitted_model = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        _logger.info(' [%s] trained model - %s', self._timer, str(self))

        del self._timer

    def predict_for_user(self, user, items, ratings=None):
        if user % 100 == 0:
            _logger.debug('predicting %s items for user %s', len(items), user)
        # user nao esta no sistema
        if user not in self.users_map:
            _logger.debug('user %s missing, returning empty predictions', user)
            return pd.Series(np.nan, index=items)

        # get user index
        idx_user = self.users_map.get_loc(user)

        # encontra o modelo do user
        rating_pred = self.user_fitted_model[idx_user].predict_for_user(
            user, items)

        _logger.debug('user %s: predicted for %d of %d items',
                      user, rating_pred.notna().sum(), len(items))
        _logger.debug(rating_pred)
        return rating_pred

    def obtain_user_bics_similarities(self, idx_user, biclustering_solution):
        user_sims = []
        # item rated by active user
        user_items_indexes = np.nonzero(
            self.rating_matrix_csr.row(idx_user))[0].tolist()

        for bic in biclustering_solution:
            users_indexes_bic = sorted(bic.rows)
            items_indexes_bic = sorted(bic.cols)

            items_indexes_interception = sorted(
                set(user_items_indexes) & set(items_indexes_bic))

            # similaridade- intercepcao
            sim_u_b_intercept = len(
                items_indexes_interception) / len(set(items_indexes_bic))

            # sem items em comum
            if sim_u_b_intercept == 0:
                user_sims.append((0, float('inf')))
                continue

            matrix_bic = np.asarray(self.rating_matrix_csr.to_scipy(
            )[np.ix_(users_indexes_bic, items_indexes_bic)].todense())

            # msr-inicial
            msr_bic = self.column_residue(matrix_bic)

            # caso user ja la esteja, msr = ao do bic
            if idx_user in users_indexes_bic:
                user_sims.append((sim_u_b_intercept, 0))
                continue

            # msr-final
            row_user_interception = [self.rating_matrix_csr.row(
                idx_user)[index] for index in items_indexes_bic]
            matrix_user = np.vstack([matrix_bic, row_user_interception])
            msr_bicanduser = self.column_residue(matrix_user)

            # versao so msrfinal
            sim_pattern = msr_bicanduser-msr_bic
            user_sims.append((sim_u_b_intercept, sim_pattern))

        # find nearest bics for the user
        nearest_bics_indexes_interception = sorted(range(len(user_sims)),
                                                   key=lambda k: user_sims[k][0],
                                                   reverse=True)
        nearest_bics_indexes_pattern = sorted(range(len(user_sims)),
                                              key=lambda k: user_sims[k][1],
                                              reverse=False)

        result = dict()
        for pos1_bic, bic in enumerate(nearest_bics_indexes_interception):
            result[bic] = pos1_bic + nearest_bics_indexes_pattern.index(bic)

        user_k_nearest_bics_indexes = [k for k, v in sorted(
            result.items(), key=lambda item: item[1])][:self.nnbics]

        #ideia de numeros de bics dinamicos, depois testar
        # user_k_nearest_bics_indexes = []
        # items_covered = set()
        # for k,v in sorted(result.items(), key=lambda item: item[1]):
        #     if ((len(items_covered) / len(self.items_map)) < 0.8) and (len(user_k_nearest_bics_indexes) < self.nnbics):
        #         user_k_nearest_bics_indexes.append(k)
        #         items_covered.update(biclustering_solution[k].cols)
        #     else:
        #         print("Para o user", idx_user, "usou", len(user_k_nearest_bics_indexes), "bics")
        #         break

        #comeca com o user, assumindo que faz sentido o user estar no bicluster
        bic_result_rows = set([idx_user])
        bic_result_cols = set()
        for bic_index in user_k_nearest_bics_indexes:
            bic_result_rows = bic_result_rows.union(
                set(biclustering_solution[bic_index].rows))
            bic_result_cols = bic_result_cols.union(
                set(biclustering_solution[bic_index].cols))

        print("Criou a matrix de dados thread do user", idx_user,
              " - dims do bic setup:", len(bic_result_rows), len(bic_result_cols))
        return (idx_user, self.bicluster_to_df([sorted(bic_result_rows), sorted(bic_result_cols)]))

    def fit_user_model(self, idx_user, user_k_nearest_bic):
        user_model = util.clone(self.algo).fit(user_k_nearest_bic[idx_user])
        if idx_user % 100 == 0:
            _logger.info("Fez o modelo do user %d", idx_user)
        return (idx_user, user_model)

    def column_residue(self, matrix):
        matrix_copy = np.copy(np.where(matrix != 0, matrix, np.nan))
        cols_nan_means = np.nanmean(matrix_copy, 0)
        residue_sum = 0
        for idx_row, row in enumerate(matrix_copy):
            for idx_col, elem in enumerate(row):
                if not np.isnan(elem):
                    residue_sum += (elem - cols_nan_means[idx_col])**2
        return(1/(np.count_nonzero(~np.isnan(matrix_copy))) * residue_sum)

    def bicluster_to_df(self, bicluster):
        rows_bicluster = bicluster[0]
        cols_bicluster = bicluster[1]
        df_bicluster = pd.DataFrame(columns=["user", "item", "rating"])
        for row_idx in rows_bicluster:
            for col_idx in cols_bicluster:
                user = self.users_map[row_idx]
                item = self.items_map[col_idx]
                rating = self.rating_matrix_csr.to_scipy()[row_idx, col_idx]
                if rating != 0:
                    df_bicluster = df_bicluster.append(
                        {"user": user, "item": item, "rating": rating},
                        ignore_index=True)
        return df_bicluster.astype({"user": int, "item": int, "rating": float})

    def __str__(self):
        return 'USBCF_CMR({},{},{},{},{})'.format(self.nnbics,
                                                  self.min_num_biclusters,
                                                  self.min_cols,
                                                  self.consistency,
                                                  self.max_overlap)


class USBCF_MSR_noAddUser(Predictor):
    """
    msr sem adicionar user
    """
    _timer = None

    def __init__(self, number_of_nearest_bics=5, nnbrs=10, num_biclusters=30, min_cols=2, consistency=1, max_overlap=0.99):
        #bbcf
        self.nnbics = number_of_nearest_bics
        #biclustering
        self.min_num_biclusters = num_biclusters
        self.min_cols = min_cols
        self.consistency = consistency
        self.max_overlap = max_overlap
        #ibknn
        self.algo = item_knn.ItemItem(
            nnbrs, min_nbrs=1, min_sim=0.0000001, center=True)

        #mapping from users/items IDs to row/col numbers.
        self.users_map = None
        self.items_map = None
        self.rating_matrix_csr = None
        self.user_fitted_model = dict()

        #extra stats
        self.stats_biclustering_solution = list()
        self.stats_nearest_bics = list()

    def fit(self, ratings, **kwargs):
        _logger.info('starting MyAlgorithm_NewSIM-train')
        self._timer = util.Stopwatch()

        self.rating_matrix_csr, self.users_map,  self.items_map = matrix.sparse_ratings(
            ratings)

        #preprocessamento
        rating_matrix_dense = self.rating_matrix_csr.to_scipy().todense()
        qubic = None
        qubic = QUBIC(num_biclusters=self.min_num_biclusters, discrete=False,
                                        minCols=self.min_cols, consistency=self.consistency,
                                        max_overlap_level=self.max_overlap)

        #generate bics - not saved as attribute to save space
        biclustering_solution = qubic.run(rating_matrix_dense).biclusters

        rows_sizes, cols_sizes = list(), list()
        for bic in biclustering_solution:
            rows_sizes.append(len(bic.rows))
            cols_sizes.append(len(bic.cols))
        self.stats_biclustering_solution.append([len(biclustering_solution), statistics.mean(rows_sizes), statistics.pstdev(rows_sizes),
                                                 statistics.mean(cols_sizes), statistics.pstdev(cols_sizes)])
        _logger.info(' [%s] biclustering run ', self._timer)

        #calculate similarities
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        result_objects = [pool.apply_async(self.obtain_user_bics_similarities, args=(
            [idx, biclustering_solution])) for idx, _ in enumerate(self.users_map)]
        user_k_nearest_bic = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        rows_sizes, cols_sizes = list(), list()
        for user in user_k_nearest_bic:
            rows_sizes.append(len(user_k_nearest_bic[user].user.unique()))
            cols_sizes.append(len(user_k_nearest_bic[user].item.unique()))
        self.stats_nearest_bics.append([statistics.mean(rows_sizes), statistics.pstdev(rows_sizes),
                                        statistics.mean(cols_sizes), statistics.pstdev(cols_sizes)])

        _logger.info(' [%s] found k nearest biclusters - %s',
                     self._timer, str(self))

        #fit user models
        ncpus = int(multiprocessing.cpu_count()/2) - \
            1 if int(multiprocessing.cpu_count()/2)-1 > 1 else 1
        pool = multiprocessing.Pool(ncpus)
        result_objects = [pool.apply_async(self.fit_user_model, args=(
            [idx, user_k_nearest_bic])) for idx, _ in enumerate(self.users_map)]

        self.user_fitted_model = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        _logger.info(' [%s] trained model - %s', self._timer, str(self))

        del self._timer

    def predict_for_user(self, user, items, ratings=None):
        if user % 100 == 0:
            _logger.debug('predicting %s items for user %s', len(items), user)
        #user nao esta no sistema
        if user not in self.users_map:
            _logger.debug('user %s missing, returning empty predictions', user)
            return pd.Series(np.nan, index=items)

        #get user index
        idx_user = self.users_map.get_loc(user)

        #encontra o modelo do user
        rating_pred = self.user_fitted_model[idx_user].predict_for_user(
            user, items)

        _logger.debug('user %s: predicted for %d of %d items',
                      user, rating_pred.notna().sum(), len(items))
        _logger.debug(rating_pred)
        return rating_pred

    def obtain_user_bics_similarities(self, idx_user, biclustering_solution):
        user_sims = []
        #item rated by active user
        user_items_indexes = np.nonzero(
            self.rating_matrix_csr.row(idx_user))[0].tolist()

        for bic in biclustering_solution:
            users_indexes_bic = sorted(bic.rows)
            items_indexes_bic = sorted(bic.cols)

            items_indexes_interception = sorted(
                set(user_items_indexes) & set(items_indexes_bic))

            #similaridade- intercepcao
            sim_u_b_intercept = len(
                items_indexes_interception) / len(set(items_indexes_bic))

            #sem items em comum
            if sim_u_b_intercept == 0:
                user_sims.append((0, float('inf')))
                continue

            matrix_bic = np.asarray(self.rating_matrix_csr.to_scipy(
            )[np.ix_(users_indexes_bic, items_indexes_bic)].todense())

            #msr-inicial
            msr_bic = self.msr_missings_adaptation(matrix_bic)

            #caso user ja la esteja, msr = ao do bic
            if idx_user in users_indexes_bic:
                user_sims.append((sim_u_b_intercept, 0))
                continue

            #msr-final
            row_user_interception = [self.rating_matrix_csr.row(
                idx_user)[index] for index in items_indexes_bic]
            matrix_user = np.vstack([matrix_bic, row_user_interception])
            msr_bicanduser = self.msr_missings_adaptation(matrix_user)

            #versao so msrfinal
            sim_pattern = msr_bicanduser-msr_bic
            user_sims.append((sim_u_b_intercept, sim_pattern))

        #find nearest bics for the user
        nearest_bics_indexes_interception = sorted(range(len(user_sims)),
                                                   key=lambda k: user_sims[k][0], reverse=True)
        nearest_bics_indexes_pattern = sorted(range(len(user_sims)),
                                              key=lambda k: user_sims[k][1], reverse=False)

        result = dict()
        for pos1_bic, bic in enumerate(nearest_bics_indexes_interception):
            result[bic] = pos1_bic + nearest_bics_indexes_pattern.index(bic)

        user_k_nearest_bics_indexes = [k for k, v in sorted(
            result.items(), key=lambda item: item[1])][:self.nnbics]

        #SEM ADICIONAR O USER
        bic_result_rows = set()
        bic_result_cols = set()
        for bic_index in user_k_nearest_bics_indexes:
            bic_result_rows = bic_result_rows.union(
                set(biclustering_solution[bic_index].rows))
            bic_result_cols = bic_result_cols.union(
                set(biclustering_solution[bic_index].cols))

        print("Criou a matrix de dados thread do user", idx_user,
              " - dims do bic setup:", len(bic_result_rows), len(bic_result_cols))
        return (idx_user, self.bicluster_to_df([sorted(bic_result_rows), sorted(bic_result_cols)]))

    def fit_user_model(self, idx_user, user_k_nearest_bic):
        user_model = util.clone(self.algo).fit(user_k_nearest_bic[idx_user])
        if idx_user % 100 == 0:
            _logger.info("Fez o modelo do user %d", idx_user)
        return (idx_user, user_model)

    def msr_missings_adaptation(self, matrix):
        matrix_copy = np.copy(np.where(matrix != 0, matrix, np.nan))
        rows_nan_means = np.nanmean(matrix_copy, 1)
        cols_nan_means = np.nanmean(matrix_copy, 0)
        matrix_nan_means = np.nanmean(matrix_copy)
        residue_sum = 0
        for idx_row, row in enumerate(matrix_copy):
            for idx_col, elem in enumerate(row):
                if not np.isnan(elem):
                    residue_sum += (elem - rows_nan_means[idx_row] -
                                    cols_nan_means[idx_col] + matrix_nan_means)**2
        return(1/(np.count_nonzero(~np.isnan(matrix_copy))) * residue_sum)

    def bicluster_to_df(self, bicluster):
        rows_bicluster = bicluster[0]
        cols_bicluster = bicluster[1]
        df_bicluster = pd.DataFrame(columns=["user", "item", "rating"])
        for row_idx in rows_bicluster:
            for col_idx in cols_bicluster:
                user = self.users_map[row_idx]
                item = self.items_map[col_idx]
                rating = self.rating_matrix_csr.to_scipy()[row_idx, col_idx]
                if rating != 0:
                    df_bicluster = df_bicluster.append(
                        {"user": user, "item": item, "rating": rating}, ignore_index=True)
        return df_bicluster.astype({"user": int, "item": int, "rating": float})

    def __str__(self):
        return 'USBCF_MSR_noAddUser({},{},{},{},{})'.format(self.nnbics, self.min_num_biclusters, self.min_cols,
                                                            self.consistency, self.max_overlap)


class USBCF_CMR_noAddUser(Predictor):
    """
    cmr sem adicionar user
    """
    _timer = None

    def __init__(self, number_of_nearest_bics=5, nnbrs=10, num_biclusters=30, min_cols=2, consistency=1, max_overlap=0.99):
        #bbcf
        self.nnbics = number_of_nearest_bics
        #biclustering
        self.min_num_biclusters = num_biclusters
        self.min_cols = min_cols
        self.consistency = consistency
        self.max_overlap = max_overlap
        #ibknn
        self.algo = item_knn.ItemItem(
            nnbrs, min_nbrs=1, min_sim=0.0000001, center=True)

        #mapping from users/items IDs to row/col numbers.
        self.users_map = None
        self.items_map = None
        self.rating_matrix_csr = None
        self.user_fitted_model = dict()

        #extra stats
        self.stats_biclustering_solution = list()
        self.stats_nearest_bics = list()

    def fit(self, ratings, **kwargs):
        _logger.info('starting MyAlgorithm_NewSIM-train')
        self._timer = util.Stopwatch()

        self.rating_matrix_csr, self.users_map,  self.items_map = matrix.sparse_ratings(
            ratings)

        #preprocessamento
        rating_matrix_dense = self.rating_matrix_csr.to_scipy().todense()
        qubic = None
        qubic = QualitativeBiclustering(num_biclusters=self.min_num_biclusters, discrete=False,
                                        minCols=self.min_cols, consistency=self.consistency,
                                        max_overlap_level=self.max_overlap)

        #generate bics - not saved as attribute to save space
        biclustering_solution = qubic.run(rating_matrix_dense).biclusters

        rows_sizes, cols_sizes = list(), list()
        for bic in biclustering_solution:
            rows_sizes.append(len(bic.rows))
            cols_sizes.append(len(bic.cols))
        self.stats_biclustering_solution.append([len(biclustering_solution), statistics.mean(rows_sizes), statistics.pstdev(rows_sizes),
                                                 statistics.mean(cols_sizes), statistics.pstdev(cols_sizes)])
        _logger.info(' [%s] biclustering run ', self._timer)

        #calculate similarities
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        result_objects = [pool.apply_async(self.obtain_user_bics_similarities, args=(
            [idx, biclustering_solution])) for idx, _ in enumerate(self.users_map)]
        user_k_nearest_bic = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        rows_sizes, cols_sizes = list(), list()
        for user in user_k_nearest_bic:
            rows_sizes.append(len(user_k_nearest_bic[user].user.unique()))
            cols_sizes.append(len(user_k_nearest_bic[user].item.unique()))
        self.stats_nearest_bics.append([statistics.mean(rows_sizes), statistics.pstdev(rows_sizes),
                                        statistics.mean(cols_sizes), statistics.pstdev(cols_sizes)])

        _logger.info(' [%s] found k nearest biclusters - %s',
                     self._timer, str(self))

        #fit user models
        ncpus = int(multiprocessing.cpu_count()/2) - \
            1 if int(multiprocessing.cpu_count()/2)-1 > 1 else 1
        pool = multiprocessing.Pool(ncpus)
        result_objects = [pool.apply_async(self.fit_user_model, args=(
            [idx, user_k_nearest_bic])) for idx, _ in enumerate(self.users_map)]

        self.user_fitted_model = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        _logger.info(' [%s] trained model - %s', self._timer, str(self))

        del self._timer

    def predict_for_user(self, user, items, ratings=None):
        if user % 100 == 0:
            _logger.debug('predicting %s items for user %s', len(items), user)
        #user nao esta no sistema
        if user not in self.users_map:
            _logger.debug('user %s missing, returning empty predictions', user)
            return pd.Series(np.nan, index=items)

        #get user index
        idx_user = self.users_map.get_loc(user)

        #encontra o modelo do user
        rating_pred = self.user_fitted_model[idx_user].predict_for_user(
            user, items)

        _logger.debug('user %s: predicted for %d of %d items',
                      user, rating_pred.notna().sum(), len(items))
        _logger.debug(rating_pred)
        return rating_pred

    def obtain_user_bics_similarities(self, idx_user, biclustering_solution):
        user_sims = []
        #item rated by active user
        user_items_indexes = np.nonzero(
            self.rating_matrix_csr.row(idx_user))[0].tolist()

        for bic in biclustering_solution:
            users_indexes_bic = sorted(bic.rows)
            items_indexes_bic = sorted(bic.cols)

            items_indexes_interception = sorted(
                set(user_items_indexes) & set(items_indexes_bic))

            #similaridade- intercepcao
            sim_u_b_intercept = len(
                items_indexes_interception) / len(set(items_indexes_bic))

            #sem items em comum
            if sim_u_b_intercept == 0:
                user_sims.append((0, float('inf')))
                continue

            matrix_bic = np.asarray(self.rating_matrix_csr.to_scipy(
            )[np.ix_(users_indexes_bic, items_indexes_bic)].todense())

            #msr-inicial
            msr_bic = self.column_residue(matrix_bic)

            #caso user ja la esteja, msr = ao do bic
            if idx_user in users_indexes_bic:
                user_sims.append((sim_u_b_intercept, 0))
                continue

            #msr-final
            row_user_interception = [self.rating_matrix_csr.row(
                idx_user)[index] for index in items_indexes_bic]
            matrix_user = np.vstack([matrix_bic, row_user_interception])
            msr_bicanduser = self.column_residue(matrix_user)

            #versao so msrfinal
            sim_pattern = msr_bicanduser-msr_bic
            user_sims.append((sim_u_b_intercept, sim_pattern))

        #find nearest bics for the user
        nearest_bics_indexes_interception = sorted(range(len(user_sims)),
                                                   key=lambda k: user_sims[k][0], reverse=True)
        nearest_bics_indexes_pattern = sorted(range(len(user_sims)),
                                              key=lambda k: user_sims[k][1], reverse=False)

        result = dict()
        for pos1_bic, bic in enumerate(nearest_bics_indexes_interception):
            result[bic] = pos1_bic + nearest_bics_indexes_pattern.index(bic)

        user_k_nearest_bics_indexes = [k for k, v in sorted(
            result.items(), key=lambda item: item[1])][:self.nnbics]

        #SEM ADICIONAR O USER
        bic_result_rows = set()
        bic_result_cols = set()
        for bic_index in user_k_nearest_bics_indexes:
            bic_result_rows = bic_result_rows.union(
                set(biclustering_solution[bic_index].rows))
            bic_result_cols = bic_result_cols.union(
                set(biclustering_solution[bic_index].cols))

        print("Criou a matrix de dados thread do user", idx_user,
              " - dims do bic setup:", len(bic_result_rows), len(bic_result_cols))
        return (idx_user, self.bicluster_to_df([sorted(bic_result_rows), sorted(bic_result_cols)]))

    def fit_user_model(self, idx_user, user_k_nearest_bic):
        user_model = util.clone(self.algo).fit(user_k_nearest_bic[idx_user])
        if idx_user % 100 == 0:
            _logger.info("Fez o modelo do user %d", idx_user)
        return (idx_user, user_model)

    def column_residue(self, matrix):
        matrix_copy = np.copy(np.where(matrix != 0, matrix, np.nan))
        cols_nan_means = np.nanmean(matrix_copy, 0)
        residue_sum = 0
        for idx_row, row in enumerate(matrix_copy):
            for idx_col, elem in enumerate(row):
                if not np.isnan(elem):
                    residue_sum += (elem - cols_nan_means[idx_col])**2
        return(1/(np.count_nonzero(~np.isnan(matrix_copy))) * residue_sum)

    def bicluster_to_df(self, bicluster):
        rows_bicluster = bicluster[0]
        cols_bicluster = bicluster[1]
        df_bicluster = pd.DataFrame(columns=["user", "item", "rating"])
        for row_idx in rows_bicluster:
            for col_idx in cols_bicluster:
                user = self.users_map[row_idx]
                item = self.items_map[col_idx]
                rating = self.rating_matrix_csr.to_scipy()[row_idx, col_idx]
                if rating != 0:
                    df_bicluster = df_bicluster.append(
                        {"user": user, "item": item, "rating": rating}, ignore_index=True)
        return df_bicluster.astype({"user": int, "item": int, "rating": float})

    def __str__(self):
        return 'USBCF_CMR_noAddUser({},{},{},{},{})'.format(self.nnbics, self.min_num_biclusters, self.min_cols,
                                                            self.consistency, self.max_overlap)


class USBCF_CMR_AdaptiveNNbics(Predictor):
    """
    Algoritmo baseado no BBCF mas com nova similaridade
    """
    _timer = None

    def __init__(self, number_of_nearest_bics=5, nnbrs=10, num_biclusters=30, min_cols=2, consistency=1, max_overlap=0.99):
        #bbcf
        self.nnbics = number_of_nearest_bics
        #biclustering
        self.min_num_biclusters = num_biclusters
        self.min_cols = min_cols
        self.consistency = consistency
        self.max_overlap = max_overlap
        #ibknn
        self.algo = item_knn.ItemItem(
            nnbrs, min_nbrs=1, min_sim=0.0000001, center=True)

        #mapping from users/items IDs to row/col numbers.
        self.users_map = None
        self.items_map = None
        self.rating_matrix_csr = None
        self.user_fitted_model = dict()

        #extra stats
        self.stats_biclustering_solution = list()
        self.stats_nearest_bics = list()
        self.items_biclust_sol = 0
        self.conta_bics = list()

    def fit(self, ratings, **kwargs):
        _logger.info('starting MyAlgorithm_NewSIM-train')
        self._timer = util.Stopwatch()

        self.rating_matrix_csr, self.users_map,  self.items_map = matrix.sparse_ratings(
            ratings)

        #preprocessamento
        rating_matrix_dense = self.rating_matrix_csr.to_scipy().todense()
        qubic = None
        qubic = QualitativeBiclustering(num_biclusters=self.min_num_biclusters, discrete=False,
                                        minCols=self.min_cols, consistency=self.consistency,
                                        max_overlap_level=self.max_overlap)

        #generate bics - not saved as attribute to save space
        biclustering_solution = qubic.run(rating_matrix_dense).biclusters

        rows_sizes, cols_sizes = list(), list()
        items_biclust_sol = set()
        for bic in biclustering_solution:
            rows_sizes.append(len(bic.rows))
            cols_sizes.append(len(bic.cols))
            items_biclust_sol.update(bic.cols)
        self.items_biclust_sol = len(items_biclust_sol)
        self.stats_biclustering_solution.append([len(biclustering_solution), statistics.mean(rows_sizes), statistics.pstdev(rows_sizes),
                                                 statistics.mean(cols_sizes), statistics.pstdev(cols_sizes)])
        _logger.info(' [%s] biclustering run ', self._timer)

        #calculate similarities
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        result_objects = [pool.apply_async(self.obtain_user_bics_similarities, args=(
            [idx, biclustering_solution])) for idx, _ in enumerate(self.users_map)]
        user_k_nearest_bic = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        rows_sizes, cols_sizes = list(), list()
        for user in user_k_nearest_bic:
            rows_sizes.append(len(user_k_nearest_bic[user]))
            cols_sizes.append(len(user_k_nearest_bic[user].item.unique()))
        self.stats_nearest_bics.append([statistics.mean(rows_sizes), statistics.pstdev(rows_sizes),
                                        statistics.mean(cols_sizes), statistics.pstdev(cols_sizes)])

        _logger.info(' [%s] found k nearest biclusters - %s',
                     self._timer, str(self))
        print("total", self.conta_bics, self.min_conta_bics, self.max_conta_bics)

        #fit user models
        ncpus = int(multiprocessing.cpu_count()/2) - \
            1 if int(multiprocessing.cpu_count()/2)-1 > 1 else 1
        pool = multiprocessing.Pool(ncpus)
        result_objects = [pool.apply_async(self.fit_user_model, args=(
            [idx, user_k_nearest_bic])) for idx, _ in enumerate(self.users_map)]

        self.user_fitted_model = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        _logger.info(' [%s] trained model - %s', self._timer, str(self))

        del self._timer

    def predict_for_user(self, user, items, ratings=None):
        if user % 100 == 0:
            _logger.debug('predicting %s items for user %s', len(items), user)
        #user nao esta no sistema
        if user not in self.users_map:
            _logger.debug('user %s missing, returning empty predictions', user)
            return pd.Series(np.nan, index=items)

        #get user index
        idx_user = self.users_map.get_loc(user)

        #encontra o modelo do user
        rating_pred = self.user_fitted_model[idx_user].predict_for_user(
            user, items)

        _logger.debug('user %s: predicted for %d of %d items',
                      user, rating_pred.notna().sum(), len(items))
        _logger.debug(rating_pred)
        return rating_pred

    def obtain_user_bics_similarities(self, idx_user, biclustering_solution):
        user_sims = []
        #item rated by active user
        user_items_indexes = np.nonzero(
            self.rating_matrix_csr.row(idx_user))[0].tolist()

        for bic in biclustering_solution:
            users_indexes_bic = sorted(bic.rows)
            items_indexes_bic = sorted(bic.cols)

            items_indexes_interception = sorted(
                set(user_items_indexes) & set(items_indexes_bic))

            #similaridade- intercepcao
            sim_u_b_intercept = len(
                items_indexes_interception) / len(set(items_indexes_bic))

            #sem items em comum
            if sim_u_b_intercept == 0:
                user_sims.append((0, float('inf')))
                continue

            matrix_bic = np.asarray(self.rating_matrix_csr.to_scipy(
            )[np.ix_(users_indexes_bic, items_indexes_bic)].todense())

            #msr-inicial
            msr_bic = self.column_residue(matrix_bic)

            #caso user ja la esteja, msr = ao do bic
            if idx_user in users_indexes_bic:
                user_sims.append((sim_u_b_intercept, 0))
                continue

            #msr-final
            row_user_interception = [self.rating_matrix_csr.row(
                idx_user)[index] for index in items_indexes_bic]
            matrix_user = np.vstack([matrix_bic, row_user_interception])
            msr_bicanduser = self.column_residue(matrix_user)

            #versao so msrfinal
            sim_pattern = msr_bicanduser-msr_bic
            user_sims.append((sim_u_b_intercept, sim_pattern))

        #find nearest bics for the user
        nearest_bics_indexes_interception = sorted(range(len(user_sims)),
                                                   key=lambda k: user_sims[k][0], reverse=True)
        nearest_bics_indexes_pattern = sorted(range(len(user_sims)),
                                              key=lambda k: user_sims[k][1], reverse=False)

        result = dict()
        for pos1_bic, bic in enumerate(nearest_bics_indexes_interception):
            result[bic] = pos1_bic + nearest_bics_indexes_pattern.index(bic)

        #user_k_nearest_bics_indexes = [k for k, v in sorted(result.items(), key=lambda item: item[1])][:self.nnbics]

        #ideia de numeros de bics dinamicos, depois testar
        #versao percetagem dos items dos biclusters
        user_k_nearest_bics_indexes = []
        items_covered = set()
        for k, v in sorted(result.items(), key=lambda item: item[1]):
            if ((len(items_covered) / self.items_biclust_sol) < 0.7):
                user_k_nearest_bics_indexes.append(k)
                items_covered.update(biclustering_solution[k].cols)
            else:
                break
        print("Para o user", idx_user, "usou", len(
            user_k_nearest_bics_indexes), "bics")
        self.conta_bics += len(user_k_nearest_bics_indexes)
        print(self.conta_bics)
        if len(user_k_nearest_bics_indexes) < self.min_conta_bics:
            self.min_conta_bics = len(user_k_nearest_bics_indexes)
        if len(user_k_nearest_bics_indexes) > self.max_conta_bics:
            self.max_conta_bics = len(user_k_nearest_bics_indexes)

        #comeca com o user, assumindo que faz sentido o user estar no bicluster
        bic_result_rows = set([idx_user])
        bic_result_cols = set()
        for bic_index in user_k_nearest_bics_indexes:
            bic_result_rows = bic_result_rows.union(
                set(biclustering_solution[bic_index].rows))
            bic_result_cols = bic_result_cols.union(
                set(biclustering_solution[bic_index].cols))

        print("Criou a matrix de dados thread do user", idx_user,
              " - dims do bic setup:", len(bic_result_rows), len(bic_result_cols))
        return (idx_user, self.bicluster_to_df([sorted(bic_result_rows), sorted(bic_result_cols)]))

    def fit_user_model(self, idx_user, user_k_nearest_bic):
        user_model = util.clone(self.algo).fit(user_k_nearest_bic[idx_user])
        if idx_user % 100 == 0:
            _logger.info("Fez o modelo do user %d", idx_user)
        return (idx_user, user_model)

    def column_residue(self, matrix):
        matrix_copy = np.copy(np.where(matrix != 0, matrix, np.nan))
        cols_nan_means = np.nanmean(matrix_copy, 0)
        residue_sum = 0
        for idx_row, row in enumerate(matrix_copy):
            for idx_col, elem in enumerate(row):
                if not np.isnan(elem):
                    residue_sum += (elem - cols_nan_means[idx_col])**2
        return(1/(np.count_nonzero(~np.isnan(matrix_copy))) * residue_sum)

    def bicluster_to_df(self, bicluster):
        rows_bicluster = bicluster[0]
        cols_bicluster = bicluster[1]
        df_bicluster = pd.DataFrame(columns=["user", "item", "rating"])
        for row_idx in rows_bicluster:
            for col_idx in cols_bicluster:
                user = self.users_map[row_idx]
                item = self.items_map[col_idx]
                rating = self.rating_matrix_csr.to_scipy()[row_idx, col_idx]
                if rating != 0:
                    df_bicluster = df_bicluster.append(
                        {"user": user, "item": item, "rating": rating}, ignore_index=True)
        return df_bicluster.astype({"user": int, "item": int, "rating": float})

    def __str__(self):
        return 'USBCF_CMR_AdaptiveNNbics({},{},{},{},{})'.format(self.nnbics, self.min_num_biclusters, self.min_cols,
                                                                 self.consistency, self.max_overlap)
