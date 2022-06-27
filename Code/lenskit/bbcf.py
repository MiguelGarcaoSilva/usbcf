import logging
import sys
import pandas as pd
import numpy as np
from lenskit.algorithms import Predictor, item_knn
from lenskit import util, matrix
from biclustering.qubic import QUBIC, QUBIC2
import multiprocessing
import statistics


_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler(sys.stdout))
_logger.setLevel(level=logging.INFO)


class BBCF(Predictor):
    """
    My implementation of BBCF - Impact of biclustering on the performance of Biclustering based
    Collaborative Filtering . https://www.sciencedirect.com/science/article/abs/pii/S0957417418303476
    
    Args:
        number_of_nearest_bics:
        num_biclusters:
        min_cols:
        consistency:
        max_overlap:
    """
    _timer = None

    def __init__(self, number_of_nearest_bics=5, nnbrs=10, num_biclusters=10000,
                 min_cols=2, consistency=1, max_overlap=0.99):
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
        _logger.info('starting BBCF-train')
        self._timer = util.Stopwatch()

        self.rating_matrix_csr, self.users_map,  self.items_map = matrix.sparse_ratings(
            ratings)

        #preprocessamento
        rating_matrix_dense = self.rating_matrix_csr.to_scipy().todense()
        qubic = None
        qubic = QUBIC(num_biclusters=self.min_num_biclusters,
                      discreteFlag=False, minCols=self.min_cols,
                      consistency=self.consistency,
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
        result_objects = [pool.apply_async(self.obtain_user_bics_simmilarities, args=(
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
        return self

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

    # calcula similaridade user com os bics

    def obtain_user_bics_simmilarities(self, idx_user, biclustering_solution):
        user_sims = list()
        #Obter items do user
        user_items_indexes = np.nonzero(
            self.rating_matrix_csr.row(idx_user))[0].tolist()
        for bic in biclustering_solution:
            users_indexes_bic = bic.rows
            items_indexes_bic = bic.cols
            #calcula similaridade
            sim_u_b = len(list(set(user_items_indexes) & set(
                items_indexes_bic))) / len(set(items_indexes_bic))
            weight_u_b = sim_u_b * len(users_indexes_bic)
            user_sims.append(weight_u_b)

        #find k nearest bics for the user
        user_k_nearest_bics_indexes = sorted(
            range(len(user_sims)), key=lambda k: user_sims[k], reverse=True)[:self.nnbics]

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
        print("Fez o modelo do user", idx_user)
        return (idx_user, user_model)

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
        return 'BBCF({},{},{},{},{},{})'.format(self.nnbics, getattr(self.algo, "nnbrs"), self.min_num_biclusters, self.min_cols,
                                                self.consistency, self.max_overlap)


class BBCF_noweight(Predictor):
    """
    My implementation of BBCF - Impact of biclustering on the performance of Biclustering based
    Collaborative Filtering . https://www.sciencedirect.com/science/article/abs/pii/S0957417418303476
    
    Args:
        number_of_nearest_bics:
        num_biclusters:
        min_cols:
        consistency:
        max_overlap:
    """
    _timer = None

    def __init__(self, number_of_nearest_bics=5, nnbrs=10, num_biclusters=10000, min_cols=2, consistency=1, max_overlap=0.99):
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
        _logger.info('starting BBCF_noweight-train')
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
        result_objects = [pool.apply_async(self.obtain_user_bics_simmilarities, args=(
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
        pool = multiprocessing.Pool(int(multiprocessing.cpu_count()/2)-1)
        result_objects = [pool.apply_async(self.fit_user_model, args=(
            [idx, user_k_nearest_bic])) for idx, _ in enumerate(self.users_map)]

        self.user_fitted_model = dict([r.get() for r in result_objects])

        pool.close()
        pool.join()

        _logger.info(' [%s] trained model - %s', self._timer, str(self))

        del self._timer
        return self

    def predict_for_user(self, user, items, ratings=None):
        if user % 100 == 0:
            _logger.info('predicting %s items for user %s', len(items), user)
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
        _logger.info(rating_pred)
        return rating_pred

    # calcula similaridade user com os bics
    def obtain_user_bics_simmilarities(self, idx_user, biclustering_solution):
        user_sims = list()
        #Obter items do user
        user_items_indexes = np.nonzero(
            self.rating_matrix_csr.row(idx_user))[0].tolist()
        for bic in biclustering_solution:
            users_indexes_bic = bic.rows
            items_indexes_bic = bic.cols
            #calcula similaridade
            sim_u_b = len(list(set(user_items_indexes) & set(
                items_indexes_bic))) / len(set(items_indexes_bic))
            user_sims.append(sim_u_b)

        #find k nearest bics for the user
        user_k_nearest_bics_indexes = sorted(
            range(len(user_sims)), key=lambda k: user_sims[k], reverse=True)[:self.nnbics]

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
        print("Fez o modelo do user", idx_user)
        return (idx_user, user_model)

    def bicluster_to_df(self, bicluster):
        rows_bicluster = bicluster[0]
        cols_bicluster = bicluster[1]
        df_bicluster = pd.DataFrame(columns=["user", "item", "rating"])
        for row_idx in rows_bicluster:
            for col_idx in cols_bicluster:
                user = self.users_map[row_idx]
                item = self.items_map[col_idx]
                rating = self.rating_matrix_csr.to_scipy()[row_idx, col_idx]
                if(rating != 0):
                    df_bicluster = df_bicluster.append(
                        {"user": user, "item": item, "rating": rating}, ignore_index=True)
        return df_bicluster.astype({"user": int, "item": int, "rating": float})

    def __str__(self):
        return 'BBCF_noweight({},{},{},{},{},{})'.format(self.nnbics, getattr(self.algo, "nnbrs"), self.min_num_biclusters, self.min_cols,
                                                         self.consistency, self.max_overlap)
class NotEnoughBicsError(Exception):
    pass
