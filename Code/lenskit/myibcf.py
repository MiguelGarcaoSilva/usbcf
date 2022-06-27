import logging
import sys
import pandas as pd
import numpy as np
from heapq import nlargest
from lenskit.algorithms import Predictor
from lenskit import util, matrix
from scipy.sparse import csr_matrix, lil_matrix, diags
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler(sys.stdout))
_logger.setLevel(level=logging.INFO)

class MyItemBasedImplementation(Predictor):
    """
    Args:
        nnbrs: the maximum number of neighbors for scoring each item.
        min_nbrs: the minimum number of neighbors for scoring each item.
        min_sim: minimum similarity threshold for considering a neighbor.
        sim_metric: name of the similarity metric to use. options : "cosine", "adjusted_cosine", "pearson".
        aggregate: the type of aggregation to do. options : "weighted-average".
    
    Attributes:
        items_map(pandas.Index): the index of item IDs. eg: 0,1
        users_map(pandas.Index): the index of known user IDs. 0,1
        _sim_matrix(dict): the similarities dict. eg:
        _neighbors(dict): the neighbors dict. eg:
        rating_matrix_csr(matrix.CSR): the user-item rating matrix for looking up users' ratings.
    """

    timer = None

    def __init__(self, nnbrs=None, min_nbrs=1, min_sim=0.0000001, sarwar=False, sim_metric = "adjusted_cosine",
                 aggregate="weighted-average", use_average=False):

        self.nnbrs = nnbrs
        self.min_nbrs = min_nbrs
        self.min_sim = min_sim
        #keep only k neighbors for item? or all and then filter?
        self.sarwar = sarwar
        self.use_average = use_average
        if sim_metric == 'adjusted_cosine':
            self.sim_metric_fn = self.adjusted_cosine_sim
        elif sim_metric == 'mean_centered_items_cosine':
            self.sim_metric_fn = self.mean_centered_items_cosine
        elif sim_metric == 'cosine':
            self.sim_metric_fn = self.cosine_sim
        elif sim_metric == "cosine_corates":
            self.sim_metric_fn = self.cosine_sim_corates
        else:
            raise Exception(f'There is no similarity metric corresponding to the name "{sim_metric}".')
        self.aggregate = aggregate
        
        #mapping row/col index pos to users/items ids.            
        self.users_map = None
        self.items_map = None
        #rating matrix
        self.rating_matrix_csr = None
        #similarity matrix
        self._sim_matrix = None
        #neighbors dict
        self._neighbours = dict()
            
    def fit(self, ratings, **kwargs):
        """
        Train a model.
        Args:
            ratings(pandas.DataFrame):
                (user,item,rating) data for computing item similarities.
        """
        self._timer = util.Stopwatch()
        _logger.info('  [%s] beginning fit, memory use %s', self._timer, util.max_memory())


        self.rating_matrix_csr, self.users_map, self.items_map = matrix.sparse_ratings(ratings)
        
        if self.nnbrs == None:
            self.nnbrs = len(self.items_map)

        _logger.info('[%s] made sparse matrix for %d items (%d ratings from %d users)',
                     self._timer, len(self.items_map), self.rating_matrix_csr.nnz, len(self.users_map))
        
        _logger.info(' [%s] computing similarity matrix', self._timer)
        self._sim_matrix = self._compute_sim_matrix()
        
        
        if self.sarwar:
            self._neighbours = self._get_neighbors_sarwar()
        else:
            self._neighbours = self._get_neighbors()
        
        _logger.info('[%s] got neighborhoods for %d of %d items',
                     self._timer, len([l[0] for l in self._neighbours.values() if l]), len(self.items_map))

        _logger.debug(' [%s] fitting done, memory use %s', self._timer, util.max_memory())
        return self
    
    
    
    def predict_for_user(self, user, items, ratings=None):
        
        _logger.debug('predicting %d items for user %s', len(items), user)
        #user nao esta no sistema
        if user not in self.users_map:
            _logger.debug('user %s missing, returning empty predictions!!! %s ........ %s', user, self.users_map,self.items_map)
            return pd.Series(np.nan, index=items)
        
        user_pos = self.users_map.get_loc(user)
        items_pos = self.items_map.get_indexer(items)
        # prediction results
        rating_preds = list()
        #por cada item a avaliar
        for item_pos in items_pos:
            ratings, similarities = list(), list()
            #item nao esta no sistema
            if item_pos == -1:
                _logger.debug('item missing!')
                rating_preds.append(np.nan)
                continue
            if not self.sarwar:
                for pos_neighbor_i, sim_neighbor_i in self._neighbours[item_pos]:
                    record_neighbor = self.rating_matrix_csr.to_scipy()[user_pos,pos_neighbor_i]
                    #se deu rating
                    if record_neighbor > 0:
                        similarities.append(sim_neighbor_i)
                        ratings.append(record_neighbor)
                    if len(ratings) >= self.nnbrs:
                        break            
            else:    
                # se nao tem minimo de neighbors
                if len(self._neighbours[item_pos]) < self.min_nbrs:
                    _logger.debug('not enough neighbors for the item!')
                    rating_preds.append(np.nan)
                    continue
                for pos_neighbor_i, sim_neighbor_i in self._neighbours[item_pos]:
                    record_neighbor = self.rating_matrix_csr.to_scipy()[user_pos,pos_neighbor_i]
                    #se nao deu rating
                    if record_neighbor == 0:
                        continue
                    similarities.append(sim_neighbor_i)
                    ratings.append(record_neighbor)
                    
                #se o active nao deu rating a nenhum dos neighbors
                if len(ratings) == 0:
                    if self.use_average:
                        _logger.debug('users without rating in the neighors , using average!')
                        rating_preds.append(self._user_average(user_pos))
                    else:
                        _logger.debug('users without rating in the neighors , returning nan')
                        rating_preds.append(np.nan)
                    continue
            
            pred = self._weighted_average(ratings,similarities)
            rating_preds.append(pred)
        results = pd.Series(rating_preds, index = items).reindex(items, fill_value=np.nan)
        _logger.debug('user %s: predicted for %d of %d items',
                      user, results.notna().sum(), len(items))
        return results
    
    def _compute_sim_matrix(self):
        """Build the similarity matrix."""
        #transpose porque queremos item similarity
        return self.sim_metric_fn(self.rating_matrix_csr.to_scipy())
    

    def _get_neighbors_sarwar(self):
        """return the k nearest neighbors of the item in pos1 (pos2,sim)"""
        k_nearest_neighbors = dict()
        for pos1 in range(len(self.items_map)):
            k_nearest_neighbors[pos1] = nlargest(self.nnbrs, filter(
                    lambda x: x[1] is not None and x[1] >= self.min_sim,
                    [(pos2, self._get_sim(pos1, pos2)) for pos2 in range(len(self.items_map))  if pos2 != pos1]), key = lambda x: x[1])
        return k_nearest_neighbors
    
    def _get_neighbors(self):
        k_nearest_neighbors = dict()
        for pos1 in range(len(self.items_map)):
            k_nearest_neighbors[pos1] = sorted(filter(
                    lambda x: x[1] is not None and x[1] >= self.min_sim,
                    [(pos2, self._get_sim(pos1, pos2)) for pos2 in range(len(self.items_map))  if pos2 != pos1]),reverse=True, key = lambda x: x[1])
        return k_nearest_neighbors  
    
    
    def cosine_sim(self, matrix):          
        return cosine_similarity(matrix.T, dense_output=False)
        
    def adjusted_cosine_sim(self,M):
        mean_rows = np.squeeze(np.asarray(M.mean(axis=1)))
        _logger.info('[%s] computed means for %d users', self._timer, len(mean_rows))
        centered_matrix = M - mean_rows[:,np.newaxis]
        if self.sarwar: #se usa os coratings only
            return self.cosine_sim_corates(csr_matrix(centered_matrix))
        return cosine_similarity(csr_matrix(centered_matrix.T), dense_output=False)
    
    def mean_centered_items_cosine(self,M):
        mean_columns = np.squeeze(np.asarray(M.mean(axis=0)))
        _logger.info('[%s] computed means for %d items', self._timer, len(mean_columns))
        centered_matrix = M - mean_columns
        return cosine_similarity(csr_matrix(centered_matrix.T), dense_output=False)
    
        
    def _get_sim(self, pos1, pos2):
        """Computes the similarity between the provided items/users."""
        return self._sim_matrix[pos1,pos2]
    

    
    def _weighted_average(self, ratings, similarities):
        """Computes the sum of the similarities
        multiplied by the ratings of each neighbour, and then divides this by the sum of
        the similarities of the neighbours."""
        sim_sum, ratings_sum = 0, 0
        for rating, similarity in zip(ratings, similarities):
            ratings_sum += similarity * rating
            sim_sum += similarity  
            
        #se o active nao deu rating a nenhum dos neighbors
        if sim_sum <= 0:
            return np.nan
        return ratings_sum / sim_sum 
    
    def _user_average(self, pos_user):
        sum_scores = self.rating_matrix_csr.to_scipy()[pos_user].sum()
        total_scores = self.rating_matrix_csr.to_scipy()[pos_user].getnnz()
        return  sum_scores / total_scores if total_scores > 0 else np.nan
    

    
    def cosine_sim_corates(self,matrix):
        matrix = matrix.T
        """ by setting
        the denominator to only take into account the co-ratings of 2 comparing users/items. - usado no sarwar"""
        if type(matrix) is not lil_matrix:
            matrix = lil_matrix(matrix)
    
        n = matrix.shape[0]
        rows, cols, data = [], [], []
        user_items = [sorted([(item, idx) for idx, item in enumerate(matrix.rows[i])]) for i in range(n)]
    
        for i in range(n):
            i_ratings, i_items = matrix.data[i], user_items[i]
            for j in range(i, n):
                j_ratings, j_items = matrix.data[j], user_items[j]
                sum_numerator, sum_denominator_i, sum_denominator_j = 0, 0, 0
                i_item_ctd, j_item_ctd = 0, 0
                while i_item_ctd < len(i_items) and j_item_ctd < len(j_items):
                    if i_items[i_item_ctd][0] > j_items[j_item_ctd][0]:
                        j_item_ctd += 1
                    elif i_items[i_item_ctd][0] < j_items[j_item_ctd][0]:
                        i_item_ctd += 1
                    else:
                        i_idx = i_items[i_item_ctd][1]
                        j_idx = j_items[j_item_ctd][1]
                        sum_numerator += i_ratings[i_idx] * j_ratings[j_idx]
                        sum_denominator_i += i_ratings[i_idx] ** 2
                        sum_denominator_j += j_ratings[j_idx] ** 2
                        i_item_ctd += 1
                        j_item_ctd += 1
    
                if sum_numerator == 0: continue
                s = sum_numerator / (math.sqrt(sum_denominator_i) * math.sqrt(sum_denominator_j))
                rows.append(i), cols.append(j), data.append(s)
                if i != j: rows.append(j), cols.append(i), data.append(s)
    
        return csr_matrix((data, (rows, cols)))

    
    
    def __str__(self):
        return 'MYIBKNN(nnbrs={},sim={},sarwar={})'.format(self.nnbrs,self.sim_metric_fn,self.sarwar)
        
        
        
        
        
        
        