import numpy as np
from scipy import stats


class Bicluster:
    """This class models a bicluster.

    Parameters
    ----------
    rows : numpy.array
        Rows of the bicluster (assumes that row indexing starts at 0).

    cols : numpy.array
        Columns of the bicluster (assumes that column indexing starts at 0).

    data : numpy.ndarray
    """

    def __init__(self, rows, cols, data=None, pvalue=None):
        if isinstance(rows, np.ndarray) and rows.dtype == np.bool and cols.dtype == np.bool:
            self.rows = np.nonzero(rows)[0]
            self.cols = np.nonzero(cols)[0]
        elif isinstance(cols, np.ndarray) and rows.dtype == np.int and cols.dtype == np.int:
            self.rows = rows
            self.cols = cols
        else:
            raise ValueError("rows and cols must be bool or int numpy.arrays")

        if data is not None:
            n, m = len(self.rows), len(self.cols)

            if isinstance(data, np.ndarray) and (data.shape == (n, m) or (len(data) == 0 and n == 0)):
                self.data = data
            else:
                raise ValueError("")
        self.pvalue = pvalue

    def set_pvalue(self, pvalue):
        self.pvalue = pvalue

    def intersection(self, other):
        """Returns a bicluster that represents the area of overlap between two biclusters."""
        rows_intersec = np.intersect1d(self.rows, other.rows)
        cols_intersec = np.intersect1d(self.cols, other.cols)
        return Bicluster(rows_intersec, cols_intersec)

    def union(self, other):
        rows_union = np.union1d(self.rows, other.rows)
        cols_union = np.union1d(self.cols, other.cols)
        return Bicluster(rows_union, cols_union)

    def overlap(self, other):
        min_area = min(self.area, other.area)
        return self.intersection(other).area / min_area

    @property
    def area(self):
        """Calculates the number of matrix elements of the bicluster."""
        return len(self.rows) * len(self.cols)

    def sort(self):
        """Sorts the array of row and the array of column indices of the bicluster."""
        self.rows.sort()
        self.cols.sort()

    def __str__(self):
        return 'Bicluster(rows={0}, cols={1})'.format(self.rows, self.cols)


class Biclustering:

    def __init__(self, biclusters):
        if all(isinstance(b, Bicluster) for b in biclusters):
            self.biclusters = biclusters
        else:
            raise ValueError("element is not a Bicluster")

    def run_constant_freq_column(self, data_matrix, labels,
                                 missing_correction):
        data_matrix = np.asarray(data_matrix)
        n_rows = data_matrix.shape[0]
        n_cols = data_matrix.shape[1]
        freq_cols = np.zeros((n_cols, len(labels)))
        for col_idx in range(n_cols):
            count = 0
            values, counts = np.unique(data_matrix[:, col_idx],
                                       return_counts=True)
            freqs_labels = []
            for v in labels:
                if v in values:
                    freqs_labels.append(counts[np.where(values == v)[0][0]])
                    count += counts[np.where(values == v)[0][0]]
                else:
                    freqs_labels.append(0)
            if missing_correction:
                count = n_rows
            if count != 0:
                freqs_labels = np.asarray(freqs_labels)/count
            freq_cols[col_idx] = freqs_labels
        for bic in self.biclusters:
            p = 1
            j = 0
            for col in bic.cols:
                p = p * freq_cols[col, labels.index(int(bic.data[0][j]))]
                j += 1
            bic.pvalue = stats.binom.sf(len(bic.rows), n_rows, p)
        return self

    def __str__(self):
        return '\n'.join(str(b) for b in self.biclusters)
