from .bicluster import Bicluster, Biclustering
from .wrapper import ExecutableWrapper
from os.path import dirname, join

import numpy as np
import re
import os


class QUBIC(ExecutableWrapper):
    """QUBIC 1.0: greedy biclustering (compiled Jan 26 2021 00:05:12)

    ===================================================================
    [Usage]
    $ ./qubic -i filename [argument list]
    ===================================================================
    [Input]
    -i : input file must be one of two tab-delimited formats
      A) continuous data (default, use pre-set discretization (see -q and -r))
         -------------------------------------
         o        cond1    cond2    cond3
         gene1      2.4      3.5     -2.4
         gene2     -2.1      0.0      1.2
         -------------------------------------
      B) discrete data with arbitray classes (turn on -d)
         use '0' for missing or insignificant data
         -------------------------------------
         o        cond1    cond2    cond3
         gene1        1        2        2
         gene2       -1        2        0
         -------------------------------------
    -q : use quantile discretization for continuous data
         default: 0.06 (see details in Method section in paper)
    -r : the number of ranks as which we treat the up(down)-regulated value
         when discretization
         default: 1
    -d : discrete data, where user should send their processed data
         to different value classes, see above
    -b : the file to expand in specific environment
    -T : to-be-searched TF name, just consider the seeds containing current TF
         default format: B1234
    -P : the flag to enlarge current biclsuter by the pvalue constrain
    -S : the flag using area as the value of bicluster to determine when stop
    -C : the flag using the lower bound of condition number
        (5 persents of the gene number)
    -l : the list of genes out of the input file on which we do bicluster
    ===================================================================
    [Output]
    -o : number of blocks to report, default: 100
    -f : filtering overlapping blocks,
         default: 1 (do not remove any blocks)
    -k : minimum column width of the block,
         default: 5% of columns, minimum 2 columns
    -c : consistency level of the block (0.5-1.0], the minimum ratio between
     the number of identical valid symbols in a column and the total number
         of rows in the output
         default: 0.95
    -s : expansion flag
    ===================================================================
    """

    def __init__(self, discreteFlag=True, quant=0.06, ranks=1,
                 num_biclusters=10, minCols=0, consistency=0.95,
                 max_overlap_level=1.0):

        super().__init__(output_filename='data.txt.blocks')
        self.discreteFlag = discreteFlag
        self.quant = quant
        self.ranks = ranks

        self.num_biclusters = num_biclusters
        self.max_overlap_level = max_overlap_level
        self.minCols = minCols
        self.consistency = consistency

    def _get_command(self, data, data_path, output_path):
        comm = [join(dirname(__file__), 'bin', 'qubic')]

        comm = comm + ["-i", data_path] + ["-o", str(self.num_biclusters)]

        if self.discreteFlag:
            comm = comm + ["-d"]
        else:
            comm = comm + ["-q", str(self.quant)] + ["-r", str(self.ranks)]
        if self.max_overlap_level != 1:
            comm = comm + ["-f", str(self.max_overlap_level)]
        if self.minCols != 0:
            comm = comm + ["-k", str(self.minCols)]
        if self.consistency != 0.95:
            comm = comm + ["-c", str(self.consistency)]

        return comm

    def _write_data(self, data_path, data):
        header = 'p\t' + '\t'.join(str(i) for i in range(data.shape[1]))
        row_names = np.char.array([str(i) for i in range(data.shape[0])])
        data = data.astype(np.str)
        data = np.hstack((row_names[:, np.newaxis], data))

        with open(data_path, 'wb') as f:
            np.savetxt(f, data, delimiter='\t', header=header, fmt='%s',
                       comments='')

    def _parse_output(self, output_path):
        biclusters = []

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                content = f.read()
                bc_strings = re.split('BC[0-9]+', content)[1:]
                biclusters.extend(self._parse_bicluster(b) for b in bc_strings)

        return Biclustering(biclusters)

    def _parse_bicluster(self, bicluster_str):
        content = re.split('Genes \[[0-9]+\]:', bicluster_str).pop()

        rows, content = re.split('Conds \[[0-9]+\]:', content)
        rows = np.array(rows.split(), dtype=np.int)

        cols = content.split('\n')[0]
        cols = np.array(cols.split(), dtype=np.int)

        pattern = content.split('\n')[1].split("\t")[1:]
        data = np.zeros((len(rows), len(cols)))
        for row in data:
            row[:] = pattern
        return Bicluster(rows, cols, data)

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError(
                "num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.ranks <= 0:
            raise ValueError("ranks must be > 0, got {}".format(self.ranks))

        if self.quant <= 0.0 or self.quant >= 1.0:
            raise ValueError(
                "quant must be > 0.0 and < 1.0, got {}".format(self.quant))

        if self.consistency <= 0.0 or self.consistency > 1.0:
            raise ValueError(
                "consistency must be > 0.0 and <= 1.0, got {}".format(self.consistency))

        if self.max_overlap_level <= 0.0 or self.max_overlap_level > 1.0:
            raise ValueError("max_overlap_level must be > 0.0 and <= 1.0, got {}".format(
                self.max_overlap_level))

    def __str__(self):
        return 'QUBIC({},{},{},{},{})'.format(self.discreteFlag, self.num_biclusters,
                                              self.minCols, self.consistency,
                                              self.max_overlap_level)


class QUBIC2(ExecutableWrapper):
    """QUBIC 2.2: greedy biclustering (compiled Jan 24 2021 00:03:03)
    ===================================================================
    [Usage]
    $ ./qubic -i filename [argument list]
    ===================================================================
    [Input]
    -i : input file must be one of two tab-delimited formats
      A) continuous data (default, use pre-set discretization (see -q and -r))
         -------------------------------------
         o        cond1    cond2    cond3
         gene1      2.4      3.5     -2.4
         gene2     -2.1      0.0      1.2
         -------------------------------------
      B) discrete data with arbitray classes (turn on -d)
         use '0' for missing or insignificant data
         -------------------------------------
         o        cond1    cond2    cond3
         gene1        1        2        2
         gene2       -1        2        0
         -------------------------------------
    -d : the flag to analyze discrete data, where user should discretize their
         data to different classes of value, see B) above
         default: FALSE
    -b : a .blocks file to be expanded in a specific .chars file
    -s : the flag of doing expansion, used together with -b
         default: FALSE
    ===================================================================
    [Discretization]
    -F : the flag to only do discretization without biclustering
    -q : use quantile discretization for continuous data
         default: 0.06 (see details in Method section in paper)
    -r : the number of ranks as which we treat the up(down)-regulated value
         when discretization
         default: 1
    -n : the flag to discretize the continuous values by a mixture normal
         distribution model
         default: FALSE
    -R : the flag to discretize the RPKM values by a mixture normal
         distribution model
         default: FALSE
    -e : the number of iterations in EM algorithm when using -n or -R
         default: FALSE
    ===================================================================
    [Biclustering]
    -f : filtering overlapping blocks,
         default: 1 (do not remove any blocks)
    -k : minimum column width of the block,
         default: 5% of columns, minimum 2 columns
    -c : consistency level of the block (0.5-1.0], the minimum ratio between
         the number of identical valid symbols in a column and the total number
         of rows in the output
         default: 1.0
    -p : the flag to calculate the spearman correlation between any pair of
         genes this can capture more reliable relationship but much slower
         default: FALSE
    -C : the flag using the lower bound of condition number
         default: 5% of the gene number in current bicluster
    -N : the flag using 1.0 biclustering,i.e., maximize min(|I|,|J|)
    ===================================================================
    [Output]
    -o : number of blocks to report
         default: 100
    ===================================================================
    """

    def __init__(self, discreteFlag=True, quant=0.06, ranks=1,
                 num_biclusters=10, minCols=0, consistency=0.95,
                 max_overlap_level=1.0):

        super().__init__(output_filename='data.txt.blocks')
        self.discreteFlag = discreteFlag
        self.quant = quant
        self.ranks = ranks

        self.num_biclusters = num_biclusters
        self.max_overlap_level = max_overlap_level
        self.minCols = minCols
        self.consistency = consistency

    def _get_command(self, data, data_path, output_path):
        comm = [join(dirname(__file__), 'bin', 'qubic2')]

        comm = comm + ["-i", data_path] + ["-o", str(self.num_biclusters)]
        if self.discreteFlag:
            comm = comm + ["-d"]
        else:
            comm = comm + ["-q", str(self.quant)] + ["-r", str(self.ranks)]
        if self.max_overlap_level != 1:
            comm = comm + ["-f", str(self.max_overlap_level)]
        if self.minCols != 0:
            comm = comm + ["-k", str(self.minCols)]
        if self.consistency != 0.95:
            comm = comm + ["-c", str(self.consistency)]

        return comm

    def _write_data(self, data_path, data):
        header = 'p\t' + '\t'.join(str(i) for i in range(data.shape[1]))
        row_names = np.char.array([str(i) for i in range(data.shape[0])])
        data = data.astype(np.str)
        data = np.hstack((row_names[:, np.newaxis], data))

        with open(data_path, 'wb') as f:
            np.savetxt(f, data, delimiter='\t',
                       header=header, fmt='%s', comments='')

    def _parse_output(self, output_path):
        biclusters = []

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                content = f.read()
                bc_strings = re.split('BC[0-9]+', content)[1:]
                biclusters.extend(self._parse_bicluster(b) for b in bc_strings)

        return Biclustering(biclusters)

    def _parse_bicluster(self, bicluster_str):
        content = re.split('Genes \[[0-9]+\]:', bicluster_str).pop()
        rows, content = re.split('Conds \[[0-9]+\]:', content)
        rows = np.array(rows.split(), dtype=np.int)

        cols = content.split('\n')[0]
        cols = np.array(cols.split(), dtype=np.int)

        pattern = content.split('\n')[1].split("\t")[1:]
        data = np.zeros((len(rows), len(cols)))
        for row in data:
            row[:] = pattern
        return Bicluster(rows, cols, data)

    def __str__(self):
        return 'QUBIC2({},{},{},{},{})'.format(self.discreteFlag, self.num_biclusters,
                                               self.minCols, self.consistency,
                                               self.max_overlap_level)
