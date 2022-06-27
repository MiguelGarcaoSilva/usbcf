from time import sleep

from sklearn.utils.validation import check_array

from .bicluster import  Biclustering

import os, shutil, tempfile, subprocess
import numpy as np



class ExecutableWrapper():
    """ create a temporary directory, save the input data
    as a txt file, run the wrapped algorithm, parse the output files and remove the
    temporary directory.

    Parameters
    ----------
    exec_comm : str
        The command line command to run the wrapped executable.

    tmp_dir : str
        Temporary directory path, where temporary files will be stored.

    sleep : bool, default: True
        Whether to make a 1 second delay before running the wrapped executable.

    data_type : numpy.dtype, default: numpy.double
        The input data type required by the algorithm.
    """

    def __init__(self, data_filename='data.txt', output_filename='output.txt', data_type=np.double, sleep=1):
        super().__init__()

        self._data_filename = data_filename
        self._output_filename = output_filename
        self._data_type = data_type
        self._sleep = sleep

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """
        self._validate_parameters()
        data = check_array(data, dtype=self._data_type, copy=True)

        tmp_dir = tempfile.mkdtemp()

        data_path = os.path.join(tmp_dir, self._data_filename)
        output_path = os.path.join(tmp_dir, self._output_filename)

        self._write_data(data_path, data)
        sleep(self._sleep)
        comm = self._get_command(data, data_path, output_path)

        try:
            print(comm)
            subprocess.check_call(comm, stderr=subprocess.STDOUT)
            biclustering = self._parse_output(output_path)

        except subprocess.CalledProcessError as error:
            print('The following error occurred while running the command {}:\n{}'.format(comm, error.output))
            print('Returning empty biclustering solution.')
            biclustering = Biclustering([])

        shutil.rmtree(tmp_dir)

        return biclustering


    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.ranks <= 0:
            raise ValueError("ranks must be > 0, got {}".format(self.ranks))

        if self.quant <= 0.0 or self.quant >= 1.0:
            raise ValueError("quant must be > 0.0 and < 1.0, got {}".format(self.quant))

        if self.consistency <= 0.0 or self.consistency > 1.0:
            raise ValueError("consistency must be > 0.0 and <= 1.0, got {}".format(self.consistency))

        if self.max_overlap_level <= 0.0 or self.max_overlap_level > 1.0:
            raise ValueError("max_overlap_level must be > 0.0 and <= 1.0, got {}".format(self.max_overlap_level))
