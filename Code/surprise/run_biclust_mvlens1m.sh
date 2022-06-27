#!/bin/bash
#SBATCH --job-name=test_biclust_mvlens # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mmgsilva@fc.ul.pt     # Where to send mail
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task / CPU
#SBATCH --cpus-per-task=40            # Number of cores per task
#SBATCH --mem=40gb                    # Job memory request
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log     # Standard output and error log

echo "Running script "

python3 test_biclust_mvlens1m.py> prints_biclust_mvlens1m

date
