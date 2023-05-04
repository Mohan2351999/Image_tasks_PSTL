#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1  # request GPU "generic resource"
#SBATCH --ntasks-per-node=8   # maximum CPU coresper GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # memory per node
#SBATCH --time=00-10:59      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-type=ALL
#SBATCH -A def-guzdial
#SBATCH --mail-user=singamse@ualberta.ca
#SBATCH --job-name="STL10 to CIFAR10 search transfer"


module load python
module load cuda/11.4
module load cudnn/8.2.0
source ~/jupyter_py3/bin/activate

python3 /home/mohan235/projects/def-guzdial/mohan235/1_Mohan_GRAF_Work/Image_tasks_PSTL/STL10_to_CIFAR10/STL10_to_CIFAR10_search_transfer.py >> STL10_to_CIFAR10_search_transfer_log.txt 2>&1