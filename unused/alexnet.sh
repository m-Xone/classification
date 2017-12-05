#!/bin/bash
#SBATCH --job-name="alexnet2" #Name of the job which appears in squeue
#SBATCH --mail-type=ALL #What notifications are sent by email
#SBATCH --mail-user=wty5dn@virginia.edu
#SBATCH --get-user-env
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --error="alexnet2.err"                    # Where to write std err
#SBATCH --output="alexnet2.output"                # Where to write stdout
#SBATCH --gres=gpu

srun python2.7 alexnet.py
