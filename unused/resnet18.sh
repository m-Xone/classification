#!/bin/bash
#SBATCH --job-name="resnet18" #Name of the job which appears in squeue
#SBATCH --mail-type=ALL #What notifications are sent by email
#SBATCH --mail-user=wty5dn@virginia.edu
#SBATCH --get-user-env
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --error="resnet18.err"                    # Where to write std err
#SBATCH --output="resnet18.output"                # Where to write stdout
#SBATCH --gres=gpu

srun python2.7 resnet18.py
