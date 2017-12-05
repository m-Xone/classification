#!/bin/bash
#SBATCH --job-name="vgg16" #Name of the job which appears in squeue
#SBATCH --mail-type=ALL #What notifications are sent by email
#SBATCH --mail-user=wty5dn@virginia.edu
#SBATCH --get-user-env
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --error="vgg16.err"                    # Where to write std err
#SBATCH --output="vgg16.output"                # Where to write stdout
#SBATCH --gres=gpu

srun python2.7 vgg16.py
