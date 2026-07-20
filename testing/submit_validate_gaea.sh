#!/bin/bash
#SBATCH --job-name=mpas_val
#SBATCH --account=bil-fire3
#SBATCH --clusters=c6
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/testing/mpas_val.%j.out
#SBATCH --error=/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/testing/mpas_val.%j.err

# Load required modules
module purge
module load ufs_gaeac6.intelllvm

# Unset conflicting legacy environment flags
unset CFLAGS CXXFLAGS FFLAGS
export CRAYPE_LINK_TYPE=dynamic

# Define benchmark parameters and custom data paths
export JW_MESH="/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/test_data/jw_baroclinic_240km/x1.40962.init.nc"
export JW_GRAPH="/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/test_data/jw_baroclinic_240km/x1.40962.graph.info.part.48"
export WORKDIR="/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/jw_validation_run"

# Configure parallel launcher for Gaea compute nodes
export NPROCS=48
export MPI_RUN_CMD="srun -n 48"
export OMP_NUM_THREADS=1

# Execute validation script
/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/testing/validate_cpp_dycore.sh
