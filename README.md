# HPAFS

Version: v1.0

MATLAB implementation of harmonic point attention-based feature selection (HPAFS) for residual oxygen concentration measurement using second harmonic signals.

HPAFS uses HPA-MOSTA, a multi-objective state transition algorithm guided by harmonic point attention, to search for feature subsets that balance modeling error and feature selection rate. The final subset is selected using repeated stratified cross-validation and weighted rank aggregation, and is then used to develop the concentration measurement model.

Related manuscript: *Harmonic Point Attention: Guiding Feature Selection for Residual Oxygen Measurement in Pharmaceutical Vials*.

Authors: Mingrui Li, Yonggang Li, Qiwu Luo, Chen Hu, Yibo Wang, and Chunhua Yang.

Affiliation: School of Automation, Central South University, Changsha 410083, China.

## Requirements

- MATLAB (tested with R2024a)
- Statistics and Machine Learning Toolbox

## Usage

1. Set the repository root folder as the MATLAB Current Folder.
2. Add this folder and its subfolders to the MATLAB search path.
3. Open `HPAFS_main.m` and click **Run**.

The optimization settings and HPA-MOSTA parameters are defined near the beginning of `HPAFS_main.m`. The script reports the selected feature point indices, optimization runtime, hypervolume (HV), and training and measurement performance metrics in the Command Window.
