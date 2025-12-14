# Lamda


# LAMDA

This repository contains pseudocode and implementation for the submitted paper:

**LAMDA: Two-Phase Multi-Fidelity HPO via Learning Promising Regions from Data**

The code has been anonymized to comply with the review process.

## Implemented Methods

We currently provide implementations of the following hyperparameter optimization (HPO) methods:

- **LAMDA + BOHB** and **BOHB**
- **LAMDA + MUMBO** and **MUMBO**
- **LAMDA + BO** and **BO**
- **LAMDA + RS** and **RS**

These methods are implemented in a unified framework to enable fair and reproducible comparisons between single-fidelity, multi-fidelity, and LAMDA-enhanced optimization strategies.

## Dependencies

The following open-source packages are used in this repository:

- **HpBandSter** for BOHB and Random Search (RS):  
  https://github.com/automl/HpBandSter

- **Emukit** for MUMBO (multi-fidelity Bayesian optimization):  
  https://github.com/EmuKit/emukit/blob/main/notebooks/Emukit-tutorial-multi-fidelity-MUMBO-Example.ipynb

- **GPyOpt** for Bayesian Optimization (BO):  
  https://github.com/SheffieldML/GPyOpt

## Citation

If you use this code in your research, please cite the following paper:








This repository contains pseudocode and implementation for the submitted paper "LAMDA: Two-Phase Multi-Fidelity HPO via Learning Promising Regions from Data". The code has been anonymized to comply with the review process.

Currently, we provide implementations for the following methods used in hyperparameter optimization (HPO):

Lamda+BOHB and BOHB
Lamda+MUMBO and MUMBO
Lamda+BO and BO
Lamda+RS and RS
We utilized the following packages for our implementations:

HpBandSter for BOHB and RS: https://github.com/automl/HpBandSter
Emukit for MUMBO: https://github.com/EmuKit/emukit/blob/main/notebooks/Emukit-tutorial-multi-fidelity-MUMBO-Example.ipynb
GPyOpt for Bayesian Optimization (BO): https://github.com/SheffieldML/GPyOpt
