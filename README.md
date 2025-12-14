# LAMDA

This repository contains pseudocode and implementation for the paper:

**LAMDA: Two-Phase Multi-Fidelity HPO via Learning Promising Regions from Data**  
F. Li, S. Wang, K. Li  
*Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)*

## Overview

LAMDA is a two-phase multi-fidelity hyperparameter optimization (HPO) framework that learns promising regions from historical evaluation data and focuses expensive high-fidelity evaluations on these regions.  
The proposed method is compatible with both Bayesian optimization and surrogate-assisted optimization frameworks.

## Implemented Methods

We provide implementations of the following HPO methods:

- **LAMDA + BOHB** and **BOHB**
- **LAMDA + MUMBO** and **MUMBO**
- **LAMDA + BO** and **BO**
- **LAMDA + RS** and **RS**

All methods are implemented in a unified framework to ensure fair and reproducible comparisons.

## Dependencies

The following open-source libraries are used:

- **HpBandSter** for BOHB and Random Search (RS):  
  https://github.com/automl/HpBandSter

- **Emukit** for MUMBO (multi-fidelity Bayesian optimization):  
  https://github.com/EmuKit/emukit

- **GPyOpt** for Bayesian Optimization (BO):  
  https://github.com/SheffieldML/GPyOpt

## Citation

If you use this code, please cite:

@inproceedings{li2026lamda,
title = {LAMDA: Two-Phase Multi-Fidelity HPO via Learning Promising Regions from Data},
author = {Li, F. and Wang, S. and Li, K.},
booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
year = {2026}
}


## License

This project is released for research purposes. Please refer to the LICENSE file for details.







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
