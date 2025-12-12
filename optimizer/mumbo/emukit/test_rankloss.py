# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:46:27 2022

@author: fl347
"""

import numpy as np
from emukit.rank_loss import rank_loss

x=[1,4,3]
y=[2,5,7]



loss,par_ri,par_all=rank_loss(x,y)
