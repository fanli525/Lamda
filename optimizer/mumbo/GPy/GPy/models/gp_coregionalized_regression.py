# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP,SparseGP
from .. import likelihoods
from .. import kern
from .. import util

class GPCoregionalizedRegression(GP):
    """
    Gaussian Process model for heteroscedastic multioutput regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X_list: list of input observations corresponding to each output
    :type X_list: list of numpy arrays
    :param Y_list: list of observed values related to the different noise models
    :type Y_list: list of numpy arrays
    :param kernel: a GPy kernel ** Coregionalized, defaults to RBF ** Coregionalized
    :type kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to list of Gaussian likelihoods
    :type likelihoods_list: None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param W_rank: number tuples of the corregionalization parameters 'W' (see coregionalize kernel documentation)
    :type W_rank: integer
    :param kernel_name: name of the kernel
    :type kernel_name: string
    """
    def __init__(self, X_list, Y_list, kernel=None, likelihoods_list=None, name='GPCR',W_rank=1,kernel_name='coreg',normalizer=False):

        #Input and Output
        X,Y,self.output_index = util.multioutput.build_XY(X_list,Y_list)
        Ny = len(Y_list)

        #Kernel
        if kernel is None:
            kernel = kern.RBF(X.shape[1]-1)
            
            kernel = util.multioutput.ICM(input_dim=X.shape[1]-1, num_outputs=Ny, kernel=kernel, W_rank=W_rank,name=kernel_name)

        #Likelihood
        likelihood = util.multioutput.build_likelihood(Y_list,self.output_index,likelihoods_list)
        super(GPCoregionalizedRegression, self).__init__(X,Y, kernel,likelihood, normalizer=normalizer,Y_metadata={'output_index':self.output_index})
        
        # from ..inference.latent_function_inference import var_dtc
        # inference_method = var_dtc.VarDTC(limit=3)
        
        # import random

        # ind =  np.array(np.arange(0, X.shape[0], 1) )
        # random.shuffle(ind)
        # n1=np.min([1*X.shape[0],100])
        # ind2=ind[0:int(n1)]
        # super(GPCoregionalizedRegression, self).__init__(X,Y,X[ind2,:],kernel,likelihood, normalizer=normalizer,Y_metadata={'output_index':self.output_index})


    # def __init__(self, X, Y, kernel, likelihood, mean_function=None, inference_method=None, name='gp', Y_metadata=None, normalizer=False):

    # def __init__(self, X, Y, Z, kernel, likelihood, mean_function=None, X_variance=None, inference_method=None,
    #              name='sparse gp', Y_metadata=None, normalizer=False):
