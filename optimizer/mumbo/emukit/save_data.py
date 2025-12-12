# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:25:23 2022

@author: fl347
"""
import os
import sys
import json
import time
import pickle
# import numpy as np
from loguru import logger



class Save:

    def __init__(self,output_path):
        if output_path is None:

            self.output_path = '/'
        else:
            self.output_path=output_path
        # self.output_path = kwargs['output_path'] if 'output_path' in kwargs else './'
        os.makedirs(self.output_path, exist_ok=True)
        self.logger = logger
        log_suffix = time.strftime("%x %X %Z")
        log_suffix = log_suffix.replace("/", '-').replace(":", '-').replace(" ", '_')

        self.start=time.time()

    def save_incumbent(self, res,name=None):
        if name is None:
            name = time.strftime("%x %X %Z", time.localtime(self.start))
            name = name.replace("/", '-').replace(":", '-').replace(" ", '_')

        # res["score"] = self.inc_score
        # res["info"] = self.inc_info
        with open(os.path.join(self.output_path, "incumbent_{}.json".format(name)), 'w') as f:
            json.dump(res, f)


    def save_history(self, history,name=None):
        if name is None:
            name = time.strftime("%x %X %Z", time.localtime(self.start))
            name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
        with open(os.path.join(self.output_path, "history_{}.pkl".format(name)), 'wb') as f:
            pickle.dump(history, f)
