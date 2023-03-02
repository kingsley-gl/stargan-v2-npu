"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import torch
import torch_npu

class CheckpointIO(object):
    def __init__(self, fname_template, device, data_parallel=False, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs
        self.data_parallel = data_parallel
        self.device = device

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            # if self.data_parallel:
            #     outdict[name] = module.module.state_dict()
            # else:
            #     outdict[name] = module.state_dict()
            outdict[name] = module.state_dict()

        torch.save(outdict, fname)

    def load(self, step):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        # if self.device == 'npu':
        #     if torch_npu.npu.is_available():
        #         module_dict = torch.load(fname, map_location=torch.device('npu'))
        #     else:
        #         module_dict = torch.load(fname, map_location=torch.device('cpu'))
        # else:
        #     if torch.cuda.is_available():
        #         module_dict = torch.load(fname)
        #     else:
        #         module_dict = torch.load(fname, map_location=torch.device('cpu'))

        module_dict = torch.load(fname, map_location=self.device)

        for name, module in self.module_dict.items():
            if self.data_parallel:
                module.module.load_state_dict(module_dict[name], False)
            else:
                module.load_state_dict(module_dict[name])
