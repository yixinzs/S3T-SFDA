from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks.hook import HOOKS, Hook
from .logger import log_every_n
from torch.nn import Module
from copy import deepcopy
import torch
from mmcv.runner import save_checkpoint
import os.path as osp
import time
from mmcv.parallel import collate, scatter

# @HOOKS.register_module()
# class SWAHook(Hook):
#     def __init__(self,
#                  swa_start=None,
#                  swa_freq=None,
#                  swa_lr=None,
#                  by_epoch=False):
#         pass
#         self._auto_mode, (self.swa_start, self.swa_freq) = \
#             self._check_params(self, swa_start, swa_freq)
#         self.swa_lr = swa_lr
#
#         self.swa_model = None
#         self.by_epoch = by_epoch
#
#         if self._auto_mode:
#             if swa_start < 0:
#                 raise ValueError("Invalid swa_start: {}".format(swa_start))
#             if swa_freq < 1:
#                 raise ValueError("Invalid swa_freq: {}".format(swa_freq))
#         else:
#             if self.swa_lr is not None:
#                 log_every_n(
#                     "Some of swa_start, swa_freq is None, ignoring swa_lr")
#             # If not in auto mode make all swa parameters None
#             self.swa_lr = None
#             self.swa_start = None
#             self.swa_freq = None
#
#         if self.swa_lr is not None and self.swa_lr < 0:
#             raise ValueError("Invalid SWA learning rate: {}".format(swa_lr))
#
#     def before_run(self, runner):
#
#         model = runner.model
#         if is_module_wrapper(model):
#             model = model.module
#         self.swa_model = AveragedModel(model)
#
#     def after_train_iter(self, runner):
#         if runner.iter >= self.swa_start and (runner.iter - self.swa_start) % self.swa_freq == 0:
#             model = runner.model
#             if is_module_wrapper(model):
#                 model = model.module
#             self.swa_model.update_parameters(model)
#             self.swa_save(runner)
#         if runner.iter + 1 == runner._max_iters:
#             update_bn(runner, self.swa_model)
#             self.swa_save(runner, True)
#
#
#     def swa_save(self, runner, update_bn=False):
#         if not update_bn:
#             path = osp.join(runner.work_dir, 'swa_epoch_' + str(runner.iter) + '.pth')
#             save_checkpoint(self.swa_model, path)
#         else:
#             path = osp.join(runner.work_dir, 'swa_final.pth')
#             save_checkpoint(self.swa_model, path)
#
#     @staticmethod
#     def _check_params(self, swa_start, swa_freq):
#         params = [swa_start, swa_freq]
#         params_none = [param is None for param in params]
#         if not all(params_none) and any(params_none):
#             log_every_n(
#                 "Some of swa_start, swa_freq is None, ignoring other")
#         for i, param in enumerate(params):
#             if param is not None and not isinstance(param, int):
#                 params[i] = int(param)
#                 log_every_n("Casting swa_start, swa_freq to int")
#         return not any(params_none), params


@HOOKS.register_module()
class SWAHook(Hook):
    def __init__(self,
                 swa_start=None,
                 swa_start_ratio=0.75,
                 swa_freq=None,
                 swa_restart_step=5,
                 swa_lr=None,
                 by_epoch=False):

        self.swa_lr = swa_lr

        self.swa_model = None
        self.by_epoch = by_epoch

        if self.swa_lr is not None:
            log_every_n(
                "Some of swa_start, swa_freq is None, ignoring swa_lr")
        # If not in auto mode make all swa parameters None
        assert (swa_start is not None) or (swa_start_ratio is not None), 'need to set param [swa_start] or [swa_start_ratio]'

        self.swa_start = swa_start
        self.swa_start_ratio = swa_start_ratio
        self.swa_freq = swa_freq
        self.swa_restart_step = swa_restart_step
        if self.swa_lr is not None and self.swa_lr < 0:
            raise ValueError("Invalid SWA learning rate: {}".format(swa_lr))

    def before_run(self, runner):

        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.swa_model = AveragedModel(model)

    def after_train_iter(self, runner):
        if self.swa_start is not None and self.swa_freq is not None:
            normal_max_iters = self.swa_start
            swa_step_max_iters = (runner._max_iters - normal_max_iters) // self.swa_freq
        else:
            normal_max_iters = int(runner._max_iters * self.swa_start_ratio)
            swa_step_max_iters = (runner._max_iters - normal_max_iters) // self.swa_restart_step
        # print('-------------------normal_max_iters:{}, swa_step_max_iters:{}'.format(normal_max_iters, swa_step_max_iters))
        if runner.iter > normal_max_iters and (runner.iter - normal_max_iters) % swa_step_max_iters == 0:
            model = runner.model
            if is_module_wrapper(model):
                model = model.module
            self.swa_model.update_parameters(model)
            # self.swa_save(runner)
        if runner.iter + 1 == runner._max_iters:
            update_bn(runner, self.swa_model)
            self.swa_save(runner, True)


    def swa_save(self, runner, update_bn=False):
        if not update_bn:
            path = osp.join(runner.work_dir, 'swa_epoch_' + str(runner.iter) + '.pth')
            save_checkpoint(self.swa_model, path)
        else:
            path = osp.join(runner.work_dir, 'swa_final.pth')
            save_checkpoint(self.swa_model, path)

    @staticmethod
    def _check_params(self, swa_start, swa_freq):
        params = [swa_start, swa_freq]
        params_none = [param is None for param in params]
        if not all(params_none) and any(params_none):
            log_every_n(
                "Some of swa_start, swa_freq is None, ignoring other")
        for i, param in enumerate(params):
            if param is not None and not isinstance(param, int):
                params[i] = int(param)
                log_every_n("Casting swa_start, swa_freq to int")
        return not any(params_none), params


class AveragedModel(Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA).
    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).
    AveragedModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.
    Args:
        model (torch.nn.Module): model to use with SWA
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device`
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            :class:`AveragedModel` parameter, the current value of :attr:`model`
            parameter and the number of models already averaged; if None,
            equally weighted average is used (default: None)
    Example:
        >>> loader, optimizer, model, loss_fn = ...
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        >>>                                     T_max=300)
        >>> swa_start = 160
        >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_model.update_parameters(model)
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()
        >>>
        >>> # Update bn statistics for the swa_model at the end
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)
    You can also use custom averaging functions with `avg_fn` parameter.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights.
    Example:
        >>> # Compute exponential moving averages of the weights
        >>> ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                            0.1 * averaged_model_parameter + 0.9 * model_parameter
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
    .. note::
        When using SWA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        You can do so by using :meth:`torch.optim.swa_utils.update_bn` utility.
    .. note::
        :attr:`avg_fn` is not saved in the :meth:`state_dict` of the model.
    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.
    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    """
    def __init__(self, model, device=None, avg_fn=None):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                    (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn

    # def forward(self, *args, **kwargs):
    #     img = kwargs['img']
    #     # print('------------------------:{}'.format(img.shape))
    #     img_metas = kwargs['img_metas']
    #     return self.module.encode_decode(img, img_metas)
    #     # return self.module.encode_decode(*args, **kwargs)

    def forward(self, img, img_metas,  **kwargs):
        device = kwargs['device']
        # print('-----------------------------device:{}'.format(device))
        img = img.data[0].to(device)  #
        # print('------------------------:{}'.format(img.shape))
        img_metas = img_metas.data[0]  #
        return self.module.encode_decode(img, img_metas)
        # return self.module.encode_decode(*args, **kwargs)


    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1


@torch.no_grad()
def update_bn(runner, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None           #BatchNorm的更新：https://blog.csdn.net/qq_39208832/article/details/117930625
        module.num_batches_tracked *= 0

    data_loader = runner.data_loader
    for index in range(len(data_loader)):
        data_batch = next(data_loader)
        device = next(model.parameters()).device
        # print('-----------------------------device:{}'.format(device))
        # if next(model.parameters()).is_cuda:
        #     # scatter to specified GPU
        #     data_batch = scatter(data_batch, [device])[0]
        # else:
        #     data_batch['img_metas'] = [i.data_batch[0] for i in data_batch['img_metas']]
        # print('--------------data_batch:{}'.format(data_batch.keys()))
        model(**data_batch, device=device)  #, runner.optimizer

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]  #bn_module.momentum不更新，更新的是module.running_mean和module.running_var
    model.train(was_training)


class IterLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)
