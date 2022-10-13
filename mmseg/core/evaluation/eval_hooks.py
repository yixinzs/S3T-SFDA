# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None

        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')

    # def _do_evaluate(self, runner):
    #     """perform evaluation and save ckpt."""
    #     if not self._should_evaluate(runner):
    #         return
    #
    #     from mmseg.apis import single_gpu_test
    #     results = single_gpu_test(
    #         runner.model, self.dataloader, show=False, pre_eval=self.pre_eval)
    #     self.latest_results = results
    #     runner.log_buffer.clear()
    #     runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
    #     key_score = self.evaluate(runner, results)
    #     if self.save_best:
    #         self._save_ckpt(runner, key_score)


    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg_custom.apis import single_gpu_test
        results = single_gpu_test(
            runner.model, self.dataloader, show=False, pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self.save_ckpt_custom(runner, key_score)

    def save_ckpt_custom(self, runner, key_score):
        """Save the best checkpoint.

        It will compare the score according to the compare function, write
        related information (best score, best checkpoint path) and save the
        best checkpoint into ``work_dir``.
        """
        import os

        import mmcv
        out_dir = osp.join(self.out_dir, 'checkpoint')
        mmcv.mkdir_or_exist(out_dir)
        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        best_score = runner.meta['hook_msgs'].get(
            'best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            runner.meta['hook_msgs']['best_score'] = best_score

            if self.best_ckpt_path and self.file_client.isfile(
                    self.best_ckpt_path):
                self.file_client.remove(self.best_ckpt_path)
                runner.logger.info(
                    (f'The previous best checkpoint {self.best_ckpt_path} was '
                     'removed'))

            best_ckpt_name = f'best_{best_score}_{current}.pth'  # self.key_indicator
            self.best_ckpt_path = self.file_client.join_path(
                out_dir, best_ckpt_name)
            runner.meta['hook_msgs']['best_ckpt'] = self.best_ckpt_path

            runner.save_checkpoint(
                out_dir, best_ckpt_name, create_symlink=False)
            runner.logger.info(
                f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(
                f'Best {self.key_indicator} is {best_score:0.4f} '
                f'at {cur_time} {cur_type}.')

        ckpt_name = f'{current}_{key_score}_checkpoint.pth'  # self.key_indicator
        self.ckpt_path = self.file_client.join_path(
            out_dir, ckpt_name)
        runner.save_checkpoint(
            out_dir, ckpt_name, create_symlink=False)

        current_model_save = os.listdir(out_dir)
        current_model_save.sort(key=lambda x: -1 if (x.split('_'))[0] == 'best' else float((x.split('_'))[2]))  # , reverse=True   #https://www.cnblogs.com/chester-cs/p/12252358.html
        if len(current_model_save) > 3:  #9
            for remove_model_file in current_model_save[1:-3]:  #9
                self.file_client.remove(os.path.join(out_dir, remove_model_file))
                runner.logger.info(
                    (f'The previous best checkpoint {remove_model_file} was '
                     'removed'))

class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``multi_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
