import time
from pcdet.utils import common_utils
from .detector3d_template import Detector3DTemplate
from pcdet.models.kd_heads.center_head.center_kd_head import CenterHeadKD


class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # the kd loss config will be propagated to dense head
        if self.model_cfg.get('KD_LOSS', None) and self.model_cfg.KD_LOSS.ENABLED:
            self.model_cfg.DENSE_HEAD.KD_LOSS = self.model_cfg.KD_LOSS

        self.module_list = self.build_networks()

        if self.dense_head is None and self.dense_head_aux is not None:
            self.dense_head = self.dense_head_aux

        self.kd_head = CenterHeadKD(self.model_cfg, self.dense_head) if model_cfg.get('KD', None) else None

        self.dense_head.kd_head = self.kd_head
        # for measure time only
        self.module_time_meter = common_utils.DictAverageMeter()

    def forward(self, batch_dict, record_time=False, det_only=False, kd_only=False, sep_idx=[-1, 10000]):
        """
        Args:
            batch_dict:
            record_time:

        Returns:

        """
        # start_idx end_idx: convenient to forward only part of module in the detector
        start_idx, end_idx = sep_idx

        for idx, cur_module in enumerate(self.module_list):
            if record_time:
                end = time.time()

            if idx > end_idx:
                break

            if idx > start_idx:
                batch_dict = cur_module(batch_dict)

            if record_time:
                module_name = str(type(cur_module)).split('.')[-1][:-2]
                self.module_time_meter.update(module_name, (time.time() - end) * 1000)

        if end_idx < len(self.module_list):
            if self.training and kd_only:
                return self.get_kd_loss(batch_dict, {}, {})
            else:
                return batch_dict

        if self.is_teacher:
            return batch_dict

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            # only student model has KD_LOSS cfg
            if self.model_cfg.get('KD_LOSS', None) and self.model_cfg.KD_LOSS.ENABLED and not det_only:
                kd_loss, tb_dict, disp_dict = self.get_kd_loss(batch_dict, tb_dict, disp_dict)
                loss += kd_loss

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if record_time:
                end = time.time()
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if record_time:
                self.module_time_meter.update('post_processing', (time.time() - end) * 1000)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn = 0
        tb_dict = {}
        if self.dense_head is not None and not self.dense_head.disable:
            loss_rpn, tb_dict = self.dense_head.get_loss()

        if self.dense_head_aux is not None and self.dense_head_aux is not self.dense_head and \
                not self.dense_head_aux.disable:
            loss_rpn_aux, tb_dict_aux = self.dense_head_aux.get_loss()
            loss_rpn += loss_rpn_aux
            tb_dict.update(tb_dict_aux)

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
