from .detector3d_template import Detector3DTemplate
from pcdet.models.kd_heads.center_head.center_kd_head import CenterHeadKD


class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        if self.dense_head is None and self.dense_head_aux is not None:
            self.dense_head = self.dense_head_aux

        self.kd_head = CenterHeadKD(self.model_cfg, self.dense_head) if model_cfg.get('KD', None) else None
        self.dense_head.kd_head = self.kd_head
        if self.kd_head is not None:
            self.kd_head.roi_head = self.roi_head

    def forward(self, batch_dict, **kwargs):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.is_teacher and self.training:
            return batch_dict

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            # only student model has KD_LOSS cfg
            if self.model_cfg.get('KD_LOSS', None) and self.model_cfg.KD_LOSS.ENABLED:
                kd_loss, tb_dict, disp_dict = self.get_kd_loss(batch_dict, tb_dict, disp_dict)
                loss += kd_loss

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
