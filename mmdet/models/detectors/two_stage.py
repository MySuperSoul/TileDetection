import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from ..utils import Scale
import numpy as np


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 train_with_tmp=False,
                 scale_origin_ratio=1.5,
                 scale_template_ratio=1.0,
                 method=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.train_with_tmp = train_with_tmp
        self.method = method  # assert in ['sub_img', 'add_img', 'sub_feat', 'scale_sub_feat']
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        if self.train_with_tmp == True and (
                self.method == 'scale_sub_feat'
                or self.method == 'scale_feat_concat'
                or self.method == 'scale_feat_concat_nine'
                or self.method == 'scale_feat_concat_backbone'
                or self.method == 'scale_feat_backbone'):
            self.scale_origin = Scale(scale_ratio=scale_origin_ratio)
            self.scale_template = Scale(scale_ratio=scale_template_ratio)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        feats = [
            self.extract_feat_with_template(img) for img in imgs
        ] if self.train_with_tmp else [self.extract_feat(img) for img in imgs]
        return feats

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def extract_feat_with_template(self, img):
        if self.train_with_tmp == True and self.method is not None:
            N, C, H, W = img.shape
            img = img.reshape(2 * N, -1, H, W)
            if self.method == 'sub_img':
                img = img[0::2, :, :, :] - img[1::2, :, :, :]
                x = self.extract_feat(img)
            elif self.method == 'add_img':
                img = img[0::2, :, :, :] + img[1::2, :, :, :]
                x = self.extract_feat(img)
            elif self.method == 'sub_feat':
                x = self.extract_feat(img)
                sub_feats = []
                for feat in x:
                    sub_feats.append(feat[0::2, :, :, :] - feat[1::2, :, :, :])
                x = tuple(sub_feats)
            elif self.method == 'scale_sub_feat':
                x = self.extract_feat(img)
                feats = []
                for feat in x:
                    feats.append(
                        self.scale_origin(feat[0::2, :, :, :]) -
                        self.scale_template(feat[1::2, :, :, :]))
                x = tuple(feats)
            elif self.method == 'concat_img_six':
                img = img.reshape(N, C, H, W)
                x = self.extract_feat(img)
            elif self.method == 'concat_img_nine':
                sub_img = img[0::2, :, :, :] - img[1::2, :, :, :]
                img = torch.cat([img[0::2, ...], sub_img, img[1::2, ...]],
                                dim=1)
                x = self.extract_feat(img)
            elif self.method == 'scale_feat_concat':
                x = self.extract_feat(img)
                feats = []
                for feat in x:
                    feats.append(
                        torch.cat([
                            feat[0::2, :, :, :],
                            self.scale_origin(feat[0::2, :, :, :]) -
                            self.scale_template(feat[1::2, :, :, :])
                        ],
                                  dim=1))
                x = tuple(feats)
            elif self.method == 'feat_concat':
                x = self.extract_feat(img)
                feats = []
                for feat in x:
                    feats.append(
                        torch.cat([
                            feat[0::2, :, :, :],
                            feat[0::2, :, :, :] - feat[1::2, :, :, :]
                        ],
                                  dim=1))
                x = tuple(feats)
            elif self.method == 'feat_concat_backbone':
                x = self.backbone(img)
                feats = []
                for feat in x:
                    feats.append(
                        torch.cat([
                            feat[0::2, :, :, :],
                            feat[0::2, :, :, :] - feat[1::2, :, :, :]
                        ],
                                  dim=1))
                feats = tuple(feats)
                x = self.neck(feats)
            elif self.method == 'scale_feat_concat_backbone':
                x = self.backbone(img)
                feats = []
                for feat in x:
                    feats.append(
                        torch.cat([
                            feat[0::2, :, :, :],
                            self.scale_origin(feat[0::2, :, :, :]) -
                            self.scale_template(feat[1::2, :, :, :])
                        ],
                                  dim=1))
                feats = tuple(feats)
                x = self.neck(feats)
            elif self.method == 'scale_feat_backbone':
                x = self.backbone(img)
                feats = []
                for feat in x:
                    feats.append(
                        self.scale_origin(feat[0::2, :, :, :]) -
                        self.scale_template(feat[1::2, :, :, :]))
                feats = tuple(feats)
                x = self.neck(feats)
            elif self.method == 'scale_feat_concat_nine':
                x = self.extract_feat(img)
                feats = []
                for feat in x:
                    feats.append(
                        torch.cat([
                            feat[0::2, :, :, :],
                            self.scale_origin(feat[0::2, :, :, :]) -
                            self.scale_template(feat[1::2, :, :, :]),
                            feat[1::2, :, :, :]
                        ],
                                  dim=1))
                x = tuple(feats)
        else:
            x = self.extract_feat(img)

        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.train_with_tmp:
            x = self.extract_feat_with_template(img)
        else:
            x = self.extract_feat(img)
        img = img.cpu()
        torch.cuda.empty_cache()

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat_with_template(
            img) if self.train_with_tmp else self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
