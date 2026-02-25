import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from models.spatial_consistency import Local_rigid, Outlier_rigid

def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances

@torch.no_grad()
def get_point_correspondences(
    src_points: torch.Tensor,
    ref_points: torch.Tensor,
    rot: torch.Tensor,
    trans: torch.Tensor,
    pos_radius: float
):
    r"""Generate ground-truth point correspondences.

    Each patch is composed of at most k nearest points of the corresponding superpoint.
    A pair of points match if the distance between them is smaller than `self.pos_radius`.

    Args:
        ref_points: torch.Tensor (M, 3)
        src_points: torch.Tensor (N, 3)
        rot: torch.Tensor (3, 3)
        trans: torch.Tensor (3, 1)
        pos_radius: float
    Returns:
        corr_indices: torch.LongTensor (C, 2)
        src_points_warp: torch.Tensor (N, 3)
    """

    src_points_warp = torch.matmul(src_points, rot.T) + trans.T

    dist_mat = torch.sqrt(pairwise_distance(src_points_warp, ref_points))
    point_overlap_mat = torch.lt(dist_mat, pos_radius)

    src_corr_indices, ref_corr_indices = torch.nonzero(point_overlap_mat, as_tuple=True)  # (C,) (C,) (C,)

    corr_indices = torch.stack([src_corr_indices, ref_corr_indices], dim=1)


    return corr_indices, src_points_warp

def compute_match_recall(conf_matrix_gt, match_pred):  # , s_pcd, t_pcd, search_radius=0.3):
    '''
    @param conf_matrix_gt:
    @param match_pred:
    @return:
    '''

    pred_matrix = torch.zeros_like(conf_matrix_gt)

    b_ind, src_ind, tgt_ind = match_pred[:, 0], match_pred[:, 1], match_pred[:, 2]
    pred_matrix[b_ind, src_ind, tgt_ind] = 1.

    true_positive = (pred_matrix == conf_matrix_gt) * conf_matrix_gt

    recall = true_positive.sum() / conf_matrix_gt.sum()

    precision = true_positive.sum() / max(len(match_pred), 1)

    return recall, precision

def blend_scene_flow(query_loc, reference_loc, reference_flow, knn=3):
    '''approximate flow on query points
    this function assume query points are sub-/un-sampled from reference locations
    @param query_loc:[m,3]
    @param reference_loc:[n,3]
    @param reference_flow:[n,3]
    @param knn:
    @return:
        blended_flow:[m,3]
    '''
    from lib.util import knn_point_np
    dists, idx = knn_point_np(knn, reference_loc, query_loc)
    # dists[dists < 1e-10] = 1e-10
    # weight = 1.0/ dists
    # weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
    # # test = reference_flow[idx].numpy() * weight.reshape([-1, knn, 1])
    # # np.sum(test, axis=1, keepdims=False)
    blended_flow = np.mean(reference_flow[idx], axis=1, keepdims=False)

    return blended_flow

def get_weight_mat(data, alpha =1):
    pcd = data['points'][0]
    len_src = data['stack_lengths'][0][0]
    len_tgt = data['stack_lengths'][0][1]
    src_pcd = pcd[:len_src, :].cpu()
    tgt_pcd = pcd[len_src:, :].cpu()
    match = data['match_pred'][:, 1:].cpu()
    flow = src_pcd[match[:, 0]] - tgt_pcd[match[:, 1]]
    flow = flow.numpy()
    ave_flow = blend_scene_flow(src_pcd[match[:, 0]].numpy(), src_pcd[match[:, 0]].numpy(), flow, knn=5)

    diff = np.linalg.norm(flow - ave_flow, axis=1)
    ratio = diff / np.linalg.norm(ave_flow, axis=1)

    weight_mat = np.zeros([len_src, len_tgt])

    weight_mat[match[:, 0], match[:, 1]] = 1 + ratio* alpha

    weight_mat = torch.from_numpy(weight_mat).to(data['match_pred'].device).unsqueeze(0)

    return weight_mat, ratio, match

class LiverLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.mat_w = config.match_weight

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        loss = pos_w * loss_pos.mean()
        return loss



    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def forward(self, data):
        loss_info = {}

        # visible loss
        src_pcd, tgt_pcd = data['src_pcd_raw'], data['tgt_pcd_raw']
        scores_vis = data['scores_vis']
        correspondence = data['correspondences']

        # only src scores
        src_idx = list(set(correspondence[:, 0].int().tolist()))
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx] = 1.
        src_gt_labels = src_gt.to(torch.device('cuda'))
        vis_loss, vis_cls_precision, vis_cls_recall = self.get_weighted_bce_loss(scores_vis, src_gt_labels)
        loss_info.update({'vis_loss': vis_loss, 'vis_recall': vis_cls_recall, 'vis_precision': vis_cls_precision})

        # mat loss
        conf_matrix_pred = data['conf_matrix_pred']

        match_gt = [correspondence.transpose(0, 1)]

        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])

        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        loss = self.vis_w * vis_loss + self.mat_w * mat_loss

        print("mat loss: ", mat_loss.item(),"\n")

        loss_info.update({'loss': loss})

        return loss_info


class LiverLoss_cluster(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.mat_w = config.match_weight
        self.cl_w = config.cl_weight

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        loss = pos_w * loss_pos.mean()
        return loss



    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt



    def forward(self, data):
        loss_info = {}

        # visible loss
        src_pcd, tgt_pcd = data['src_pcd_raw'], data['tgt_pcd_raw']
        scores_vis = data['scores_vis']
        correspondence = data['correspondences']

        # only src scores
        src_idx = list(set(correspondence[:, 0].int().tolist()))
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx] = 1.
        src_gt_labels = src_gt.to(torch.device('cuda'))
        vis_loss, vis_cls_precision, vis_cls_recall = self.get_weighted_bce_loss(scores_vis, src_gt_labels)
        loss_info.update({'vis_loss': vis_loss, 'vis_recall': vis_cls_recall, 'vis_precision': vis_cls_precision})

        # mat loss
        conf_matrix_pred = data['conf_matrix_pred']

        match_gt = [correspondence.transpose(0, 1)]

        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])

        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        # cluster mat loss
        conf_matrix_pred_cl = data['conf_matrix_pred_cl']
        src_xyz_cl = data['src_xyz_cl']
        rot = data['rot']
        trans = data['trans']

        node_size, cluster_size, _ = src_xyz_cl.size()
        #correspondence_cl = data['corr_cl_gt']  # n, c, 2

        corr_cl_gt = []
        corr_cl_gt_cl_index = []
        for i in np.arange(node_size):

            corr_gt, src_cl_warp = get_point_correspondences(src_xyz_cl[i], tgt_pcd, rot, trans, 0.05)

            # from lib.visualization import viz_coarse_nn_correspondence_mayavi
            # viz_coarse_nn_correspondence_mayavi(src_cl_warp.cpu(), tgt_pcd.cpu(), corr_gt.T)


            if len(corr_gt) / cluster_size > 0.0:
                corr_cl_gt.append(corr_gt.transpose(0, 1))
                corr_cl_gt_cl_index.append(i)

        conf_matrix_gt_cl = self.match_2_conf_matrix(corr_cl_gt, conf_matrix_pred_cl[corr_cl_gt_cl_index])
        data['conf_matrix_gt_cl'] = conf_matrix_gt_cl

        mat_loss_cl = self.compute_correspondence_loss(conf_matrix_pred_cl[corr_cl_gt_cl_index], conf_matrix_gt_cl, weight=None)

        mat_recall_cl, mat_precision_cl = compute_match_recall(conf_matrix_gt_cl, data['match_pred_cl'][corr_cl_gt_cl_index])

        loss_info.update({'mat_loss_cl': mat_loss_cl, 'mat_recall_cl': mat_recall_cl, 'mat_precision_cl': mat_precision_cl})

        loss = self.vis_w * vis_loss + self.mat_w * mat_loss + self.cl_w *mat_loss_cl

        loss_info.update({'loss': loss})

        return loss_info

class LiverLoss_vote(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.mat_w = config.matrix_weight

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        loss = pos_w * loss_pos.mean()
        return loss



    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def forward(self, data):
        loss_info = {}

        # visible loss
        src_pcd, tgt_pcd = data['src_pcd_raw'], data['tgt_pcd_raw']
        scores_vis = data['scores_vis']
        correspondence = data['correspondences']

        # only src scores
        src_idx = list(set(correspondence[:, 0].int().tolist()))
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx] = 1.
        src_gt_labels = src_gt.to(torch.device('cuda'))
        vis_loss, vis_cls_precision, vis_cls_recall = self.get_weighted_bce_loss(scores_vis, src_gt_labels)
        loss_info.update({'vis_loss': vis_loss, 'vis_recall': vis_cls_recall, 'vis_precision': vis_cls_precision})

        # vote
        vote_xyz = data['vote_xyz']
        vote_center = torch.mean(src_pcd[src_idx], dim=0)
        vote_err = src_pcd + vote_xyz.squeeze(0) - vote_center

        vote_err = torch.mean(torch.sum(torch.abs(vote_err), dim=1), dim=0)

        loss_info.update({'vote_err': vote_err})

        # mat loss
        conf_matrix_pred = data['conf_matrix_pred']

        match_gt = [correspondence.transpose(0, 1)]

        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])

        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        loss = self.vis_w * vis_loss + self.mat_w * mat_loss + vote_err

        #print("mat loss: ", mat_loss.item(),"\n")

        loss_info.update({'loss': loss})

        return loss_info


class LiverLoss_neg(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.mat_w = config.matrix_weight

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None, dual = True):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        if dual:
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
            loss = pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
        else:
            pos_conf = conf[pos_mask]
            loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
            loss = pos_w * loss_pos.mean()
        return loss



    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def forward(self, data):
        loss_info = {}

        # visible loss
        src_pcd, tgt_pcd = data['src_pcd_raw'], data['tgt_pcd_raw']
        scores_vis = data['scores_vis']
        correspondence = data['correspondences']

        # only src scores
        src_idx = list(set(correspondence[:, 0].int().tolist()))
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx] = 1.
        src_gt_labels = src_gt.to(torch.device('cuda'))
        vis_loss, vis_cls_precision, vis_cls_recall = self.get_weighted_bce_loss(scores_vis, src_gt_labels)
        loss_info.update({'vis_loss': vis_loss, 'vis_recall': vis_cls_recall, 'vis_precision': vis_cls_precision})

        # mat loss
        conf_matrix_pred = data['conf_matrix_pred']

        match_gt = [correspondence.transpose(0, 1)]

        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])

        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        loss = self.vis_w * vis_loss + self.mat_w * mat_loss

        print("mat loss: ", mat_loss.item(),"\n")

        loss_info.update({'loss': loss})

        return loss_info

class LiverLoss_weight(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.mat_w = config.matrix_weight

        self.alpha = config.alpha
        self.dual = config.dual

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None, dual = True):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = conf* weight
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        if dual:
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
            # if weight is not None:
            #     loss_pos = loss_pos * weight[pos_mask]
            #     loss_neg = loss_neg * weight[neg_mask]
            loss = pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
        else:
            pos_conf = conf[pos_mask]
            loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
            # if weight is not None:
            #     loss_pos = loss_pos * weight[pos_mask]
            loss = pos_w * loss_pos.mean()
        return loss



    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def forward(self, data):
        loss_info = {}

        # visible loss
        src_pcd, tgt_pcd = data['src_pcd_raw'], data['tgt_pcd_raw']
        scores_vis = data['scores_vis']
        correspondence = data['correspondences']

        # only src scores
        src_idx = list(set(correspondence[:, 0].int().tolist()))
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx] = 1.
        src_gt_labels = src_gt.to(torch.device('cuda'))
        vis_loss, vis_cls_precision, vis_cls_recall = self.get_weighted_bce_loss(scores_vis, src_gt_labels)
        loss_info.update({'vis_loss': vis_loss, 'vis_recall': vis_cls_recall, 'vis_precision': vis_cls_precision})

        # mat loss
        conf_matrix_pred = data['conf_matrix_pred']

        match_gt = [correspondence.transpose(0, 1)]

        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt

        weight, _, _ = get_weight_mat(data, alpha=self.alpha)
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=weight, dual= self.dual)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])

        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        loss = self.vis_w * vis_loss + self.mat_w * mat_loss

        print("mat loss: ", mat_loss.item(),"\n")

        loss_info.update({'loss': loss})

        return loss_info

class LiverLoss_prune(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.mat_w = config.matrix_weight

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        loss = pos_w * loss_pos.mean()
        return loss



    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def forward(self, data):
        loss_info = {}

        # outlier classification loss
        correspondence = data['correspondences']
        pred_match = data['match_pred'][:, 1:]
        conf_score = data['conf_score'][0, :, 0]
        conf_score_gt = torch.zeros(conf_score.size(0))

        for i in np.arange(len(pred_match)):

            #torch.any(torch.all(correspondence == pred_match[i, :].unsqueeze(0), dim=1)) * 1
            # print(correspondence.size())
            # print(pred_match[i, :].unsqueeze(0).size())
            # print(torch.any(torch.all(correspondence == pred_match[i, :].unsqueeze(0),dim=1)))
            conf_score_gt[i] = torch.any(torch.all(correspondence == pred_match[i, :].unsqueeze(0),dim=1))*1

        conf_score_gt = conf_score_gt.to(torch.device('cuda'))

        con_loss, con_cls_precision, con_cls_recall = self.get_weighted_bce_loss(conf_score, conf_score_gt)
        loss_info.update({'vis_loss': con_loss, 'vis_recall': con_cls_recall, 'vis_precision': con_cls_precision})


        # mat loss
        conf_matrix_pred = data['conf_matrix_pred']

        match_gt = [correspondence.transpose(0, 1)]

        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])

        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        loss = self.vis_w * con_loss + self.mat_w * mat_loss

        print("mat loss: ", mat_loss.item(),"\n")

        loss_info.update({'loss': loss})

        return loss_info

class LiverLoss_multi(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.mat_w = config.matrix_weight

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        loss = pos_w * loss_pos.mean()
        return loss

    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def forward(self, data):
        loss_info = {}

        # fine level
        correspondence = data['correspondences'][0]
        conf_matrix_pred = data['conf_matrix_pred'][0]
        match_gt = [correspondence.transpose(0, 1)]
        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])
        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        # coarse level
        correspondence_c = data['correspondences'][1]
        conf_matrix_pred_c = data['conf_matrix_pred'][1]
        match_gt_c = [correspondence_c.transpose(0, 1)]
        conf_matrix_gt_c = self.match_2_conf_matrix(match_gt_c, conf_matrix_pred_c)
        data['conf_matrix_gt_c'] = conf_matrix_gt_c
        mat_loss_c = self.compute_correspondence_loss(conf_matrix_pred_c, conf_matrix_gt_c, weight=None)
        print(mat_loss_c)
        mat_recall_c, mat_precision_c = compute_match_recall(conf_matrix_gt_c, data['match_pred_c'])
        loss_info.update({'mat_loss_c': mat_loss_c, 'mat_recall_c': mat_recall_c, 'mat_precision_c': mat_precision_c})

        loss = self.mat_w * (mat_loss + mat_loss_c)

#        print("loss: ", loss.item(), "\n")

        loss_info.update({'loss': loss})

        return loss_info

class LiverLoss_flow(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.flow_w = config.flow_weight
        self.mat_w = config.matrix_weight
        self.use_mask = config.use_mask

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        loss = pos_w * loss_pos.mean()
        return loss



    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def compute_flow_loss(self, pred_flow, gt_flow, mask, use_mask =True):

        error = pred_flow - gt_flow
        if use_mask:
            error = error[mask>0]
        loss = torch.mean(torch.abs(error))

        # print(loss)

        return loss

    def forward(self, data):
        loss_info = {}

        src_pcd, tgt_pcd = data['src_pcd_raw'], data['tgt_pcd_raw']
        correspondence = data['correspondences']
        match_pred = data['match_pred'][:, 1:]

        # mat loss
        conf_matrix_pred = data['conf_matrix_pred']

        match_gt = [correspondence.transpose(0, 1)]

        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])

        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        # flow loss

        tgt_idx_pred = match_pred[:, 1]

        flow = data['flow'].squeeze(0)
        flow_gt = data['inverse_flow']

        tgt_mask = torch.zeros(tgt_pcd.size(0)).to(torch.device(flow.device))
        tgt_mask[tgt_idx_pred] = 1.

        flow_loss = self.compute_flow_loss(flow, flow_gt, tgt_mask, use_mask=self.use_mask)
        loss_info.update({'flow_loss': flow_loss})

        loss = self.mat_w * mat_loss + self.flow_w * flow_loss

        print("flow loss: ", flow_loss.item(), "\n")
        print("mat loss: ", mat_loss.item(), "\n")
        print("loss: ", loss.item(), "\n")

        loss_info.update({'loss': loss})

        return loss_info

class LiverLoss_reg(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.mat_w = config.matrix_weight
        self.warp_w = config.warp_weight

        # Local rigid
        self.rigid = Local_rigid(config)

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        loss = pos_w * loss_pos.mean()
        return loss

    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def forward(self, data):
        loss_info = {}

        # visible loss
        src_pcd, tgt_pcd = data['src_pcd_raw'], data['tgt_pcd_raw']
        scores_vis = data['scores_vis']
        correspondence = data['correspondences']

        # only src scores
        src_idx = list(set(correspondence[:, 0].int().tolist()))
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx] = 1.
        src_gt_labels = src_gt.to(src_pcd.device)
        vis_loss, vis_cls_precision, vis_cls_recall = self.get_weighted_bce_loss(scores_vis, src_gt_labels)
        loss_info.update({'vis_loss': vis_loss, 'vis_recall': vis_cls_recall, 'vis_precision': vis_cls_precision})

        # mat loss
        conf_matrix_pred = data['conf_matrix_pred']

        match_gt = [correspondence.transpose(0, 1)]

        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])

        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        # total loss
        loss = self.vis_w * vis_loss + self.mat_w * mat_loss

        # warping loss
        if mat_recall > 0.2 and self.warp_w > 0:
            # R_s2t_pred = data["R_s2t_pred"]
            # t_s2t_pred = data["t_s2t_pred"]
            # sflow_gt = data["sflow_gt"].unsqueeze(0)
            # src_pcd = data['src_pcd_raw'].unsqueeze(0)
            # mask = (src_gt_labels>0).unsqueeze(0)
            #
            # #compute predicted flow. Note, if 4dmatch, the R_pred,t_pred try to find the best rigid fit of deformation
            # src_pcd_wrapped_pred = (torch.matmul(R_s2t_pred, src_pcd.transpose(1, 2)) + t_s2t_pred).transpose(1, 2)
            # sflow_pred = src_pcd_wrapped_pred - src_pcd
            # e1 = torch.sum(torch.abs(sflow_pred - sflow_gt), 2)
            # e1 = e1[mask]  # [data['src_mask']]
            # l1_loss = torch.mean(e1)
            # l1_loss = data['l1_loss']
            l1_loss = self.rigid(data)
            #print(l1_loss.grad_fn)
            if l1_loss is not None:
                loss = loss + self.warp_w * l1_loss
                loss_info.update({'warp_loss': l1_loss})
                print("warp loss: ", l1_loss.item(), "\n")
            else:
                loss_info.update({'warp_loss': None})
        else:
            loss_info.update({'warp_loss': None})

        print("mat loss: ", mat_loss.item(),"\n")


        loss_info.update({'loss': loss})

        return loss_info

class LiverLoss_outlier(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.mat_w = config.matrix_weight
        self.warp_w = config.warp_weight

        # Local rigid
        self.rigid = Outlier_rigid(config)

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        loss = pos_w * loss_pos.mean()
        return loss

    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def forward(self, data):
        loss_info = {}

        # visible loss
        src_pcd, tgt_pcd = data['src_pcd_raw'], data['tgt_pcd_raw']
        scores_vis = data['scores_vis']
        correspondence = data['correspondences']

        # only src scores
        src_idx = list(set(correspondence[:, 0].int().tolist()))
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx] = 1.
        src_gt_labels = src_gt.to(src_pcd.device)
        vis_loss, vis_cls_precision, vis_cls_recall = self.get_weighted_bce_loss(scores_vis, src_gt_labels)
        loss_info.update({'vis_loss': vis_loss, 'vis_recall': vis_cls_recall, 'vis_precision': vis_cls_precision})

        # mat loss
        conf_matrix_pred = data['conf_matrix_pred']

        match_gt = [correspondence.transpose(0, 1)]

        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])

        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        # total loss
        loss = self.vis_w * vis_loss + self.mat_w * mat_loss

        # warping loss
        if mat_recall > 0.2 and self.warp_w > 0:
            # R_s2t_pred = data["R_s2t_pred"]
            # t_s2t_pred = data["t_s2t_pred"]
            # sflow_gt = data["sflow_gt"].unsqueeze(0)
            # src_pcd = data['src_pcd_raw'].unsqueeze(0)
            # mask = (src_gt_labels>0).unsqueeze(0)
            #
            # #compute predicted flow. Note, if 4dmatch, the R_pred,t_pred try to find the best rigid fit of deformation
            # src_pcd_wrapped_pred = (torch.matmul(R_s2t_pred, src_pcd.transpose(1, 2)) + t_s2t_pred).transpose(1, 2)
            # sflow_pred = src_pcd_wrapped_pred - src_pcd
            # e1 = torch.sum(torch.abs(sflow_pred - sflow_gt), 2)
            # e1 = e1[mask]  # [data['src_mask']]
            # l1_loss = torch.mean(e1)
            # l1_loss = data['l1_loss']
            l1_loss = self.rigid(data)
            #print(l1_loss.grad_fn)
            if l1_loss is not None:
                loss = loss + self.warp_w * l1_loss
                loss_info.update({'warp_loss': l1_loss})
                print("warp loss: ", l1_loss.item(), "\n")
            else:
                loss_info.update({'warp_loss': None})
        else:
            loss_info.update({'warp_loss': None})

        print("mat loss: ", mat_loss.item(),"\n")


        loss_info.update({'loss': loss})

        return loss_info



class LiverLoss_SPC(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.mat_w = config.matrix_weight

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        loss = pos_w * loss_pos.mean()
        return loss

    def match_2_conf_matrix_cluster(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        matrix_gt[matches_gt[:, 0], matches_gt[:, 1], matches_gt[:,2]] = 1

        return matrix_gt

    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def forward(self, data):
        loss_info = {}

        # fine level

        match_gt = data['fine_match_pred']
        conf_matrix_pred = data['fine_conf_matrix_pred']
        conf_matrix_gt = self.match_2_conf_matrix_cluster(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['fine_match_pred'])
        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        # coarse level

        conf_matrix_pred_c = data['coarse_conf_matrix_pred']
        match_gt_c = data['coarse_match_gt']
        conf_matrix_gt_c = self.match_2_conf_matrix_cluster(match_gt_c, conf_matrix_pred_c)
        data['conf_matrix_gt_c'] = conf_matrix_gt_c
        mat_loss_c = self.compute_correspondence_loss(conf_matrix_pred_c, conf_matrix_gt_c, weight=None)
        #print(mat_loss_c)
        mat_recall_c, mat_precision_c = compute_match_recall(conf_matrix_gt_c, data['coarse_match_pred'])
        loss_info.update({'mat_loss_c': mat_loss_c, 'mat_recall_c': mat_recall_c, 'mat_precision_c': mat_precision_c})

        loss = 0* mat_loss + mat_loss_c

#        print("loss: ", loss.item(), "\n")

        loss_info.update({'loss': loss})

        return loss_info





