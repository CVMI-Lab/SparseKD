import torch
import matplotlib.pyplot as plt
from . import kd_utils


def vis_channel_sptial_magnitude(feature, k=16):
    ch_mag = kd_utils.cal_channel_attention_mask(feature)
    spatial_mag = kd_utils.cal_spatial_attention_mask(feature)
    ch_one_hot_mask = ch_mag > 0
    
    # magnitude range
    # print('student ch mag range: [%f, %f]' % (ch_mag.min(), ch_mag.max()))
    # print('student spatial mag range: [%f, %f]' % (spatial_mag.min(), spatial_mag.max()))

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(ch_mag.view(k, -1).cpu().numpy())
    ax[0].set_title('channel mask')
    ax[1].imshow(spatial_mag[0].cpu().numpy())
    ax[1].set_title('spatial mask')
    ax[2].imshow(ch_one_hot_mask.view(k, -1).detach().cpu().numpy(), vmin=0, vmax=1)
    ax[2].set_title('channel one-hot mask')

    plt.show()

    return


def cal_feature_dist(feature, feature_tea, mode='cosine', topk=None):
    """_summary_

    Args:
        feature (_type_): _description_
        feature_tea (_type_): _description_
        mode (str, optional): [cosine, kl]. Defaults to 'cosine'.
    """
    bs = feature.shape[0]

    if topk is not None:
        feature = select_topk_feature_channel(feature, k=topk)
        feature_tea = select_topk_feature_channel(feature_tea, k=topk)

    if mode == 'cosine':
        dist = torch.nn.functional.cosine_similarity(feature.view(bs, -1), feature_tea.view(bs, -1))
    elif mode == 'kl':
        dist = torch.nn.functional.kl_div(feature, feature_tea)
    else:
        raise NotImplementedError

    return dist


def select_topk_feature_channel(feature, k):
    ch_mag = kd_utils.cal_channel_attention_mask(feature)
    _, channel_idx = torch.topk(ch_mag, k=k)

    return feature[:, channel_idx, ...]
 
