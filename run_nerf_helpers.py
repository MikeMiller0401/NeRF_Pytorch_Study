import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
# get_embedder和Embedder：位置编码
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []  # 创建一个空列表embed_fns来存储编码函数
        d = self.kwargs['input_dims']
        out_dim = 0  # 初始化输出维度，默认为0
        if self.kwargs['include_input']:
            # 若为include_input == True
            embed_fns.append(lambda x: x)  # 添加一个lambda进入embed_fns中
            out_dim += d  # d累加到out_dim

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            # 使用torch.linspace生成0-9的等比数列：a = tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
            # freq_bands == 2 ** a，生成1到2的9次方的新数列：tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        # y(x) = (sin(1 * x), cos(1 * x), sin(2 * x), cos(2 * x), ... ,sin(512 * x), cos(512 * x))
        for freq in freq_bands:  # freq遍历freq_bands
            for p_fn in self.kwargs['periodic_fns']:  # p_fn遍历self.kwargs，self.kwargs为[sin(), cos()]
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))  # 向embed_fns列表中添加一个函数p_fn(x * freq)
                out_dim += d  # out_dim累加

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        # 从self.embed_fns中接受函数作为fn，并将input输入给fn，最后拼接在一起
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    # 传入最大频率multires
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,  # 如果为真，最终的编码结果包含原始坐标
        'input_dims': 3,  # 输入给编码器的数据的维度，默认为3即点的三维坐标
        'max_freq_log2': multires - 1,  # 最大频率的对数值-1
        'num_freqs': multires,  # 最大频率的对数值，即论文中的公式（4）中的L-1；对于三维坐标来说，L=10。
        'log_sampling': True,  # 是否采用对数采样
        'periodic_fns': [torch.sin, torch.cos],  # 周期性函数列表
    }

    embedder_obj = Embedder(**embed_kwargs)  # 创建的Embedder对象被赋值给embedder_obj
    # 创建了一个匿名函数 embed，并将 embedder_obj 作为默认参数传递给该函数，以便在调用时使用特定的 Embedder 对象进行嵌入操作。
    # 这样做的好处是，每次调用 embed 函数时不需要显式传递 embedder_obj，而是使用默认参数的值。
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim  # 返回embed和输出的维度out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch  # Position Encoding之后的 位置vector通道数（63）
        self.input_ch_views = input_ch_views  # Position Encoding之后的 位姿的vector通道数（27）
        self.skips = skips  # 在第4层有跳跃连接
        self.use_viewdirs = use_viewdirs

        # 前8层的MLP实现：输入为63，输出为 256
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +  # 第一层输入为63，输出63
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)])  # 列表推导式：
        # 如果当前层的索引 i 不在 self.skips 中，则输入通道数为256，输出通道数为256；否则，输入通道数为256+63，输出通道数为256

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # 第九层的MLP：输入层为 位姿（27）+256，输出为128
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        # 若使用视角方向，需要定义几个额外的层
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)  # 特征向量层：输入256，输出256
            self.alpha_linear = nn.Linear(W, 1)   # 体密度：输入256，输出1
            self.rgb_linear = nn.Linear(W // 2, 3)  # RGB层，输入128，输出3
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        # 根据self.input_ch和self.input_ch_views将输入拆分为input_pts, input_views
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        # 使用pts_linears对input_pts进行逐层处理
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)  # 激活
            if i in self.skips:  # 若i == skips（4），则将原始input_pts和h进行拼接
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:  # # 若使用视角方向
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)  # 将特征向量feature和输入input_views进行拼接

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)  # 输出rgb和体密度的拼接
        else:
            outputs = self.output_linear(h)  # 直接输出h

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


# Ray helpers
def get_rays(H, W, K, c2w):
    # 与下面的get_rays_np()函数基本一致，只是需要转置操作
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    # H，W，K分别为图像的高宽和相机的内参矩阵
    # c2w为相机到世界坐标系的变换矩阵
    # 函数返回射线的原点和方向
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
# 分层体积采样
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
