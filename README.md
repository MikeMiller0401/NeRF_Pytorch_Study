# NeRF-Pytorch-Study

NeRF在三维重建和新视角合成上的新思路和照片级的合成结果激发了我对该方法的兴趣，这是我学习该方法的一个记录。

## 一、项目的结构
三个文件夹：configs 包含数据集的配置文件；data 用于存放数据集；logs 用于存放结果和日志文件。

六个python脚本：其中四个 load_XXX.py 均为数据集的读取脚本，run_nerf.py 为 nerf 的主要文件，run_nerf_helpers.py 中编写了 nerf 的一些组件。

## 二、流程
**NeRF 的 MLP 的输入是一系列点的五维坐标，输出的是预测的点的颜色和体密度**，那么这些点的坐标是如何获得的呢？这就涉及到了光线生成的步骤。
### 1、光线生成
将穿过图片某个像素的一条光线离散化就能够得到一系列点，而确定光线的方向就能获得这些点的数学表示。

假设图片坐标系中某个像素的坐标:

$$ [x, y]^T $$

那么相机坐标系下该像素的坐标:

$$ [X_c, Y_c, Z_c]^T = [x, y]^T * K^{-1}（K为 3*3 的内参矩阵） $$

世界坐标系下该像素的坐标：

$$ [X, Y, Z, 1]^T = [X_c, Y_c, Z_c, 1]^T * c2w^{-1}（c2w为 4*4 的外参矩阵） $$

相机的位置和朝向由相机的外参（extrinsic matrix）决定，投影属性由相机的内参（intrinsic matrix）决定，NeRF 的内外参都是给定的。
    
通过相机的内外参数，可以将一张图片的**像素坐标**转换为**统一的世界坐标系下的坐标**,我们可以确定一个坐标系的**原点 o**,而一张图片的每个像素都可以根据原点以及图片像素的位置计算出该像素相对于原点的**单位方向向量 d**,改变不同的**深度 t**，就可以通过构建一系列离散的点模拟出一条经过该像素的光线:
$$ r(t)=o+td $$
这些点的坐标和方向就是 NeRF 的 MLP 输入,输出经过体渲染得到的值与这条光线经过的像素的值得到 loss。

这部分的实现是在 run_nerf_helpers.py 文件中的 def get_rays_np(H, W, K, c2w)函数中实现的：
```python
def get_rays_np(H, W, K, c2w):
    """
    H，W，K分别为图像的高宽和相机的内参矩阵；c2w为相机坐标系到世界坐标系的变换矩阵（相机的外参矩阵）
    返回值：光线的原点和方向
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')  # 返回大小[H*W]的i和j，i的每一行代表x轴坐标，j的每一行代表y轴坐标。
    # 计算每个像素坐标相对于光心的单位方向，得到每个像素点关于光心o的方向dir
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)   #
    # 将方向dir从相机坐标系转变为世界坐标系
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # 将原点从相机坐标系转变为世界坐标系，这也是这条光线上所有点的原点
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d
```
### 2、位置编码

位置编码的原理和作用在论文的 5.1 进行了论述，这里主要关注一下具体的实现方法，设计到的主要公式为：

<img src='imgs/pe.png'/>

代码位于 run_nerf_helpers.py 中：
```python
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
```

返回的是一个编码函数embed以及输出的维度embedder_obj.out_dim，其用法仅在 run_nerf.py 的create_nerf(args)中：
```python
def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # 使用get_embedder函数获取位置编码器embed_fn和输入mlp的坐标的维度
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    ...
    ...
```

### 3、MLP的构建
论文的 fig.7 给出了MLP的结构：

<img src='imgs/mlp.png'/>

代码同样位于 run_nerf_helpers.py 中：
```python
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
```
其用法同样仅在 run_nerf.py 的create_nerf(args)中：
```python
def create_nerf(args):
  ...
  ...
  model = NeRF(D=args.netdepth, W=args.netwidth,
              input_ch=input_ch, output_ch=output_ch, skips=skips,
              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
  ...
  if args.N_importance > 0:
      model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                       input_ch=input_ch, output_ch=output_ch, skips=skips,
                       input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
  ...
  ...
```

----

## 以下内容为yenchenlin博士改写的基于pytorch的nerf[仓库](https://github.com/yenchenlin/nerf-pytorch)中的README

----

[NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields) is a method that achieves state-of-the-art results for synthesizing novel views of complex scenes. Here are some videos generated by this repository (pre-trained models are provided below):

![](https://user-images.githubusercontent.com/7057863/78472232-cf374a00-7769-11ea-8871-0bc710951839.gif)
![](https://user-images.githubusercontent.com/7057863/78472235-d1010d80-7769-11ea-9be9-51365180e063.gif)

This project is a faithful PyTorch implementation of [NeRF](http://www.matthewtancik.com/nerf) that **reproduces** the results while running **1.3 times faster**. The code is based on authors' Tensorflow implementation [here](https://github.com/bmild/nerf), and has been tested to match it numerically. 

## Installation

```
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - PyTorch 1.4
  - matplotlib
  - numpy
  - imageio
  - imageio-ffmpeg
  - configargparse
  
The LLFF data loader requires ImageMagick.

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.
  
</details>

## How To Run?

### Quick Start

Download data for two example datasets: `lego` and `fern`
```
bash download_example_data.sh
```

To train a low-res `lego` NeRF:
```
python run_nerf.py --config configs/lego.txt
```
After training for 100k iterations (~4 hours on a single 2080 Ti), you can find the following video at `logs/lego_test/lego_test_spiral_100000_rgb.mp4`.

![](https://user-images.githubusercontent.com/7057863/78473103-9353b300-7770-11ea-98ed-6ba2d877b62c.gif)

---

To train a low-res `fern` NeRF:
```
python run_nerf.py --config configs/fern.txt
```
After training for 200k iterations (~8 hours on a single 2080 Ti), you can find the following video at `logs/fern_test/fern_test_spiral_200000_rgb.mp4` and `logs/fern_test/fern_test_spiral_200000_disp.mp4`

![](https://user-images.githubusercontent.com/7057863/78473081-58ea1600-7770-11ea-92ce-2bbf6a3f9add.gif)

---

### More Datasets
To play with other scenes presented in the paper, download the data [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Place the downloaded dataset according to the following directory structure:
```
├── configs                                                                                                       
│   ├── ...                                                                                     
│                                                                                               
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern                                                                                                                             
│   │   └── flower  # downloaded llff dataset                                                                                  
│   │   └── horns   # downloaded llff dataset
|   |   └── ...
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── ship    # downloaded synthetic dataset
|   |   └── ...
```

---

To train NeRF on different datasets: 

```
python run_nerf.py --config configs/{DATASET}.txt
```

replace `{DATASET}` with `trex` | `horns` | `flower` | `fortress` | `lego` | etc.

---

To test NeRF trained on different datasets: 

```
python run_nerf.py --config configs/{DATASET}.txt --render_only
```

replace `{DATASET}` with `trex` | `horns` | `flower` | `fortress` | `lego` | etc.


### Pre-trained Models

You can download the pre-trained models [here](https://drive.google.com/drive/folders/1jIr8dkvefrQmv737fFm2isiT6tqpbTbv). Place the downloaded directory in `./logs` in order to test it later. See the following directory structure for an example:

```
├── logs 
│   ├── fern_test
│   ├── flower_test  # downloaded logs
│   ├── trex_test    # downloaded logs
```

### Reproducibility 

Tests that ensure the results of all functions and training loop match the official implentation are contained in a different branch `reproduce`. One can check it out and run the tests:
```
git checkout reproduce
py.test
```

## Method

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution  
  
<img src='imgs/pipeline.jpg'/>

> A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views


## Citation
Kudos to the authors for their amazing results:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

However, if you find this implementation or pre-trained models helpful, please consider to cite:
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
