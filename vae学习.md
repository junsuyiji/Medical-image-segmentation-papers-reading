# VAE学习

## 基础知识

### 什么是VAE

变分自编码器（Variational Autoencoder，VAE），依据李宏毅老师的讲解，**VAE作为一个生成模型，其基本思路是很容易理解的：把一堆真实样本通过编码器网络变换成一个理想的数据分布，然后这个数据分布再传递给一个解码器网络，得到一堆生成样本，生成样本与真实样本足够接近的话，就训练出了一个自编码器模型。**

那VAE(变分自编码器)就是在自编码器模型上做进一步变分处理，使得编码器的输出结果能对应到目标分布的均值和方差，如下图所示，：

![img](https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/1704791-20200830201004477-1353724877.png)

**我们想建一个产生式模型，而不是一个只是储存图片的网络。现在我们还不能产生任何未知的东西，因为我们不能随意产生合理的潜在变量。因为合理的潜在变量都是编码器从原始图片中产生的。这里有个简单的解决办法。我们可以对编码器添加约束，就是强迫它产生服从单位高斯分布的潜在变量。正是这种约束，把VAE和标准自编码器给区分开来了**

现在，产生新的图片也变得容易：**我们只要从单位高斯分布中进行采样，然后把它传给解码器就可以了。**

事实上，我们还需要在重构图片的精确度和单位高斯分布的拟合度上进行权衡。

我们可以让网络自己去决定这种权衡。对于我们的损失函数，我们可以把这两方面进行加和。一方面，是图片的重构误差，我们可以用平均平方误差来度量，另一方面。我们可以用KL散度来度量我们潜在变量的分布和单位高斯分布的差异。

为了优化KL散度，我们需要应用一个简单的参数重构技巧：不像标准自编码器那样产生实数值向量，VAE的编码器会产生两个向量:一个是均值向量，一个是标准差向量。

vae和ae有什么区别,下面这张图

![img](https://img2020.cnblogs.com/blog/1704791/202008/1704791-20200830201006202-1550811897.png)

![img](https://img2020.cnblogs.com/blog/1704791/202008/1704791-20200830201006575-286481737.png)

你,看懂了吗?

### VAE和ＧＡＮ的区别,VAE和AE的区别

#### GAN

VAE（Variational Auto-Encoder）和GAN（Ganerative Adversarial Networks）都是生成模型（Generative model）

AE是可以理解为VAE中encoder输出的方差为0的一种情况，这个时候就是单值映射了。GAN中引入的随机数是为了输出的多样性，而VAE引入随机数是为了增大泛化性。

相比于变分自编码器, GAN没有引入任何决定性偏置( deterministic bias),变分方法引入决定性偏置,因为他们优化对数似然的下界,而不是似然度本身,这看起来导致了VAE生成的实例比GAN更模糊。当然要区别看待这个问题：**GAN的目的是为了生成，而VAE目的是为了压缩，目的不同效果自然不同。**

#### AE

**相同点**：两者都是X->Z->X'的结构

**不同点**：AE寻找的是单值映射关系，即：z=f(x)。而VAE寻找的是分布的映射关系，即：DX→DZ

## 数学理论推导

现在我们假设知道的确有这样的latent variables，但是我们不知道它到底是一个怎样的分布，现在能够观测到的就是样本 x**,希望在x的条件下,推出z的分布,即p(z|x)**

根据贝叶斯公式可知:
$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}
$$
但是后验概率p(z|x)是不可解的,因为p(x)无法计算

根据全概率公式可知:$p(x) = \int p(x|z)p(z) dz$

如果z是一个维度很高的变量，你想想是不是有无穷多重的积分，会变成大概这样的计算式子:
$$
p(x) = \int\int \int \int \int \int \int .... p(x|z)p(z) dz
$$
于是这里大致有两种方法求解： - ① 蒙特卡罗算法（Mote Carlo method），不断取样z，取样越多p(x)越接近真实的分布。 - ② Variational Inference，既然 p(z|x)不可解,那么我就尝试用一可解的q(z|x)来逼近p(z|x)即approximation的方式。

我们希望这两个分布间的差异越小越好，KL散度就可以派上用场,我们希望 $minKL(p(z|x) || q(z|x))$

![[公式]](https://www.zhihu.com/equation?tex=KL%28q%28z%7Cx%29%7C%7Cp%28z%7Cx%29%29%3D%5Cint+q%28z%7Cx%29log%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28z%7Cx%29%7Ddz+%5C%5C%3D+%5Cint+q%28z%7Cx%29log%5Cfrac%7Bq%28z%7Cx%29%7D%7B%5Cfrac%7Bp%28x%7Cz%29p%28z%29%7D%7Bp%28x%29%7D%7Ddz+%5C%5C%3D+%5Cint+q%28z%7Cx%29log+q%28z%7Cx%29dz%2B%5Cint+q%28z%7Cx%29logp%28x%29dz-%5Cint+q%28z%7Cx%29log%5Bp%28x%7Cz%29p%28z%29%5Ddz+%5C%5C%3D+%5Cint+q%28z%7Cx%29log+q%28z%7Cx%29dz%2Blogp%28x%29+%5Cint+q%28z%7Cx%29dz-%5Cint+q%28z%7Cx%29log%5Bp%28x%7Cz%29p%28z%29%5Ddz+%EF%BC%88%E6%B3%A8%E6%84%8F%5Cint+q%28z%7Cx%29dz%3D1%EF%BC%89+%5C%5C%3D+logp%28x%29+%2B+%5Cint+q%28z%7Cx%29log+q%28z%7Cx%29dz-%5Cint+q%28z%7Cx%29log%5Bp%28x%7Cz%29p%28z%29%5Ddz+%EF%BC%88%E6%8A%8A%E7%AC%AC%E4%BA%8C%E9%A1%B9%E6%8F%90%E5%89%8D%EF%BC%89%5C%5C+)

$minKL(p(z|x) || q(z|x))$等价于最小化最后两项，我们记作为L:

![[公式]](https://www.zhihu.com/equation?tex=L%3D+%5Cint+q%28z%7Cx%29log+q%28z%7Cx%29dz-%5Cint+q%28z%7Cx%29log%5Bp%28x%7Cz%29p%28z%29%5Ddz+%5C%5C%3D+%5Cint+q%28z%7Cx%29log+q%28z%7Cx%29dz-%5Cint+q%28z%7Cx%29logp%28x%7Cz%29dz-%5Cint+q%28z%7Cx%29logp%28z%29dz+%5C%5C%3D+%5Cint+q%28z%7Cx%29log+%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28z%29%7Ddz-%5Cint+q%28z%7Cx%29logp%28x%7Cz%29dz+%5C%5C%3DKL%28q%28z%7Cx%29%7C%7Cp%28z%29%29-+E_%7Bz%E6%9C%8D%E4%BB%8Eq%28z%7Cx%29%7D%5Blogp%28x%7Cz%29%5D)

即最大化

![[公式]](https://www.zhihu.com/equation?tex=E_%7Bz%E6%9C%8D%E4%BB%8Eq%28z%7Cx%29%7D%5Blogp%28x%7Cz%29%5D-KL%28q%28z%7Cx%29%7C%7Cp%28z%29%29.................................%EF%BC%88I%EF%BC%89)

看这个式子:

第一项实际上是一个重建error，为什么？好，我们说从z到x_hat是的转换是神经网络干的事情，它是一个函数，虽然说我们不知道它具体的表达式是什么，无论我输入一个怎样的input，总是会给我一个相应的output，所以 (I)中第一项可以看作logp(x|x_hat).假设p是一个正态分布，你自己想想会是什么均方误差,如果你假设p是伯努利分布，那就是交叉熵了。

第二项，我们说是为了让我们假设的后验证分布q(z|x)和先验分布p(z)尽量接近,论文中假设p(z)是一个标准高斯分布，为什么这么假设呢？

理论上，任何一个分布都可以由正态分布经过一个复杂函数变换映射得到，一定程度上，你可以认为decoder 网络的前几层就是真的在试图从正态分布转化到latent variable 空间上。所以，正态分布的随机变量其实是latent variable 的”前身”。 

然后将q(z|x)也当作一个正态分布,然后求解

![img](https://pic2.zhimg.com/80/v2-08091fc4fa1460c9fa611cfcf0608105_1440w.jpg)



这里的 *d* 是隐变量 *Z* 的维度，而 *μ*(*i*) 和 σ_{(i)}^{2} 分别代表一般正态分布的均值向量和方差向量的第 *i* 个分量。直接用这个式子做补充 loss，就不用考虑均值损失和方差损失的相对比例问题了。

显然，这个 loss 也可以分两部分理解：



![img](https://pic4.zhimg.com/80/v2-af1049578e84eddf1c817422aa8a3bbf_1440w.jpg)

由于我们考虑的是各分量独立的多元正态分布，因此只需要推导一元正态分布的情形即可，根据定义我们可以写出：

![img](https://pic3.zhimg.com/80/v2-7a3c7ea64e7f11c475cf35cd44fa3ca2_1440w.jpg)

整个结果分为三项积分，第一项实际上就是 −log*σ^*2 乘以概率密度的积分（也就是 1），所以结果是 −log*σ^*2；第二项实际是正态分布的二阶矩，熟悉正态分布的朋友应该都清楚正态分布的二阶矩为 μ^2  + σ^2；而根据定义，第三项实际上就是“-方差除以方差=-1”。所以总结果就是：

![img](https://pic4.zhimg.com/80/v2-603cd66d01ad6bac42ac1d2a38bad61f_1440w.jpg)

## VAE常见问题解答

1. VAE 的本质是什么？VAE 虽然也称是 AE（AutoEncoder）的一种，但它的做法（或者说它对网络的诠释）是别具一格的。

答:**它本质上就是在我们常规的自编码器的基础上，对 encoder 的结果（在VAE中对应着计算均值的网络）加上了“高斯噪声”，使得结果 decoder 能够对噪声有鲁棒性；而那个额外的 KL loss（目的是让均值为 0，方差为 1），事实上就是相当于对 encoder 的一个正则项，希望 encoder 出来的东西均有零均值。**

1. 是不是必须选择正态分布？可以选择均匀分布吗？

   答:这个本身是一个实验问题，两种分布都试一下就知道了.但是从直觉上来讲，正态分布要比均匀分布更加合理，因为正态分布有两组独立的参数：均值和方差，而均匀分布只有一组。

   **在 VAE 中，重构跟噪声是相互对抗的，重构误差跟噪声强度是两个相互对抗的指标，而在改变噪声强度时原则上需要有保持均值不变的能力，不然我们很难确定重构误差增大了，究竟是均值变化了（encoder的锅）还是方差变大了（噪声的锅）**。

   而均匀分布不能做到保持均值不变的情况下改变方差，所以正态分布应该更加合理。

2. **变分在哪里**

答:因为 *KL*(*p*(*x*)‖*q*(*x*))实际上是一个泛函，要对泛函求极值就要用到变分法，当然，这里的变分法只是普通微积分的平行推广，还没涉及到真正复杂的变分法。而 VAE 的变分下界，是直接基于 KL 散度就得到的。所以直接承认了 KL 散度的话，就没有变分的什么事了.

一句话，VAE 的名字中“变分”，是因为它的推导过程用到了 KL 散度及其性质。



**如果有标签数据，那么能不能把标签信息加进去辅助生成样本呢？**

当然，这是肯定可以的，我们把这种情况叫做 **Conditional VAE**，或者叫 CVAE（相应地，在 GAN 中我们也有个 CGAN）。

但是，CVAE 不是一个特定的模型，而是一类模型，总之就是把标签信息融入到 VAE 中的方式有很多，目的也不一样。这里基于前面的讨论，给出一种非常简单的 VAE。





![img](https://pic3.zhimg.com/80/v2-d148bf520bb386c0c4bb11756e4798ee_1440w.jpg)



**▲** 一个简单的CVAE结构

在前面的讨论中，我们希望 *X* 经过编码后，*Z* 的分布都具有零均值和单位方差，这个“希望”是通过加入了 KL loss 来实现的。

如果现在多了类别信息 *Y*，**我们可以希望同一个类的样本都有一个专属的均值** ***μ^Y\*（方差不变，还是单位方差），这个** ***μ^Y*** **让模型自己训练出来**。

这样的话，有多少个类就有多少个正态分布，而在生成的时候，我们就可以**通过控制均值来控制生成图像的类别**。

事实上，这样可能也是在 VAE 的基础上加入最少的代码来实现 CVAE 的方案了，因为这个“新希望”也只需通过修改 KL loss 实现：





![img](https://pic2.zhimg.com/80/v2-6cfa68ced2a4a089f8db3da7236f7129_1440w.jpg)

## 代码实现

```python
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if torch.cuda.is_available():
    model.cuda()

reconstruction_function = nn.MSELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(img),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                loss.data / len(img)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataloader.dataset)))
    if epoch % 10 == 0:
        save = to_img(recon_batch.cpu().data)
        save_image(save, './vae_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './vae.pth')
```

## 学习的文章

1. [变分自编码器VAE：原来是这么一回事 | 附开源代码 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/34998569)
2. [变分自编码器介绍、推导及实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/83865427)
3. https://www.cnblogs.com/yifanrensheng/p/13586468.html
4. https://www.cnblogs.com/huangshiyu13/p/6209016.html
5. 