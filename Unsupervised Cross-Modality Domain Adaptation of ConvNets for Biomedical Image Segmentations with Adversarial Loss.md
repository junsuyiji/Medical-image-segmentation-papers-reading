## 论文题目：Unsupervised Cross-Modality Domain Adaptation of ConvNets for Biomedical Image Segmentations with Adversarial Loss

### 论文的思路

面临的问题，泛化convnet？让它可以在不同的域上继续进行工作

对，这就是我们要面对的问题。先看一下常见的泛化模型的方法：**正则，模型优化，对抗攻击**

这篇论文采用了GAN的思想，使用相互博弈来让模型泛化，~~最理想的状态是达到纳什均衡点，不过很难达到。~~

文中，我们将DAM也就是convnet当做生成器，将DCM当做判别器。

如果我们抛开优化和细节的话，先捋一捋大致的思路，应该是这样的

图像首**先输入到 Conv 层**，然后转发到 **3 个残差模块**（称为 RM，每个模块由 2 个堆叠的残差块组成）并**下采样 8 倍**。接下来，将**另外三个 RM 和一个扩张的 RM 堆叠起来形成一个深度网络**。为了扩大提取全局语义特征的感受野，在 RM7 中使用了 4 个扩张卷积层和扩张因数 2。对于我们的分割任务中的密集预测，我们在 Conv10 层进行上采样，然后是5×5 卷积平滑特征图。最后，一个softmax 层用于像素的概率预测。

嗯，这就是正常情况下，不进行域变换时，我们要进行的流程。很顺畅。

但是现在我们要进行域变换，单单是上面的网络已经不足以解决我们面临的问题，嗯对。具体点来说就是我们在这个域上提取到的特征不一定在另一个域上能用，所以我要提取高级特征，提取它们都有的特征，这就是ConvNet Segmenter 应该干的事情，但是仅仅这一个还不行，上面说了ConvNet  Segmenter 相当于生成器，所以再来一个判别器就可以了。

我想具体情况应该是这样：先说一下名字，我们将提取原图像的ConvNet叫做ConvNet，提取另一个目标域的的ConvNet叫做DAM，然后呢，判别器叫做DCM。这个时候ConvNet和DAM同时将提取到的特征，给到了DCM。DCM看了看ConvNet，又看了看DCM，没有说话，只是对DCM直摇头，懂得都懂。然后DCM和DAM就在一次次对抗中不断的进化，终于DAM和ConvNet提取的特征一样，然后DCM就只能开摆，最后DCM也给自己域的图片进行了分割。

### DAM和DCM架构

<img src="https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/uTools_1650025933376.png" alt="uTools_1650025933376" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/uTools_1650026403416.png" alt="uTools_1650026403416" style="zoom:50%;" />

在对抗性学习中，DAM 与对手对抗：一种隐含估计 W($P_s, P_g$) 的判别模型。我们将我们的判别器称为域评论模块并用 DCM表示。具体来说，我们构建的 DCM 由几个堆叠的残差块组成，如上图所示。在每个块中，特征图的数量增加一倍，直到达到 512，同时它们的大小减小。 我们将多层次的特征图连接起来作为 DCM 的输入。 该鉴别器将区分源域和目标域之间的复杂特征空间。 通过这种方式，我们的域适应方法**不仅在开始时删除了特定于源的模式，而且不允许在更高层恢复它们。**在无监督学习中，我**们通过对抗性损失联合优化生成器 M（DAM）和鉴别D（DCM）**。 具体来说，以 $X^t$ 为目标集，学习 DAM 的损失为：

<img src="https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/uTools_1649940830291.png" alt="uTools_1649940830291" style="zoom:67%;" />

此外，$X^s$ 代表一组源图像，DCM 通过以下方式进行优化：

<img src="https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/uTools_1649940896274.png" alt="uTools_1649940896274" style="zoom:67%;" />

在 M 和 D 的交替更新期间，DCM 在来自两个域的特征空间分布之间输出更精确的 W($P_s, P_g$) 估计。 更新后的 DAM 更有效地生成类似源的特征图以进行跨模态域自适应。

**解释一下上面各个字母的意思：**

1. 其中 K 是将 Lipschitz 约束应用于 D 的常数

2. $X^s$ 代表一组源图像

3. **在实践中，我们从冻结的较高层中选择几个层(因为选择一个层不足以具有普遍性)，并将它们对应的特征图称为 $F_H$(·) 的集合，其中 H ={k, ..., q} 是选定层索引的集合。**

4. 我们用 $M_A$(·) 表示 DAM 的选定特征图，其中 A 是选定层集。

5. 目标域的特征空间是（$M_A(x^t),F_H(x^t)$），（$M^s_A(x^s),F_H(x^s)$）是源域的对应空间。

6. 给定 $(M_A(x^t),F_H(x^t)) \sim P_g$ 和 $(M^s_A(x^s),F_H(x^s)) \sim P_s$ 的分布，这两个域分布之间需要最小化的距离表示为 W($P_s, P_g$)。对于稳定的训练，我们使用两个分布之间的 Wassertein 距离，如下所示：

   <img src="https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/uTools_1649940085930.png" alt="uTools_1649940085930" style="zoom:67%;" />

   **其中$\prod_{}^{}$ ($P_s,P_g$) 表示边缘分布为 $P_S$ 和 $P_g$ 的所有联合分布 $\gamma $(x, y) 的集合。**

7. **输入到标签空间的映射 $M^s$ 是以分割 ConvNet 的形式隐式学习的。**

### 损失函数

通过上面我们知道了怎么训练DAM和DCM，但是我们还不知怎么训练这一个完整的模型，因为什么？因为ConvNet是有损失的，下面我们抛弃DAM和DCM回归到这个ConvNET的本身，来看看它的损失。

这里面细节，我们先不谈，仅仅说主体的脉络

我们使用 $N^s$的标记数据集，表示来自源域的样本，这意味着$X^s$ = {($x^s_1,y^s_1$),……($x^s_{N^s},y^s_{N^s}$)}.**我们进行监督学习以建立从输入图像到标签空间 $Y^s$ 的映射**，在我们的设置中，**$x^s_i$代表医学图像的样本（像素或补丁）和 $y^s_i$是解剖结构的类别。**为便于表示，**下面省略索引i，直接用$x^s$和$y^s$表示来自源域的样本和标签。**输入到标签空间的映射 是$M^s$ 

通过最小化由多类交叉熵损失和 Dice 系数损失组成的混合损失 $\mathcal{Lseg}$ 来优化。最初的ConvNet

形式上，我们将 $y^s_{i,c}$ 表示为关于样本 x 中的类 c∈ C 的标签$x^s_i$，其概率预测为 $\hat{p^s_{i,c}}$，标签预测为 $\hat{y^s_{i,c}}$，则
源域分割器损失函数如下：

<img src="https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/uTools_1649934331722.png" alt="uTools_1649934331722" style="zoom:67%;" />

其中第一项是像素级的交叉熵损失分类，其中 $w^s_c$ 是要处理的权重因子阶级不平衡的问题。

现**在我们知道了源域分割器的损失函数，如果我们的DCM和DAM训练好了，也就是说DAM可以生成特征和源域一样的标签，那不就意味着，我们把源域换成目标域不就产生了我们要损失函数了吗**？

### 最后了，来一张完整的图

![uTools_1650028257841](https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/uTools_1650028257841.png)

### 对比

这篇论文和Annotation-Efficient Learning for Medical Image Segmentation based on Noisy Pseudo Labels and Adversarial Learning都是解决域适应的问题，后者用的是cyclegan，前者用的是DAM和DCM。

前者的思路是采用了GAN的博弈思想，后者这是直接在cyclegan上进行改进，嗯，。