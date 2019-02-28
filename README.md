# CMR_GAN
基于生成式对抗网络的去除核磁共振图像伪影的研究

下图就是一张含有伪影的MRI图像，这副图像的伪影比较明显。大部分实际的伪影是不太明显的（有些只有专业的医生才能分辨）（上面的图像比较明显，下面的不明显）

<img src="https://github.com/buptzhang0414/Multi-scaleSR_For_MRI_Blur/blob/master/blurImage.jpg" width="256pt" height="256pt">

<img src="https://github.com/buptzhang0414/Multi-scaleSR_For_MRI_Blur/blob/master/blurImage_little.jpg" width="256pt" height="256pt">

这个代码的基本思想是使用生成式对抗网络来矫正核磁共振图像的伪影。

网络结构如下

<img src="https://github.com/buptzhang0414/CMR_GAN/blob/master/pic/network.png" width="506pt" height="448pt">

部分结果如下
<img src="https://github.com/buptzhang0414/CMR_GAN/blob/master/pic/Fig1_Result.png" width="440pt" height="564pt">

执行[train_baseline_all.sh](https://github.com/buptzhang0414/CMR_GAN/blob/master/train_baseline_all.sh)的脚本即可开始训练。

由于数据集是非公开数据集，所以会在后续公布。

代码注解后续更新。
