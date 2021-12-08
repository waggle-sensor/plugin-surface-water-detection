# DeepLab with PyTorch

This is an **PyTorch** implementation of **DeepLab v2** [[1](##references)] with a **ResNet-101** backbone. 
* **COCO-Stuff** dataset [[2](##references)] is supported.

## Performance

### COCO-Stuff

<table>
    <tr>
        <th>Train set</th>
        <th>Eval set</th>
        <th>Code</th>
        <th>Weight</th>
        <th>CRF?</th>
        <th>Pixel<br>Accuracy</th>
        <th>Mean<br>Accuracy</th>
        <th>Mean IoU</th>
        <th>FreqW IoU</th>
    </tr>
        <td rowspan="2">
            164k <i>train</i>
        </td>
        <td rowspan="2">164k <i>val</i></td>
        <td rowspan="2"><strong>This repo</strong></td>
        <td rowspan="2"><a href="https://github.com/kazuto1011/deeplab-pytorch/releases/download/v1.0/deeplabv2_resnet101_msc-cocostuff164k-100000.pth">Download</a> &Dagger;</td>
        <td></td>
        <td>66.8</td>
        <td>51.2</td>
        <td>39.1</td>
        <td>51.5</td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>67.6</td>
        <td>51.5</td>
        <td>39.7</td>
        <td>52.3</td>
    </tr>
</table>

&dagger; Images and labels are pre-warped to square-shape 513x513<br>
&Dagger; Note for [SPADE](https://nvlabs.github.io/SPADE/) followers: The provided COCO-Stuff 164k weight has been kept intact since 2019/02/23.

## Setup

### Download datasets

* [COCO-Stuff 10k/164k](data/datasets/cocostuff/README.md)

| Dataset         | Config file                  | #Iterations | Classes                      |
| :-------------- | :--------------------------- | :---------- | :--------------------------- |
| COCO-Stuff 164k | `configs/cocostuff164k.yaml` | 100,000     | 182 thing/stuff              |

Note: Although the label indices range from 0 to 181 in COCO-Stuff 10k/164k, only [171 classes](https://github.com/nightrome/cocostuff/blob/master/labels.md) are supervised.

## References

1. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image
Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*,
2018.<br>
[Project](http://liangchiehchen.com/projects/DeepLab.html) /
[Code](https://bitbucket.org/aquariusjay/deeplab-public-ver2) / [arXiv
paper](https://arxiv.org/abs/1606.00915)

2. H. Caesar, J. Uijlings, V. Ferrari. COCO-Stuff: Thing and Stuff Classes in Context. In *CVPR*, 2018.<br>
[Project](https://github.com/nightrome/cocostuff) / [arXiv paper](https://arxiv.org/abs/1612.03716)

1. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object
Classes (VOC) Challenge. *IJCV*, 2010.<br>
[Project](http://host.robots.ox.ac.uk/pascal/VOC) /
[Paper](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)

## funding
[NSF 1935984](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1935984)

## collaborators
Bhupendra Raut, Dario Dematties Reyes, Joseph Swantek, Neal Conrad, Nicola Ferrier, Pete Beckman, Raj Sankaran, Robert Jackson, Scott Collis, Sean Shahkarami, Sergey Shemyakin, Wolfgang Gerlach, Yongho kim
