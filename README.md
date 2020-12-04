# Towards-Transferable-Targeted-Attack
Codes for CVPR2020 paper "Towards Transferable Targeted Attack".

## checkpoints

The used normally trained and adversarially trained models are available in
[https://drive.google.com/drive/folders/1dHFQSCHsfClbz9h1Q_G1_wkKxJpu1yKh?usp=sharing](https://drive.google.com/drive/folders/1dHFQSCHsfClbz9h1Q_G1_wkKxJpu1yKh?usp=sharing)

Please put them in the fold `./checkpoint`.

## Guide

`incep_v3_trip_po.sh`, `incep_v3_ce.sh`, and `incep_v3_po.sh` are three examples to run our attacks and the baselines. Everyone can change the python file names in these `.sh` files to attack different models.

Set the probability of DI<sup>2</sup>-FGSM, use `--prob=0.7`, `--prob=0` means no diverse input pattern.
Set TI-FGSM, uncomment the line ``noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')`` in function `graph`.
Set MI-FGSM, use `--prob=0` and comment out this line mentioned above.

##  Cite by

```latex
@inproceedings{li2020towards,
  title={Towards Transferable Targeted Attack},
  author={Li, Maosen and Deng, Cheng and Li, Tengjiao and Yan, Junchi and Gao, Xinbo and Huang, Heng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={641--649},
  year={2020}
}
```
