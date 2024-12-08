## Data Attribution for Text-to-Image Models by Unlearning Synthesized Images
[**Project**](https://peterwang512.github.io/AttributeByUnlearning/) | [**Paper**](https://arxiv.org/abs/2406.09408)

[Sheng-Yu Wang](https://peterwang512.github.io/)<sup>1</sup>, [Aaron Hertzmann](https://www.dgp.toronto.edu/~hertzman/)<sup>2</sup>, [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/)<sup>3</sup>, [Jun-Yan Zhu](https://cs.cmu.edu/~junyanz)<sup>1</sup>, [Richard Zhang](http://richzhang.github.io/)<sup>2</sup>.
<br> Carnegie Mellon University<sup>1</sup>, Adobe Research<sup>2</sup>, UC Berkeley<sup>3</sup>
<br>In NeurIPS, 2024.

<p align="center">
<img src="teaser.png" width="800px"/>
</p>

### Abstract
The goal of data attribution for text-to-image models is to identify the training images that most influence the generation of a new image. Influence is defined such that, for a given output, if a model is retrained from scratch without the most influential images, the model would fail to reproduce the same output. Unfortunately, directly searching for these influential images is computationally infeasible, since it would require repeatedly retraining models from scratch. In our work, we propose an efficient data attribution method by simulating unlearning the synthesized image. We achieve this by increasing the training loss on the output image, without catastrophic forgetting of other, unrelated concepts. We then identify training images with significant loss deviations after the unlearning process and label these as influential. We evaluate our method with a computationally intensive but "gold-standard" retraining from scratch and demonstrate our method's advantages over previous methods.

### Code layout
Please refer to the subdirectories for different sets of experiments.
| Directory          | Experiments |
| :-------------: |:-------------:|
| mscoco | attribution for text-to-image diffusion models trained on MSCOCO |
| custom_diffusion | attribution for customized text-to-image models from [AbC](https://github.com/PeterWang512/GenDataAttribution/tree/main) benchmark |

### Citation
```
@inproceedings{wang2024attributebyunlearning,
  title={Data Attribution for Text-to-Image Models by Unlearning Synthesized Images},
  author={Wang, Sheng-Yu and Hertzmann, Aaron and Efros, Alexei A and Zhu, Jun-Yan and Zhang, Richard},
  booktitle={NeurIPS},
  year = {2024},
}
```

### Acknowledgements
We thank Kristian Georgiev for answering all of our inquiries regarding JourneyTRAK implementation and evaluation, and providing us their models and an earlier version of JourneyTRAK code. We thank Nupur Kumari, Kangle Deng, Grace Su for feedback on the draft. This work is partly supported by the Packard Fellowship, JPMC Faculty Research Award, and NSF IIS-2239076.