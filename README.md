# Small-footprint keyword spotting

<p align="center">
<b>Group members</b> // Daniele Ninni, Nicola Zomer
</p>

In this work, we experiment with several neural network architectures as possible approaches for the keyword spotting (KWS) task.
We run our tests on the Google [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands), one of the most popular datasets in the KWS context.
We define a CNN model that outperforms our baseline model and we use it to study the impact of different preprocessing, regularization and feature extraction techniques.
We see how, for instance, the log Mel-filterbank energy features lead to the best performance and we discover that the introduction of background noise on the training set with a reduction coefficient of 0.5 helps the model to learn.
Then, we explore different machine learning models, such as ResNets, RNNs, attention-based RNNs and Conformers in order to achieve an optimal trade-off between accuracy and footprint.
We find that this architectures offer between a 30-40% improvement in accuracy compared to the baseline, while reducing up to 10x the number of parameters.

## Notebooks

- [Data analysis and preprocessing inspection](./notebooks/01_data_analysis_and_preprocessing_inspection.ipynb) <br>
    This notebook takes care of loading and preparing the dataset, splitting it into training, validation, and testing sets. It also provides some information about the dataset with plots. Moreover, it introduces the functions used to preprocess the data (for example adding noise).

- [Keyword spotting: general training notebook](./notebooks/02_keyword_spotting_intro.ipynb) <br>
    This notebook defines the general training and testing pipeline, giving some information about the validation metrics used. The training is performed using our baseline model `cnn-one-fpool3`, taken from [Arik17].

- [Bayesian optimization and feature comparison with CNN](./notebooks/03_cnn_bo_fc.ipynb) <br>
    This notebook is used to train our custom CNN models. With the first of these models we perform a Bayesian optimization, and we use it for inspecting the importance of dropout and batch normalization, realizing a feature comparison and studying the effect of data augmentation on the training set.

- [Keyword spotting: ResNet architecture and triplet loss implementation](./notebooks/04_resnet.ipynb) <br>
    In this notebook we play with ResNet models for the keyword spotting task. We start by implementing a simple ResNet architecture inspired by [Tang18] and then, motivated by [Vygon21], we modify such model and we train it to get a meaningful embedded representation of the input signals. We finally use k-NN to perform the classification task on these intermediate representations.

- [Keyword spotting: a neural attention model for speech command recognition](./notebooks/05_crnn_with_attention.ipynb) <br>
    This notebook implements an attention model for speech command recognition. It is obtained as a modification of a [Demo notebook](https://github.com/douglas125/SpeechCmdRecognition/blob/master/Speech_Recog_Demo.ipynb) prepared by the authors of the paper [*A neural attention model for speech command recognition*](https://arxiv.org/abs/1808.08929).

- [Keyword spotting: Conformer](./notebooks/06_conformer_bo.ipynb) <br>
    In this notebook, thanks to the library `audio_classification_models`, we implement a baseline Conformer architecture inspired by [Gulati20]. This model combines **Convolutional Neural Networks** and **Transformers** to get the best of both worlds by modeling both local and global features of an audio sequence in a parameter-efficient way. In detail, we use only one Conformer block in order to reduce the number of model parameters. Moreover, we perform hyperparameter tuning by means of Bayesian optimization in order to find, among the models with less than 2M parameters, the one that leads to the best accuracy.

- [Keyword spotting: GAN-based classification](./notebooks/07_conditional_dcgan.ipynb) <br>
    In this notebook we try to implement a GAN-based classifier inspired by the paper [*GAN-based Data Generation for Speech Emotion Recognition*](https://www.isca-speech.org/archive_v0/Interspeech_2020/pdfs/2898.pdf). Unfortunately, to date we have not been able to figure out how to properly train the generator and discriminator in this specific case. As a result, we cannot currently test this approach.

## Utilities

- [Models](./utils/models_utils.py)
- [Plotting](./utils/plot_utils.py)
- [Preprocessing](./utils/preprocessing_utils.py)

## Demo App

In this repository you can find a demo application that can be run as a Python script with `python demo_ks.py`. It allows you to select the model you want to use and, when started, it detects the words of the Speech Commands Dataset through the microphone (or any chosen input device).

You can also find a [notebook](./play_commands.ipynb) that can be used to play some commands from the dataset, in order to test such application with non-real-time signals.

***

<h5 align="center">Human Data Analytics<br>University of Padua, A.Y. 2022/23</h5>

<p align="center">
  <img src="https://user-images.githubusercontent.com/62724611/166108149-7629a341-bbca-4a3e-8195-67f469a0cc08.png" alt="" height="70"/>
</p>