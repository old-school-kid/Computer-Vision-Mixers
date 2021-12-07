# MLP-Mixer
Tensorflow/Keras implementation of the paper "MLP-Mixer: An all-MLP Architecture for Vision"

My implementation of the paper [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) (tolstikhin et.al., 2021). It introduces a complete Computer Vision architecture based only on MLP(Multi Layer Percepetron).

![Model architecture](https://github.com/old-school-kid/MLP-Mixer/blob/main/images/MLP%20mixer.png "Model architecture")

Model was trained for only 30 epochs for restricted computational resources on CIFAR-10 dataset. It achived an accuracy of 78.94% on training data and 70.15% on test data.

![Loss plot](https://github.com/old-school-kid/MLP-Mixer/blob/main/images/loss%20mlp%20mixer.png "loss plot") ![Accuracy plot](https://github.com/old-school-kid/MLP-Mixer/blob/main/images/accuracy%20mlp%20mixer.png "Accuracy plot")

### ToDo
- [ ] Make a wrapper around 
- [ ] Add more patches model
- [ ] Add Conv Mixer
- [ ] Option to prune keras model
- [ ] Perturbation Control
- [ ] CLI support
- [ ] Documentation