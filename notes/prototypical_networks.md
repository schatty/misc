## Prototypical Networks for Few-shot Learning

* Few-shot learning - problem at which we aim to classifiy classes not seen during training given only a few examples of each of these classes.
* The model from 2016 paper __Matching Networks__ uses attention over a learned embedding of the labeled set of examples to predict corresponding unlabeld samples. It can be interpreted as a weighted nearest-neighbor classifier applied within an embedding space.
* Matching Networks introduce notation of _episodes_, mini batches that mimic few shot learning task i.e. contains labeled set of unseen classes and corresponding unlabeled set of the same classes which we try to predict.
* __Prototypical Networks__ is based on assumption that there exists and embedding in which points of the same class cluster around a single prototype representation for that class. To learn embedding ResNet like CNN is used producing high-level feature for the sample. Then the mean of the support examples of the class represent the _prototype_. Classification is performed just as finding nearest prototype for the unlabeled sample.

### Mathematical Formulation

* __episode__ - the set of __support__ set of labeled samples and corresponding __query__ set of samples we try to predict.
* __way__ - number of classes involved in episode
* __shot__ - number of support examples for each class. 1-shot learning means that we have only 1 example to train on.
_Notation_
* $$S=\{ (x_1,y_1),...,(x_N,y_N) \}$$ - support set
* $$x_i \in \mathbb{R}^D$$ 
* $$y_i \in \{ 1, ..., K \}$$
* $$c_k \in \mathbb{R}^M$$ - prototype
* $$f_\phi:\mathbb{R}^D\to\mathbb{R}^M$$ - embedding function
* $$c_k =\frac{1}{|S_k|}\sum_{(x_i, y_i)\in S_k} f_\phi(x_i)$$ - prototype calculation
* $$d:\mathbb{R}^M\times\mathbb{R}^M\to[0, + \infty]$$ - distance function to produce $$p_\phi$$
* $$p_\phi(y=k|x)=\frac{\exp(-d(f_\phi(x), c_k)}{\sum_{k'}\exp(-d(f_\phi(x), c_{k'})}$$ - distribution over classes for a query point
* $$J(\phi)=-\log{p_\phi}(y=k|x)$$

### Training Procedure
1. Select class indices for episode
2. For each of the class
2.1. Select support and query examples for each class
2.2. Forward them through encoder
2.3. Compute prototype for support examples
3. For each of the class
3.3 Add to the loss log of sum of exponents over negative distances between support objects and computed prototypes

### Design Choices
* For the __distance function__ different metrics can be applied. For both matching and prototypical networks euclidian and cosine distance funcions are permissible. Experiments showed that using squared euclidian distance improved results for both models.
* Different strategies can be employed for __episode composition__. The most evidient way is to choose way parameter to be equal in the train and test parts. However, experiments showed that using greater way in training is highly benefitial. On the other hand using the same shot value in the train and test times is usually the best choice.

### Zero-Shot Learning
* Zero shot learning is problem when instead of having support set of actual training points we are given class meta-data vector $$v_k$$ for each class
* the data vector may be determined in advance or learned from given raw text
* the modification to deal with zero-shot learning problems is introduce seperate embedding to the meta-data vector $$c_k=g_v(v_k)$$

## Experiments
### Omniglot Settings
* Omniglot contains 1623 handwritten characters collected from 50 alphabet with 20 examples of each character. Each image is 28x28 binary grid.
* Each class was rotated in 90 degrees three times producing 3 more classes.
* Encoder has architecture of four blocks of Conv2D->BatchNorm->ReLU->MaxPool.
* way was set to 60 for training, number of support and query points are 5 per class.
* classification accuracy is computed as mean of 1000 episodes.
* to my knowledge validation set was merged to training to achive higher metrics.
* Results outperformed Matching Networks as follows
    * 5-way-1-shot (98.8% vs 98.1)
    * 5-way-5-shot (99.7% vs 98.9%) 

### MiniImagenet Settings
* Spilts by Vynials consist of 60k 84x84x3 images dividen into 100 classes  with 600 examples each.
* We have 64 classes for training, 16 for validation and 20 for test.
* The encoder architecture is the same, but now embedding is 1600-dimensional
* way was set to 30 for training 1-shot and 20 for 5-shot. Query set consists of 15 images for each class
* Results outperformed Matching Networks as follows
    * 5-way-1-shot: 98.8% vs 98.1%
    * 5-way-5-shot: 99.7% vs 98.9%



