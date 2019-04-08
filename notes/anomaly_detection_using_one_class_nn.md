### Anomaly detection: motivation and challenges
_Anomaly_ is the instance that stands out as being dissimilar to all others. The goald of _anomaly detection_ is to determine all these instances in data-driven fashion. __One-class Support Vector Machnes__ (OC-SVM) are videly used unsupervised technique to identify unomalies. However, performacne of OC-SVM is sub-optimal on complex, high dimensional datasets. Deep learning methods for anomaly detection can be classified into deep autoencoders and hybrid models. Models involving autoencoders utilize magnitude of residual vector for making anomaly assesments. Hybrid models mainly use autoencoder as feature extractor, where the hidden layer representations are used as input to traditional anomaly detection algorithms such as OC-SVM. Using generic pre-trained networks is efficient, but sometimes learning representations from scratch for modestly sized datasets is shown to perform better. The paper presents the theory of integrating OC-SVM equivalent objective into the NN. The approach is novel in that way that data representation is driven by the OC-NN objective and thus customized for specific anomaly detection.
### Background and related works on anomaly detection

__Robust Deep Autoencoders for anomaly detection__

Robust Deep Autoencoder (RDA) or Robust Deep Convolutional Autoencoder (RCAE) decompose input data $$X$$ into two parts $$X=L_D+S$$, where $$L_D$$ represents the latent representation of the hidden layer of the autoencoder, matrix $$S$$ captures noise and outliers which are hard to reconstruct. The decomposition is carried out by optimizing following objective funtion
$$\min_{\theta, S}+||L_D-D_\theta(E_\theta(L_D))||_2 + \lambda||S^T||_{2,1}$$
$$s.t. X-L_D-S=0$$
The above optimization problem is solved using a combination of backpropagation and Altenating Direction Method of Multipliers (ADMM)
__One-Class SVM for anomaly detection__
OC-SVM is the special case of SVM whihc learns a hyperplane to separate all the data points from the origin in a reproducing kernel Hilbert space (RKHS) and maximises the distance from this hyperplane to the origin. Intuitevly in OC-SVM all the data points are considered as poisitively labele instances and the origin as the only negative labeled instance.
$$X$$ - the training data without any class information
$$\Phi(X)$$ - a RKHS map function from the input space to the feature space $$F$$
$$f(X_n)=w^T\Phi(X_n)-r$$ - linear decision function in the feature space $$F$$ cunstructed to separate as many as possible of the mapped vectors $$\Phi(X_n)$$ from the origin.
$$w$$ - the norm perpendicular to the hyperplane
$$r$$ - the bias of the hyperplane
Optimization problem to obtain $$w$$ and $$r$$
$$min_{w,r}\frac{1}{2}||w||^2_2+\frac{1}{\nu}\cdot \frac{1}{N}\sum_n^Nmax(0, r-\langle w, \Phi(X_n) \rangle)-r$$
$$\nu\in(0,1)$$ - parameter that controls a trade off between maximizing the distance of the hyper-plane from the origin and the number of data poinsta that allowed to corss the hypr-plane
### From One Class SVM to One Class Neural Networks
The paper proposes a simple feed forward network with one hidden layer having linear or sigmoid activation $$g(\cdot)$$ and one output node with the following objective
$$\min_{w,V,r}\frac{1}{2} ||w||^2_2+\frac{1}{2}||V||^2_F+\frac{1}{\nu}\cdot\frac{1}{N}\sum_{n=1}^N\max(0, \langle(w, g(VX_n)\rangle)-r$$
$$w$$ - scalar output obtained from the hidden to output layer (why to take norm of it then?)
$$V$$ - weight maxtrix from input to the hidden units
The key insight of the paper is to substitute dot product $$\langle w,\Phi(X_n)\rangle$$ in OC-SVM with $$\langle w, g(VX_n) \rangle$$. The change will make it possible to leverage transfer learning features obtained using an autoencoder and create additional layer to refine the feature fro anomaly detection. The price that the problem becomes non-convex.

