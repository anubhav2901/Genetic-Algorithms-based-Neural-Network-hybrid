# Genetic-Algorithms-based-Neural-Network-hybrid
In this project, the Artificial Neural Networks are optimised using the Genetic Algorithms based training algorithm. The problem of finding the optimal set of weights for the Artificial Neural Networks, is transformed into finding the fittest individual in the population.
Here, weights of the ANN are considered as a gene in the chromosome.So every individual in the population represents a possible solution in the search space. Real encoding is used for encoding the weights in the chromosome. 
### Encoding
Every chromosome consists of a set genes and every gene consists of alleles. Here, length of every gene is considered to be the same i.e. 5 by default or can be specified by the user. Every allele is a digit from 0-9. Therefore, an individual is represented by a chromosome of length, n = gene_length * number of weights.

![Individual](https://latex.codecogs.com/gif.latex?I_{i}&space;=&space;x_{1}x_{2}x_{3}........x_{n-1}x_{n})

### Decoding
Since every gene represent a weight, therefore the chromosome is split into equal portions of the specified gene_length. To retrieve the value of weight from the gene, following method is used say, for gene_length = 5.

#### The sign of the weight is determined using: -

<img src="https://latex.codecogs.com/gif.latex?sign(x_{1})&space;=&space;\left\{\begin{matrix}&space;&plus;,&space;&&space;x_{1}>=0&space;&&space;\\&space;-,&space;&&space;x_{1}<0&space;\end{matrix}\right." title="sign(x_{1}) = \left\{\begin{matrix} +, & x_{1}>=0 & \\ -, & x_{1}<0 \end{matrix}\right." />

#### Value of the weight is determined by using: -

<img src="https://latex.codecogs.com/gif.latex?W_{i}&space;=&space;\frac{x_{2}x_{3}x_{4}x_{5}}{10^{genelength-2}}" title="W_{i} = \frac{x_{2}x_{3}x_{4}x_{5}}{10^{genelength-2}}" />

Now using genetic operators, the population evolves and the best traits are passed onto the new generations. 

### Selection
This model uses Elitsm as the process of selection of individuals for reproduction. In Elitism, only the best individuals are allowed to form the mating pool. The weaker individuals are replaced by the best individual and then the mating pool of population participates in the reproduction. For increasing the searching ability of the model and keeping the best known chromosome, only the best individual is put on hold and rest of the individuals in the mating pool takes part in the reproduction.

### Crossover
In crossover, the population is randomly paired for recombination of the genetic material of the parents. For each pair, crossover sites are randomly generated and the genetic material between the sites in exchanged between the parents. The crossover rate is used for determining the size of the mating pool. Crossover rate can be modulated to determining the perfect size of population for mating.

### Mutation
The process of crossover is followed by the process of mutation where some randomness is associated with the composition of the genetic material of the offsprings. Mutation is an essential process as it can introduce a better combination of genetic material in the population. This can be useful for avoiding the convergence of the population to a local minima. Mutation process is modulated using the mutation rate. Having a high value of the mutation rate, can lead to enhanced global search of the model but, it also causes divergence in the model and hence, the model never converges. While having a low value of mutation rate, increases the rate of convergence of the model but, the model can converge to the local minima.

### Inversion
This operator is used as a backup for introducing randomness in the population. Inversion of the genetic material of the individuals can be beneficial to the model as inversion can be viewed as transposition of the individual to the opposite domain in the search space. It can be used for avoiding convergence to the local minima.

## Modeling
### ACTIVATIONS.PY
This file consists of the necessary activation functions for the ANNs. Example, linear, logistic, relu, tanh, and rbf functions.
### ERROR.PY
This file consists of the necessary error functions for the ANNs. Example, MSE, MAE, Mean Log Cosh Error, and Log Loss functions.
### WARNINGS.PY
This file is used for raising computational error in the model.
### GA_NN.PY
This file consists of the implementation of the Genetic Alorithm based ANN classifier.
### GENETIC_OPERATORS.PY
This file consists of the implementation of different genetic operators such as selection, crossover, mutation, and inversion.
### TESTING
For the pupose of testing the model is trained on the Social Network Ads data from the UCI repository. The data is named as adsdata.csv in this repository. Same Neural Network architecture was used for testing both the models.
#### Genetic Algorithm Based Neural Network
![GANN](https://github.com/Snorlexing/Genetic-Algorithms-based-Neural-Network-hybrid/blob/master/Gann.png)
#### Backward Propagation Based Neural Network
![ANN](https://github.com/Snorlexing/Genetic-Algorithms-based-Neural-Network-hybrid/blob/master/nn.png)

### Conclusion and Future Work
Analysis of the results clearly suggests that Genetic Algorithm based Neural Network outperforms Backward Propagation based Neural Network. This model, GA based ANN, can be enhanced by using various techniques such as integrating some momentum to the change of weights for every individual or using additional backward propagation term to change the position of the parents before applying genetic operators.

>References: https://github.com/yugrocks/Genetic-Neural-Network
