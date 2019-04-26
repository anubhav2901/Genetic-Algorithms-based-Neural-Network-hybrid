# implementation of genetic backpropagation neural network
import numpy as np
import genetic_operators as go
import warnings
from WARNINGS import ComputationWarning
from ACTIVATIONS import activations
from ERROR import errors


class GBPMultiLayerPerceptron:
    """
        Base class for Multi Layer Perceptron having Genetic Backpropagation.
        MLP developed can be used for both Classification and Regression problems.

        WARNING:
            Do not use this class directly.
            Use derived classes instead.
    """

    # initialize parameters
    def __init__(self, gene_length, population_size, hidden_layer,
                 cross_over_rate, mutation_rate, inversion_rate,
                 max_generation, activation, error):
        self.gene_length = gene_length
        self.population_size = population_size
        self.hidden_layer = hidden_layer
        self.h_layer = len(hidden_layer)
        self.cross_over_rate = cross_over_rate
        self.mutation_rate = mutation_rate
        self.inversion_rate = inversion_rate
        self.max_generation = max_generation
        self.activation = activations[activation]
        self.error = errors[error]

    def _initialize(self):
        self.genes = None
        self.chromosome_length = None
        self.population = None
        self.fitness = []
        self.fit_gen = []
        self.shapes = []
        self.optimum_matrices = None


    def _set_matrices_shape(self, x, y):
        """
        Determines dimensions of weights matrices.

        :param x: training data {n_samples, n_features}
        :param y: target data {n_samples, n_target}

        """
        # determine number of nodes
        # for input layer
        input_nodes = 1
        x_dim = x.shape
        if len(x_dim) != 1:
            input_nodes = x_dim[1]
        # for output layer
        output_nodes = 1
        y_dim = y.shape
        if len(y_dim) != 1:
            output_nodes = y_dim[1]
        # determine shape of weight matrices
        row = input_nodes
        for h in self.hidden_layer:
            # shape of matrices of weights
            self.shapes.append((row + 1, h))
            row = h

        # weight matrix of output layer
        self.shapes.append((row + 1, output_nodes))

    def _set_genes(self):
        """
        Determines number of genes present in chromosomes.
        Every weight is represented by a gene in chromosome.
        """
        # calculate number of genes
        genes = 0
        for t in self.shapes:
            genes += (t[0] * t[1])
        # total number of genes
        self.genes = genes

    def _set_chrom_len(self):
        """
        Determines length of chromosome.
        Length of chromosome = number of genes * length of genes
        """
        # length of chromosome
        self.chromosome_length = self.genes * self.gene_length

    def _set_population(self):
        """
        Determines random initial population.
        """
        # population size
        if self.population_size is None:
            self.population_size = self.chromosome_length
        # generate initial population
        self.population = np.random.randint(0, 10, size=(self.population_size, self.chromosome_length), dtype=int)

    # retrieve weights
    def _retrieve_weights(self, chromosome):
        """
        Determines weights of the Perceptron from the chromosome.
        :param chromosome: it is a numpy array of specified length and consists of genes
                           which represent weights of the MLP.
        :return: an array of the derived weights from the chromosome
        """

        weights = []
        # iterate over the chromosome for different genes
        for i in range(0, self.chromosome_length, self.gene_length):

            # first digit of the gene is used to determine the sign of the weight
            x = chromosome[i]
            # initialise weight
            weight = 0

            # evaluate weight
            for j in range(1, self.gene_length):
                weight = weight * 10 + chromosome[i + j]

            # assign a sign to the weight i.e. ["+", "-"]
            weight = go.sign_of_weight(x, weight)
            # standardise weights to capture the complexity of the hyperplane
            weight /= 10**(self.gene_length - 2)

            # append this weight to the list
            weights.append(weight)
        return weights

    # set up weight matrices
    def _set_weight_matrices(self, weights):
        """
        Generate weight matrices for layers

        :param weights: array of weights determined from a chromosome
        :return: an array of matrices of weights
        """

        # initialise array of matrices of weights
        matrices = []
        # count the instance of weight in weights array
        count = 0

        # iterate in the shapes array
        # shapes array contain dimensions of weight matrices
        # between the layers of MLP
        for shape in self.shapes:
            # dimension of matrix
            rows, columns = shape[0], shape[1]
            # initialise matrix
            mat = []

            # iterate to capture rows of matrix
            # weights are set in row by row manner
            for _ in range(rows):
                # initialise row
                row = []

                # iterate to capture columns
                # weights are added in left to right direction
                for __ in range(columns):

                    # add weight to row
                    row.append(weights[count])
                    # increment the weight index
                    count += 1

                # add row to matrix
                mat.append(row)
            # add matrix to matrices array
            matrices.append(np.array(mat))
        return matrices

    # feeding input to neural network
    def _feed_forward(self, matrices, x):
        """
        Evaluates output of the model network.
        1. Determines input at every layer by using dot product of weights
           between the layers to the output of the previous layer.
        2. Evaluates the activation function on the input variable.

        :param matrices: array of weights of different layers.
        :param x: training data {n_samples, n_features}

        :return: evaluated output of the MLP.
        """

        # output of the input layer
        # linear transformation is performed in the input layer
        layer_output = x

        # iterate over the weight matrices
        for i, mat in enumerate(matrices):

            # appending bias input to layer input data
            train = np.column_stack((np.ones(layer_output.shape[0]), layer_output))
            # calculate the layer input
            layer_input = np.dot(train, mat)
            # operate activation function on the layer input
            if i < self.h_layer:
                layer_output = self.activation(layer_input)
            else:
                layer_output = activations["logistic"](layer_input)

        # return output of last layer
        return layer_output

    # calculate fitness of chromosome
    def _fitness_function(self, error):
        """
        Return fitness value of a chromosome.
        Since, we have to minimize the error therefore, we evaluate fitness
        of a chromosome on the basis of the error obtained using the reciprocal
        of the error as the fitness value of the chromosome. The problem now,
        transforms to maximisation of the fitness value of a chromosome.
        This transformation of minimisation problem to maximisation problem,
        i.e. maximisation of fitness value is favourable according to the
        survival of the fittest theory of Prof. Charles Darwin.

        :param error: Evaluated error value on training,
                      using weights obtained from a chromosome.

        :return: Fitness value
        """

        return 1 / error

    # calculate fitness of population
    def _fitness_of_population(self, x, y):
        """
        Evaluates fitness value of each chromosome in the population.
        """

        # initialise array of fitness values
        fitness = []

        # iterate over the population
        for i, chromosome in enumerate(self.population):

            # retrieve weights
            weights = self._retrieve_weights(chromosome)
            # create weight matrices
            matrices = self._set_weight_matrices(weights)
            # feeding training data
            y_pred = self._feed_forward(matrices, x)
            # calculate error

            error = self.error(y_pred, y)
            # print("error: ", error)
            # calculate fitness
            f = self._fitness_function(error)
            fitness.append(f)
        self.fitness = np.array(fitness)

    # mating pool
    def _ga_operate(self):
        """
        Performs Genetic operators on the population.
        Order of operators:-
        1. Cross Over
        2. Mutation
        3. Inversion
        """

        # separating fittest individual from mating pool
        fittest = self.population[0].copy()
        mating_pool = np.delete(self.population, 0, axis=0)

        # cross over operator
        crossed_population = go.cross_over(mating_pool, self.cross_over_rate)

        # mutation operator
        # mutation is performed on the population determined after cross over
        new_population = go.mutation(crossed_population, self.mutation_rate)

        # check if there is any change in the population
        if (new_population == mating_pool).all():
            print("No change in population")

            # if, there is no change in the population after cross over and mutation
            # now we perform inversion operator
            new_population = go.inversion(mating_pool, self.inversion_rate)
            if (new_population == mating_pool).all():
                # if there is no change in the population
                # even after inversion
                # return True to increase mutation rate
                return True

        # recover population
        fittest.resize((1, self.chromosome_length))
        self.population = np.concatenate((fittest, new_population), axis=0)
        return False

    # labelling the output
    def _binarizer(self, x):
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x

    # check validity of hyper parameters
    def _validate_hyper_parameters(self):
        """
        Validate the hyper parameters provided.
        """
        if self.gene_length <= 0:
            raise ValueError("gene length must be > 0, but provided {}".format(self.gene_length))
        if self.gene_length > 6:
            warnings.warn("increasing gene length more than 6 "
                          "will increase overall computation time "
                          "and there will be no major effect on "
                          "overall result.", ComputationWarning)
        if self.gene_length > 10:
            raise ValueError("gene length must not exceed 10, but provided {}".format(self.gene_length))
        if self.population_size <= 0:
            raise ValueError("population size must be > 0, but provided {}".format(self.population_size))
        if np.any(np.array(self.hidden_layer) <= 0):
            negative_nodes = [x for x in self.hidden_layer if x <= 0]
            raise ValueError("number of nodes in hidden layers must be > 0, "
                             "but provided {}".format(negative_nodes))
        if (self.cross_over_rate <= 0) or (self.cross_over_rate > 1):
            raise ValueError("cross over rate has range (0, 1], but provided {}".format(self.cross_over_rate))
        if (self.mutation_rate <= 0) or (self.mutation_rate > 1):
            raise ValueError("mutation rate has range (0, 1], but provided {}".format(self.cross_over_rate))
        if (self.inversion_rate <= 0) or (self.inversion_rate > 1):
            raise ValueError("inversion rate has range (0, 1], but provided {}".format(self.cross_over_rate))
        if self.max_generation <= 0:
            raise ValueError("number of maximum generations must be > 0, but provided {}".format(self.cross_over_rate))

    # separate fit genes and generate new generation
    def _gen(self):
        self.fit_gen.append((self.fitness[0], self.population[0]))
        self._set_population()

    # finding convergence
    def _converge(self):
        mean = np.mean(self.fitness)
        max_fit = np.max(self.fitness)
        std = np.std(self.fitness)
        # measuring convergence on the basis of dispersion of the fitness of population
        if max_fit >= 25:
            print("max: ", max_fit)
            print("std: ", std)
            print("mean: ", mean)
            return True
        elif std <= 1 and max_fit < 20:
            self.inversion_rate += 0.009
            if self.inversion_rate >= 1:
                self.inversion_rate -= 0.09
            # reset new generation
            # print("New Generation")
            # self._gen()
            return False
        else:
            if self.inversion_rate > 0.001:
                self.inversion_rate -= 0.009
            return False

    # optimized model
    def _optimum_model(self):
        chromosome = self.population[0]
        if len(self.fit_gen) != 0:
            # sort fittest gen
            self.fit_gen.sort(key=lambda x: x[0], reverse=True)
            # fittest chromosome
            chromosome = np.array(self.fit_gen[0][1])
        # retrieve weights
        weights = self._retrieve_weights(chromosome)
        # create weight matrices
        self.optimum_matrices = self._set_weight_matrices(weights)

    # initialise parameters for the model
    def _init_population(self, x, y):
        # initialize model
        self._initialize()
        # determine shape of matrices
        self._set_matrices_shape(x, y)
        # determine number of genes i.e. number of weights
        self._set_genes()
        # determine chromosome length
        self._set_chrom_len()
        # generate population
        self._set_population()
        # check validity of hyper parameters
        self._validate_hyper_parameters()

    # for training
    def _fit(self, train, target):
        # initialise population
        self._init_population(train, target)
        # find convergence
        i = 0
        # flag for change in population
        flag = True
        while i < self.max_generation:
            # calculate fitness of population
            self._fitness_of_population(train, target)
            # check convergence
            converged = self._converge()
            if converged:
                break
            # select chromosomes for mating pool
            self.fitness, self.population = go.selection(self.fitness, self.population)
            # crossing over on selected population
            check = self._ga_operate()
            if check and flag:
                self.mutation_rate += 0.09
                flag = False
            elif not flag:
                self.mutation_rate -= 0.09
                flag = True
            i += 1
        self._optimum_model()

    def fit(self, x, y):
        return self._fit(x, y)

    # predicting for new data
    def predict(self, test):
        y_pred = self._feed_forward(self.optimum_matrices, test)
        return self._binarizer(y_pred)


class GBMLPClassifier(GBPMultiLayerPerceptron):
    def __init__(self, gene_length=5, population_size=None, hidden_layer=(100,),
                 cross_over_rate=1.0, mutation_rate=0.01, inversion_rate=0.001,
                 max_generation=1000, activation="relu", error="log_loss"):
        sup = super(GBMLPClassifier, self)
        sup.__init__(gene_length=gene_length, population_size=population_size, hidden_layer=hidden_layer,
                     cross_over_rate=cross_over_rate, mutation_rate=mutation_rate,
                     inversion_rate=inversion_rate, max_generation=max_generation,
                     activation=activation, error=error)

    def fit(self, x, y):
        return self._fit(x, y)


# if __name__ == "__main__":
#     # training and target data
#     train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     target = np.array([[0], [1], [1], [0]])
#
#     # creating the model
#     nn = GBMLPClassifier(hidden_layer=(2,))
#
#     # training the model
#     nn.fit(train, target)
#     print("mean   : ", np.mean(nn.fitness))
#     print("std: ", np.std(nn.fitness))
#
#     # predicting the same for training data
#     y_pred = nn.predict(train)
#     print(y_pred)
