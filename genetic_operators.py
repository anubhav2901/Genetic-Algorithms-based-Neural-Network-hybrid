# implementation of genetic operators
import numpy as np
import operator as op


def sign_of_weight(x, weight):
    """
    Determines the sign of a weight, based on the value of the first digit of the gene.

    :param x: int, range [0, 9]
                First digit of the gene.
    :param weight: int
                Weight determined from the gene.

    :return: int
            Signed weight.
    """
    if x >= 5:
        return weight
    else:
        return -weight


def sort_population(fitness, population):
    """
    Arranges population in decreasing order of fitness values of chromosomes.

    :param fitness: {array}, shape {population_size,}
                    Fitness value associated with each chromosome.
    :param population: {array}, shape {population_size, chromosome_length}
                    Collection of chromosomes in the population.

    :return: Sorted arrays of fitness values and population.
            fitness: {array}, shape{population_size,}
                    Arranged in descending order.
            population: {array}, shape{population_size, chromosome_length}
                    Arranged according to the order of fitness values of each chromosomes.
    """

    # add fitness values, as a column to the chromosomes
    fitness_population = np.column_stack((fitness, population))
    # sort the whole combination in descending order
    fitness_population = fitness_population[fitness_population[:, 0].argsort()[::-1]]
    # separate fitness values and chromosomes
    fitness, population = np.hsplit(fitness_population, [1])

    # return ordered fitness values and chromosomes
    return fitness, population


def selection(fitness, population):
    """
    Perform selection operation to determine mating pool.
    Based on survival of the fittest theory.
    Least fit individuals are not considered for mating pool.
    Fittest individuals are duplicated in position of the least fit,
    to maintain constant population size.
    """

    # sorting chromosomes on the basis of fitness
    fitness, population = sort_population(fitness, population)
    # determine least fitness value
    min_fit = np.min(fitness)

    # iterate to replace least fit chromosomes
    for i, p in enumerate(population):
        if fitness[i] > min_fit:
            continue
        # replacing by fittest chromosome
        fitness[i] = fitness[0]
        population[i] = population[0]

    # return mating pool, consisting of selected chromosomes
    return fitness, population


# cross over operator
def cross_over(population, cross_over_rate=1.0):
    """
    Performs cross over genetic operation on the population.

    Algorithm:-
    1. Separate population for crossing using cross over rate provided.
    2. Randomly pair up chromosomes.
    3. Repeat for a pair.
        For a pair randomly pick two sites.
        Swap the sliced portions of the chromosomes.
    4. Concatenate the crossed population to the remaining population.

    :param population: array, shape{population_size, chromosome_length}
                        Collection of chromosomes in the population.
    :param cross_over_rate: double, optional, default 1, range (0, 1]
                        Represents proportion of population to be used
                        for cross over.

    :return: array, shape{population_size, chromosome_length}
                        New population having crossed chromosomes.
    """

    # determine proportion of population to be used for cross over.
    # population size and chromosome length.
    population_size, chromosome_length = population.shape

    # index for splitting population.
    cross_split = int((1.0 - cross_over_rate) * population_size)

    # split population
    cross_population = np.array(population[cross_split:])
    rest_population = np.array(population[:cross_split])

    # update population size for cross over
    population_size = cross_population.shape[0]

    # if population size is odd then, eliminate a chromosome from population.
    # marker for odd numbered population.
    odd_chromosome = None
    if population_size % 2 != 0:
        index = np.random.randint(0, population_size)
        odd_chromosome = cross_population[index].copy()
        population = np.delete(cross_population, index, axis=0)
        population_size -= 1

    # pair up chromosomes
    pairs = np.random.choice(a=population_size, size=(int(population_size / 2), 2), replace=False)

    # initialise new population
    crossed_population = []
    # iterate over the pair
    for p in pairs:
        # determine chromosomes
        chr1, chr2 = population[p[0]], population[p[1]]

        # determine crossing sites
        site = np.random.choice(chromosome_length, size=2, replace=True)
        site.sort()

        # cross over
        chr1[site[0]:site[1] + 1], chr2[site[0]:site[1] + 1] = chr2[site[0]:site[1] + 1], chr1[site[0]:site[1] + 1].copy()

        # add crossed chromosomes to new population
        crossed_population.append(chr1)
        crossed_population.append(chr2)

    # add odd chromosome if, present.
    if odd_chromosome is not None:
        crossed_population.append(odd_chromosome)

    # return new population
    return np.concatenate((rest_population, crossed_population), axis=0)


# mutation operator
def mutation(population, mutation_rate=0.01):
    """
    Performs mutation genetic operation on the population.

       Algorithm:-
       1. Separate population for mutation using mutation rate provided.
       2. Randomly pair up chromosomes.
       3. Repeat for a pair.
            For a pair randomly pick two sites.
            Repeat for digit in sites.
                Randomly pick two operators.
                Operate these operator on the digits of both chromosomes.
       4. Concatenate the mutated population to the remaining population.

    :param population: array, shape{population_size, chromosome_length}
                           Collection of chromosomes in the population.
    :param mutation_rate: double, optional, default 0.01, range (0, 1]
                           Represents proportion of population to be used for mutation.

    :return: array, shape{population_size, chromosome_length}
                           New population having mutated chromosomes.
    """

    # separate proportion of population for mutation
    # population size and chromosome length
    population_size, chromosome_length = population.shape

    # select individuals on the basis of probability of mutation
    # generate mutation probability
    p_m = np.random.rand(population_size)
    # set probability cut off for mutation
    p_m_limit = 1 - mutation_rate

    # split population
    mut_population = population[p_m > p_m_limit].astype(int)
    rest_population = population[p_m <= p_m_limit]

    # number of chromosomes selected for mutation
    mut_population_size = len(mut_population)

    if mut_population_size > 1:
        print("mutation")
        # if population size is odd then, eliminate a chromosome from population.
        # marker for odd numbered population.
        odd_chromosome = None
        if mut_population_size % 2 != 0:
            index = int(np.random.randint(0, mut_population_size))
            odd_chromosome = mut_population[index]
            mut_population = np.delete(mut_population, index, axis=0)
            mut_population_size -= 1

        # operators to be used for mutation
        operators = {"&": op.and_, "|": op.or_, "^": op.xor, "~": lambda x: 9 - x}

        # generate pairs of chromosomes for mutation
        pairs = np.random.choice(mut_population_size, size=(int(mut_population_size / 2), 2), replace=False)

        for p in pairs:
            # determine sites
            sites = np.random.choice(chromosome_length, size=2, replace=False)
            sites.sort()
            # mutation
            for s in range(sites[0], sites[1]):
                # determine operators
                operator = np.random.choice(list(operators.keys()), size=2)
                for i, o in enumerate(operator):
                    if o == "~":
                        mut_population[p[i]][s] = operators[o](mut_population[p[i]][s])
                    else:
                        mut_population[p[i]][s] = operators[o](mut_population[p[0]][s], mut_population[p[1]][s]) % 10

        # add odd chromosome if, any.
        if odd_chromosome is not None:
            odd_chromosome.resize((1, chromosome_length))
            mut_population = np.concatenate((mut_population, odd_chromosome), axis=0)

    # return new population
    return np.concatenate((rest_population, mut_population), axis=0)


def inversion(population, inversion_rate=0.001):
    """
    Performs inversion genetic operation on the population.

    Algorithm:-
    1. Separate population for inversion using inversion rate provided.
    2. Repeat for chromosomes.
        Select two inversion sites.
        Repeat for digits in sites.
            Take complement of the digit.
    4. Concatenate the inverted population to the remaining population.

    :param population: array, shape{population_size, chromosome_length}
                        Collection of chromosomes in the population.
    :param inversion_rate: double, optional, default 0.001, range (0, 1]
                        Represents proportion of population to be used for inversion.

    :return: array, shape{population_size, chromosome_length}
                        New population having inverted chromosomes.
    """

    # determine proportion of population for inversion
    # population size and chromosome length
    population_size, chromosome_length = population.shape

    # select individuals on the basis of probability of inversion
    # generate inversion probability
    p_inv = np.random.rand(population_size)
    # set probability cut off for inversion
    p_inv_limit = 1 - inversion_rate

    # split population
    inv_population = population[p_inv > p_inv_limit].astype(int)
    rest_population = population[p_inv <= p_inv_limit]

    # return if, no chromosome is selected for inversion
    if len(inv_population) == 0:
        return population

    print("inversion")
    # iterate over the population
    for chromosome in inv_population:
        # determine sites
        sites = np.random.choice(chromosome_length, size=2)
        sites.sort()

        # inversion
        for s in range(sites[0], sites[1] + 1):
            chromosome[s] = 9 - chromosome[s]

    # return new population
    return np.concatenate((rest_population, inv_population), axis=0)

# a = np.array([1, 9, 8])
# b = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
# b = inversion(b, 0.1)
# print(a>5)
# print(b)