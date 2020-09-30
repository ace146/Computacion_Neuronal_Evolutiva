# -*- coding: utf-8 -*-
"""
main.py clase Práctica6 con Beer_sales para hacer predicción
@author: Vadim Budagov y Willow Maui Garcia
"""
import operator

import math

import random

import numpy

from deap import algorithms

from deap import base

from deap import creator

from deap import tools

from deap import gp


import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout as pgv


global weeks
global date
global price12PK 
global price12PK_LN
global PRICE12PK
global PRICE12PK_LN
# in this function the file is processed column by column.
def getFile(filename):
    global weeks
    global date
    global price12PK
    global price12PK_LN 
    global PRICE12PK
    global PRICE12PK_LN
    weeks = []
    date = []
    price12PK = []
    price12PK_LN = []
    with open(filename, 'r') as datei:
        lines = datei.readlines()

    for line in lines:
        columnas = [element for element in line.strip().split(';')]
        weeks.append(columnas[0])
        date.append(columnas[1])
        price12PK.append(columnas[2])
        price12PK_LN.append(columnas[3])

    PRICE12PK = numpy.asanyarray(price12PK)
    PRICE12PK_LN = numpy.asarray(price12PK_LN)
    PRICE12PK = numpy.delete(PRICE12PK, 0)
    PRICE12PK_LN = numpy.delete(PRICE12PK_LN, 0)
    print("Price12PK:", price12PK)
    print("Price12PK_LN:", price12PK_LN)

# Define new functions
def protectedDiv(left, right):

    try:
        return left / right

    except ZeroDivisionError:

        return 1


# Primitive operators mapped from library operator
pset = gp.PrimitiveSet("MAIN", 1) # number of inputs, hier we have 1-dim problem with only one input

pset.addPrimitive(operator.add, 2)# 2 is a number of arity, number of entries

pset.addPrimitive(operator.sub, 2)

pset.addPrimitive(operator.mul, 2)

pset.addPrimitive(protectedDiv, 2)

pset.addPrimitive(operator.neg, 1)

pset.addPrimitive(math.cos, 1)

pset.addPrimitive(math.sin, 1)

#pset.addEphemeralConstant("rand100", lambda: random.randint(-1,1)) # appends a constant ephemeral to a tree.

pset.renameArguments(ARG0='x')


# Create two objects: Individuals containing the genotype and a fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


# we want to register some parameters specific to the evolution process
toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)


# receive an individual as input, and return the corresponding fitness.
# @individual genotype?
# @points phenotype?
# @return MSE Mean Squared Error
def evalSymbReg(individual, points):

    # Transform the tree expression in a callable function

    func = toolbox.compile(expr=individual)

    # Evaluate the mean squared error between the expression

    # and the real function : x**4 + x**3 + x**2 + x

    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)

    return math.fsum(sqerrors) / len(points),

newPrice12PK = numpy.fromstring(PRICE12PK, dtype=int, sep=',')
newPrice12PK_LN = numpy.fromstring(PRICE12PK_LN, dtype=int, sep=',')

toolbox.register("evaluate", evalSymbReg, points=newPrice12PK) # get new points to evaluation

toolbox.register("select", tools.selTournament, tournsize=3) #select function

toolbox.register("mate", gp.cxOnePoint)# crossover

toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # mutation



toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))



def getGrapf():
    
    expr = toolbox.individual()
    nodes, edges, labels = gp.graph(expr)
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = pgv(g, prog="dot")
    
    nx.draw(g, pos)
    nx.draw(g, pos)
    nx.draw(g, pos, labels)
    plt.show()
    
def configureEvolutionaryStats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    stats_size = tools.Statistics(len)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    mstats.register("avg", numpy.mean)

    mstats.register("std", numpy.std)

    mstats.register("min", numpy.min)

    mstats.register("max", numpy.max)
    
    return mstats

def main():

    #random.seed(100)
    getFile('Beer_sales.csv')
    #getGrapf() 

    pop = toolbox.population(n=300) # number of population
    
    # The hall of fame is a specific structure which contains the n best individuals (here, the best one only).
    hof = tools.HallOfFame(1)

    mstats = configureEvolutionaryStats()
    
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,

                                   halloffame=hof, verbose=True)

    # print log

    return pop, log, hof



if __name__ == "__main__":

    main()