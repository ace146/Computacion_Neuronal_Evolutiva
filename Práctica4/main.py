# -*- coding: utf-8 -*-
"""
main.py clase con toda la configuración y ejecución del algoritmo en una sola clase. 
@author: Willow Maui Garcia Y Vadim Budagov
"""


import random
import matplotlib.pyplot as plt
from deap import base, creator
from deap import tools
from deap import algorithms
import sys
import numpy as np 
global filas #Numero de filas
global columnas #Números de columnas
global vehiculos #Número de vehiculos
global max_viajes #Número de viajes máximos
global bonus #Bonus por adelanto
global steps #Número máximo de iteraciones en la ejecución
global viajes  #Distinos viajes a realizar


def cargar(filename):
    global filas #Numero de filas
    global columnas #Números de columnas
    global vehiculos #Número de vehiculos
    global max_viajes #Número de viajes máximos
    global bonus #Bonus por adelanto
    global steps #Número máximo de iteraciones en la ejecución
    global viajes  #Distinos viajes a realizar
    viajes=[]
    with open(filename, 'r') as fin:
        line = fin.readline() 
        filas, columnas, vehiculos, max_viajes, bonus, steps = [
            int(num) for num in line.split()]
    
        for i in range(max_viajes):
            line = fin.readline()
            a,b, x,y , s, f = [int(num) for num in line.split()]
            viajes.append([a,b,x,y,s,f])

def eval(individual):
    rides=[]
    value=0
    val=0
    for x in range(vehiculos):
        rides.append([])
    for i in individual:
        rides[i].append(viajes[val])
        val+=1
    for x in range(vehiculos):
        rides[x].sort(key = lambda x:x[4]) 
        pos=[0,0]
        time=0
        for i in rides[x]:
            time+=abs(i[0]-pos[0])+abs(i[1]-pos[1])
            if time < i[4]:
                time=i[4]
            recorrido=abs(i[2]-pos[0])+abs(i[3]-pos[1])
            pos=[i[2],i[3]]
            time+=recorrido
            if time <= i[5]:
                value+=recorrido
                bono=i[5]-time
                if bono> 0:
                    value+= bono*bonus 
    return (int(value),)

def configuracionAlgoritmo(toolbox):  
    # Se seleccionan procedimiento standard para cruce, mutacion y seleccio
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
	# Se define cómo se evaluará cada individuo
	# En este caso, se hará uso de la función de evaluación que se ha definido en el modulo Evaluacion.py
    toolbox.register("evaluate", eval)

#%% Se define como se realiza la Evolución de la busqueda de la solución
def realizaEvolucion(toolbox, stats):

    # Se configura cómo se define cada individuo. Ver fichero correspondiente
    configuraPoblacion(toolbox)

    configuracionAlgoritmo(toolbox)

    # Se inicializa la poblacion con 300 individuos
    population = toolbox.population(n=200)
    # Se llama al algoritmo que permite la evolucion de las soluciones
    population, logbook = algorithms.eaSimple(population, toolbox, 
	                               cxpb=0.5, mutpb=0.2, # Probabilidades de cruce y mutacion
	                               ngen=60, verbose=False, stats=stats) # Numero de generaciones a completar y estadisticas a recoger

    # Por cada generación, la estructura de logbook va almacenando un resumen de los
    # avances del algoritmo.
#    print("El resultado de la evolución es: ")
#    print(logbook)
#
#    # Comprobamos cual es la mejor solucion encontrada por evolucion
#    print("La mejor solucion encontrada es: ")
#    print(tools.selBest(population,1)[0])
#    print("Con un fitness de "+str(EvaluacionSolucion.eval(tools.selBest(population,1)[0])[0]))
    
    return logbook

def configuraEstadisticasEvolucion():

    # Se configura que estadísticas se quieren analizar sobre la evolucion
    stats = tools.Statistics(lambda ind: ind.fitness.values) 
    stats.register("avg", np.mean) 
    stats.register("std", np.std) 
    stats.register("min", np.min) 
    stats.register("max", np.max) 
    
    return stats
    
    #%% Visualizamos una estadística para comprobar como fue la evolucion
def visualizaGrafica(log):
    # Se recuperan los datos desde el log
    gen = log.select("gen")
    avgs = log.select("avg")
    maxs = log.select("max")
    mins = log.select("min")
    
    # Se establece una figura para dibujar
    fig, ax1 = plt.subplots()
    
    # Se representa la media del valor de fitness por cada generación
    line1 = ax1.plot(gen, avgs, "b-", label="Average Fitness")  
    line2 = ax1.plot(gen, maxs, "r-", label="Max Fitness") 
    line3 = ax1.plot(gen, mins, "g-", label="Min Fitness") 
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    
    ''' Notad que se deberían representar mas cosas. Por ejemplo el maximo y el minimo de
    ese fitness se están recogiendo en las estadisticas, aunque en el ejemplo no se
    representen '''
    
    plt.plot()
    
def configuraPoblacion(toolbox):

	''' Se configura el fitness que se va a emplear en los individuos
	 En este caso se configura para:
	 1.buscar un único objetivo: es una tupla de solo un numero
	 2.maximizar ese objetivo (se multiplica por un num. positivo)'''
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))

	''' Se configura el individuo para que utilice la descripción anterior
	de fitness dentro de los individuos '''
	creator.create("Individual", list, fitness=creator.FitnessMax)

	''' Ejemplo de genotipo cuyos genes son de tipo float '''
	#toolbox.register("attribute", random.random)
	''' Ejemplo de Genotipo cuyos genes son de tipo booleano '''
	toolbox.register("attribute", random.randint, 0, vehiculos-1) #En realidad, se indica que serán entereos entre 0 y 1
	''' El individuo se crea como una lista (o repeticion) de "attribute", definido justo antes
	Tendrá una longitud de 5 atributos '''
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len(viajes))
	''' La población se crea como una lista de "individual", definido justo antes'''
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
if __name__ == "__main__":
    archivos=["./data/a_example.in","./data/b_should_be_easy.in","./data/c_no_hurry.in","./data/d_metropolis.in","./data/e_high_bonus.in"]    
 #   archivos=["./data/d_metropolis.in"]    
    for i in archivos:
        print(i)
        cargar(i)
        toolbox = base.Toolbox()
        stats = configuraEstadisticasEvolucion()
        log = realizaEvolucion(toolbox, stats)
        visualizaGrafica(log)
#    Desactivar ls siguientes líneas si se desea introducir por argumento y comentar las anteriores.
#    cargar(sys.argv[1])
#    toolbox = base.Toolbox()
#    stats = configuraEstadisticasEvolucion()
#    log = realizaEvolucion(toolbox, stats)
#    visualizaGrafica(log)