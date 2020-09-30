# -*- coding: utf-8 -*-
"""
main.py clase con la configurtación optima de 
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
global vehiculos_esp #Array con los vehiculos especiales
global max_viajes #Número de viajes máximos
global bonus #Bonus por adelanto
global steps #Número máximo de iteraciones en la ejecución
global viajes  #Distinos viajes a realizar


def cargar(filename):
    global filas #Numero de filas
    global columnas #Números de columnas
    global vehiculos #Número de vehiculos
    global vehiculos_esp #Array con los vehiculos especiales
    global max_viajes #Número de viajes máximos
    global bonus #Bonus por adelanto
    global steps #Número máximo de iteraciones en la ejecución
    global viajes  #Distinos viajes a realizar
    viajes=[]
    vehiculos_esp=[]
    with open(filename, 'r') as fin:
        line = fin.readline() 
        filas, columnas, vehiculos, max_viajes, bonus, steps = [
            int(num) for num in line.split()]   
        for i in range(max_viajes):
            line = fin.readline()
            a,b, x,y , s, f , esp= [int(num) for num in line.split()]    
            viajes.append([a,b,x,y,s,f,(esp==1)])
        line=fin.readline()
        [vehiculos_esp.append(num=='1') for num in line.split()]
    fin.close()


def eval(individual):
    rides=[]
    value=0
    val=0
    no_adap=0
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
            if i[6]==True and vehiculos_esp[x]==False:
                no_adap+=1
            time+=abs(i[0]-pos[0])+abs(i[1]-pos[1])
            if time < i[4]:
                value+=bonus
                time=i[4]
            recorrido=abs(i[2]-pos[0])+abs(i[3]-pos[1])
            pos=[i[2],i[3]]
            time+=recorrido
            if time <= i[5]:
                value+=recorrido
    return (int(value),no_adap)

            

def configuracionAlgoritmo(toolbox):  
    # Se seleccionan procedimiento standard para cruce, mutacion y seleccio
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.emo.selSPEA2)
	# Se define cómo se evaluará cada individuo
	# En este caso, se hará uso de la función de evaluación que se ha definido en el modulo Evaluacion.py
    toolbox.register("evaluate", eval)


#%% Se define como se realiza la Evolución de la busqueda de la solución
def realizaEvolucion(toolbox, stats):

    # Se configura cómo se define cada individuo. Ver fichero correspondiente
    configuraPoblacion(toolbox)

    configuracionAlgoritmo(toolbox)

    # Se inicializa la poblacion con 300 individuos
    population = toolbox.population(n=300)
    # Se llama al algoritmo que permite la evolucion de las soluciones
    population, logbook = algorithms.eaSimple(population, toolbox, 
	                               cxpb=0.5, mutpb=0.2, # Probabilidades de cruce y mutacion
	                               ngen=30, verbose=False, stats=stats) # Numero de generaciones a completar y estadisticas a recoger

    # Por cada generación, la estructura de logbook va almacenando un resumen de los
    # avances del algoritmo.
#    print("El resultado de la evolución es: ")
#    print(logbook)
#
#    # Comprobamos cual es la mejor solucion encontrada por evolucion
#    print("La mejor solucion encontrada es: ")
#    print(tools.selBest(population,1)[0])
#    print("Con un fitness de "+str(EvaluacionSolucion.eval(tools.selBest(population,1)[0])[0]))
    
    return logbook, population

def configuraEstadisticasEvolucion():
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size,)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats
    
    #%% Visualizamos una estadística para comprobar como fue la evolucion
def visualizaGrafica(log, pop):
    # Se recuperan los datos desde el log
    gen = log.select("gen")
    mins = log.chapters["fitness"].select("min")
    maxs = log.chapters["fitness"].select("max")
    avgs = log.chapters["fitness"].select("avg")
    size_avgs = log.chapters["size"].select("avg")
    size_mins = log.chapters["size"].select("min")
    size_maxs = log.chapters["size"].select("max")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, avgs, "g-", label="Average Fitness")  
    line2 = ax1.plot(gen, maxs, "r-", label="Max Fitness") 
    line3 = ax1.plot(gen, mins, "b-", label="Min Fitness") 
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    
    fig2, ax2 = plt.subplots()
    line4 = ax2.plot(gen, size_mins, "c-", label="Min Adap")
    line5 = ax2.plot(gen, size_avgs, "m-", label="Average Adap")  
    line6 = ax2.plot(gen, size_maxs, "y-", label="Max Adap") 
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("No Adap", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")
#    lns = line1 + line2 + line3
#    labs = [l.get_label() for l in lns]
#    ax1.legend(lns, labs, loc="center right")
#    lns = line4+ line5+line6
#    labs = [l.get_label() for l in lns]
#    ax2.legend(lns, labs, loc="center right")
    fitness=[]
    adap=[]
    paretos=tools.ParetoFront()
    paretos.update(pop)
    for i in paretos:
        fitness.append(i.fitness.values[0])
        adap.append(i.fitness.values[1])     
    fig3, ax3=plt.subplots()
    ax3.plot(fitness, adap, "bx", label = "Frente de Pareto")   
    fitness=[]
    adap=[]
    for i in pop:
        if not(i in paretos):
            fitness.append(i.fitness.values[0])
            adap.append(i.fitness.values[1])
    ax3.plot(fitness, adap, "ro", label = "Población")    

    
    ax3.set_xlabel("Fitness")
    ax3.set_ylabel("No adap-", color="r")
    ax3.set_title("Frente de Pareto")
    plt.show()

#def visualizaPareto():
#    figura=plt.plot(Paret)

def configuraPoblacion(toolbox):

	''' Se configura el fitness que se va a emplear en los individuos
	 En este caso se configura para:
	 1.buscar un único objetivo: es una tupla de solo un numero
	 2.maximizar ese objetivo (se multiplica por un num. positivo)'''
	creator.create("FitnessMax", base.Fitness, weights=(1.0,-1.0))

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
#    archivos=["./a_example_sp.in","./b_should_be_easy_sp.in","./c_no_hurry_sp.in","./d_metropolis_sp.in"] #El primer archivo no suele aportar datos valiosos, pero si se quiere añadir copiarlo a archivos
    archivos=["./b_should_be_easy_sp.in","./c_no_hurry_sp.in","./d_metropolis_sp.in"]  
    for i in archivos:
        cargar(i)
        toolbox = base.Toolbox()
        stats = configuraEstadisticasEvolucion()
        log, pop = realizaEvolucion(toolbox, stats)
        visualizaGrafica(log, pop)
#        break
#    Desactivar ls siguientes líneas si se desea introducir por argumento y comentar las anteriores.
#    cargar(sys.argv[1])
#    toolbox = base.Toolbox()
#    stats = configuraEstadisticasEvolucion()
#    log = realizaEvolucion(toolbox, stats)
#    visualizaGrafica(log)