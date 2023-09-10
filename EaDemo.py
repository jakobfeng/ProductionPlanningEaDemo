
'''
Author: Jakob Engebretsen
Date: September 2023
Version 1.0
Description: Script for running a simplified evolutionary algorithm for a fictional hydro power scenario
Input Variables: (all input variables are declared in the __main__ method)
    - Date to run for
    - Number of generations
    - Minimum production (constant)
    - Maximum production (constant)
    - Optimum production (constant)
    - Reservoir level starting point
    - 1-dimensional electricity price forecast per hour
Output:
    - MWh hydro production per hour for the given day
    - Production plots for the starting population and the optimal solution per generation
'''
# Import libraries
import pandas as pd
import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

'''
Description: Method for getting the price per hour forecast for the given day
'''
def getPricePerHour():
    filePath = 'priceForecasts\\' + dateToRunFor + '.xlsx'
    price_df = pd.read_excel(filePath);
    return price_df

'''
Description: Method for creating the starting population based on production constants (minimum, maximum, otpimal) and a randomly generated individual
'''
def getStartPopulation():
    startPopulation = [] # Return variable. List of individuals

    # Add a production plan per constant value to starting population
    for productionConstant in [minProduction, maxProduction, optimalProduction]:
        df = pd.DataFrame(index=range(24), columns=['Production'])
        df['Production'] = productionConstant
        startPopulation.append(df)

    # Create a random production plan
    random_df = pd.DataFrame(index=range(24), columns=['Production'])
    random_df['Production'] = np.random.randint(minProduction, maxProduction, random_df.shape[0])
    startPopulation.append(random_df) # Add random production plan to starting population

    return startPopulation

'''
Description: Method for plotting a production plan solution and save plot to folder
'''
def visualizeSolution(graphTitle, solution):
    plt.figure(figsize=(10,6))
    plt.plot(solution.loc[0:23, 'Production'], '-', color='#006ab3', markersize=10)
    plt.title(graphTitle)
    plt.xlabel('Hour of Day ({})'.format(dt.datetime.strptime(dateToRunFor, '%d%m%Y').date()))
    plt.ylabel('Production [MWh]')
    ax = plt.gca()
    ax.set_ylim([minProduction-10, maxProduction+10])
    plt.savefig('solutionPlots\{}.png'.format(graphTitle.replace(' ', '')), bbox_inches='tight')
    plt.close()

'''
Description: Method for creating a probaility matrix used as the probability distribution when selecting parents
'''
def getProbabilityDistribution(populationSize):
    probabilityDistribution = pd.DataFrame(index=range(populationSize), columns=['Probability'])
    probabilityDistribution['Population'] = probabilityDistribution.shape[0] * 2
    probabilityDistribution['Probability Count'] = probabilityDistribution['Population'] - probabilityDistribution.index
    probabilityDistribution['Probability'] = probabilityDistribution['Probability Count'] / probabilityDistribution['Probability Count'].sum()
    return probabilityDistribution[['Probability']]

'''
Description: Method for randomly picking parents to offspring creation. Randomness based on fitness probability distribution
Output: Two parents as pandas Series objects
'''
def getParents(population, probabilityDistribution):
    # Extract parent 1 (pick random index value with some probability distribution)
    parent1Index = np.random.choice(probabilityDistribution.index, p=probabilityDistribution['Probability'])
    # Extract parent 2
    parent2Index = parent1Index
    while parent2Index == parent1Index:
        # Extract parent 2 (pick random index value with some probability distribution. Cannot be the same as index of parent 1)
        parent2Index = np.random.choice(probabilityDistribution.index, p=probabilityDistribution['Probability'])

    parent1 = population[parent1Index].loc[0:23]['Production'] # First parent. Pandas Series Type (no columns)
    parent2 = population[parent2Index].loc[0:23]['Production'] # Second parent. Pandas Series Type (no columns)
    return parent1, parent2

'''
Description: Method for getting a gene value based on two parents and the global gene picker probability distribution
'''
def getGeneValueFromParents(parent1, parent2, geneIndex):
    geneValue = None; # Return value
    geneRandomIndexPicker = np.random.choice(range(len(genePickerProbability)), p=genePickerProbability)

    if geneRandomIndexPicker == 0: # Meaning: Pick gene from Parent 1
        geneValue = parent1.loc[geneIndex]
    elif geneRandomIndexPicker == 1: # Meaning: Pick gene from Parent 2
        geneValue = parent2.loc[geneIndex]
    elif geneRandomIndexPicker == 2: # Meaning: Make a new gene as the mean of Parent 1 and Parent 2
        geneValue= int((parent1.loc[geneIndex] + parent2.loc[geneIndex]) / 2)
    else: # Meaning: Mutate a new gene
        geneValue = np.random.randint(minProduction, maxProduction)

    return geneValue

'''
Description: Method for identifying and returning the individual with highest fitness from population 
'''
def getFittestIndividualFromPopulation(population):
    bestFitness = -100000
    fittestIndividual = None
    # Iterate population
    for individual in population:
        individualFitness = individual.loc[24, 'Fitness'] # Fitness value in stored in the 24th row of the 'Fitness' columns
        if individualFitness > bestFitness: # Compare fitness to the current highest population fitness
            fittestIndividual = individual # Update variable for fittest individual (return value)
            bestFitness = individualFitness # Update variable for highest fitnes
    return fittestIndividual

'''
Description: Evaluate the fitness of individuals that have recently been added to population. Sort individuals in the population based on fitness in descending order. 
'''
def evaluatePopulation(population, price_df):
    # Ensure all individuals are evaluated
    for individual in population:
        if not 'Fitness' in individual.columns.values: # only evaluate individuals not already evaluated
            evaluateIndividual(individual, price_df)

    # only let the fittest individuals survive to next generation. Sort the population based on fitness in descending order
    if len(population) > populationMaxSize:
        populationForNextGeneration = [] # New list of individuals to represent next generation of individuals
        evaluationScheeme = pd.DataFrame(index=range(len(population)), columns=['Fitness']) # Dataframe to compare fitness of current generations individuals
        
        # Iterate indidivuals and add their value fo the evaluation scheeme
        for index in range(len(population)):
            individual = population[index]
            evaluationScheeme.loc[index, 'Fitness'] = individual.loc[24, 'Fitness']
        evaluationScheeme.sort_values(by='Fitness', inplace = True, ascending=False) # Sort the evaulation scheeme descending

        # Add only the fittest individuals from the population into next generation
        for i in range(populationMaxSize):
            individualIndexValue = evaluationScheeme.iloc[i].name # Current population index value of a surviving individual
            populationForNextGeneration.append(population[individualIndexValue]) # Add the individual to the population of next generation (survival)

    # Population has not reached its maximum limit yet. All individuals survive to next generation
    else:
        populationForNextGeneration = population

    return populationForNextGeneration

'''
Description: Helping method for evaluatePopulation(). Calculate the fitness of an individual production plan. No return value, instead inplace calculation
'''
def evaluateIndividual(individual, price_df):
    # Calculate revenue
    individual['Revenue'] = individual['Production'] * price_df['Price [€/MWh]']
    revenue = individual['Revenue'].sum()

    # Calculate the deviation factor based on distance from optimum production level
    individual['Optimal Deviation Factor'] = 1 + (abs(individual['Production'] - optimalProduction)/100)**2

    # Calculate water consumption
    individual['Consumption'] = individual['Production'] * (individual['Optimal Deviation Factor'])

    # Calculate value of remaining water
    remainingWater = startReservoirLevel - individual['Consumption'].sum()
    valueOfRemainingWater = remainingWater * waterValue

    # Calculate fitness
    fitness = round(revenue + valueOfRemainingWater - startReservoirValue, 1)

    # Store fitness in the dataframe as the 24th index of 'Fitness' column
    individual['Fitness'] = np.nan
    individual.loc[24, 'Fitness'] = fitness

'''
Description: Helping method for runEa(). Display information to end-user based on input parameters
'''
def displayInformation():
    date = dt.datetime.strptime(dateToRunFor, '%d%m%Y').date() # Get date in string format
    print('\n------ Starting script: Run {} Evolutionary Algorithm generations for {} ------\n'.format(numberOfgenerations, date))

    print('The following price matrix is given: \n')
    print(price_df.to_string(index=False))

    # Concatenate production plans to one dataframe to display for informational purpose
    startPopulation_df = pd.DataFrame(index=range(24))
    for i in range(len(startPopulation)):
        solution_df = startPopulation[i]
        startPopulation_df['Solution ' + str(i+1)] = solution_df['Production']

    print('\nThe starting population has the following production plans:\n')
    print(startPopulation_df)
    print('-------------------------------------------------------------------------\nStarting Evolutionary Algorithm:\n')

'''
Description: Save the optimal solution (fittest individual) to csv file after all generations are completed
'''
def saveSolutionToCsv(optimalSolution):
    optimalFitness = f"{optimalSolution.loc[24, 'Fitness']:,}"
    visualizeSolution('Optimal Production Plan - Fitness {}'.format(optimalFitness), optimalSolution) # Save plot of optimal solution
    optimalSolution = optimalSolution.loc[0:23, :]
    optimalSolution['Time'] = price_df['Time']
    optimalSolution = optimalSolution[['Time', 'Production']]
    print('\nOptimal Solution - Fitness {}: \n\n{}'.format(optimalFitness, optimalSolution))
    optimalSolution.to_csv('optimalSolution{}.csv'.format(dateToRunFor), index=False)


# Description: MAIN METHOD FOR RUNNING THE EVOLUTIONARY ALGORITHM**************************************************************************************************
def runEa():
    displayInformation() # Print starting conditions in terminal

    # Get population. Index value in list represents the fitness in descending order
    population = evaluatePopulation(startPopulation, price_df) # Evaluate the starting individuals
    
    for i in range(numberOfgenerations): # Iterate through the generations until we reach the terminal generation
        populationSize = len(population) # Save population size in variable

        # Get fittest individual from population
        fittestIndividual = getFittestIndividualFromPopulation(population) # Get fittest individual
        otpimalFitness = fittestIndividual.loc[24, 'Fitness'] # Get the fitness of the fittest individual
        numberOfOffspring = max(2, math.ceil(populationSize * generationIncrease)) # Calculate how many offspring to create in this generation

        print('Running generation {}. Optimal fitness {}, population size {}, new offspring {}'.format(i, f"{otpimalFitness:,}", populationSize, numberOfOffspring))

        # Save plot of fittest individual for the current generation
        visualizeSolution('Fittest Individual generation {} - Fitness {}'.format(i, f"{otpimalFitness:,}"), fittestIndividual)

        # Create probability scheme (representing the individuals likelihood of becoming parents)
        probabilityDistribution = getProbabilityDistribution(populationSize)
        
        # Create children. Pick two parents for each child to create
        for i in range(numberOfOffspring):

            # Get parents
            parent1, parent2 = getParents(population, probabilityDistribution)
            offspring = pd.DataFrame(columns=['Production']) # Create Dataframe to for representing new offspring

            # Iterate genome and pick gene from either Parent 1, Parent 2, mean of parents, or mutation
            for geneIndex in range(parent1.shape[0]):
                offspring.loc[geneIndex, 'Production'] = getGeneValueFromParents(parent1, parent2, geneIndex)
    
            population.append(offspring) # Add offspring to population
        
        # Evalute the new offspring, and structure the surviving individuals in descending order based on their fitness
        population = evaluatePopulation(population, price_df)
    
    # Terminal generation is reached: Return the fittest individual to represent the optimal solution 
    fittestIndividual = getFittestIndividualFromPopulation(population) # Get fittest individual
    fittestIndividual.loc[0:23, 'Production'] = fittestIndividual.loc[0:23,'Production'].astype(int) # Convert production plan to integer MWh values
    return fittestIndividual 
#******************************************************************************************************************************************************************

# Main method *****************************************************************************************************************************************************
if __name__ == "__main__":

    # Global attributes to govern the evolution process
    numberOfgenerations = 10 # How many generations to run before the process completed
    populationMaxSize = 200 # Maximum number of individuals to carry over to the next generation
    generationIncrease = 0.25 # Rate at which the population increases per generation
    genePickerProbability = [0.3, 0.3, 0.3, 0.1] # represent [Parent 1, Parent 2, (Parent 1 + Parent 2) / 2, mutation] respectively

    # Global attributes defining the hydro attributes
    minProduction = 0 # MWh
    maxProduction = 100 # MWh
    optimalProduction = 60 # MWh
    startReservoirLevel = 10000 # m³/s

    # Define input paramaters
    dateToRunFor = '14092023' # Date format ddmmY (used for extracting the correct price forecast)
    price_df = getPricePerHour() # Dataframe for the chosen days onde-dimensional price forecast
    waterValue = price_df['Price [€/MWh]'].mean() # Estimate water value as an average of the price
    startReservoirValue = startReservoirLevel * waterValue # Value of reservoir before production starts

    # Create starting population
    startPopulation = getStartPopulation() # List of individuals to start the evolutionary algorithm from
    # Visualize the individuals in the starting population
    for i in range(len(startPopulation)):
        visualizeSolution('Starting Individual ' + str(i+1), startPopulation[i])

    # Run evolutionary algorithm to find the optimal solution
    optimalSolution = runEa()
    # Save best solution to csv file
    saveSolutionToCsv(optimalSolution)
# *****************************************************************************************************************************************************************