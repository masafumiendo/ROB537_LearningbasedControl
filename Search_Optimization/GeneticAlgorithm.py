"""
Author: Masafumi Endo
Date: 2019/10/14
Version: 1.0.0
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time

class GeneticAlgorithm:

    # Public method
    def genetic(self, cities, num_cities, num_population, num_generation, tournament_num_select, tournament_size, num_elite, probability_co, probability_mu, i):

        # Initialization
        location, route = self.__initializer(cities, num_cities, num_population)
        best_value_list = []
        # Evaluation
        evaluation, best_value, best_route = self.__evaluator(location, route, i)
        best_value_list.append(best_value)
        # Loop for next generation
        for loop in range(num_generation):

            population_select, population_elite = self.__selection(route, evaluation, tournament_num_select, tournament_size, num_elite)
            population_next = []

            while True:
                population_1, population_2 = self.__crossover(population_select, probability_co)
                population_1 = self.__mutation(population_1, probability_mu, num_cities)
                population_2 = self.__mutation(population_2, probability_mu, num_cities)

                population_next.append(population_1)
                population_next.append(population_2)

                if len(population_next) >= num_population - num_elite:
                    break

            population_next.extend(population_elite)

            evaluation, best_value, best_route = self.__evaluator(location, population_next, i, loop=loop+1)
            best_value_list.append(best_value)
            route = population_next

        return best_value, best_route, best_value_list

    # Private method
    def __initializer(self, cities, num_cities, num_population):

        x_coordinate = [row[0] for row in cities]
        y_coordinate = [row[1] for row in cities]

        coordinate_system = [[x_coordinate[i], y_coordinate[i]] for i in range(num_cities)]

        location = {}

        for i in range(num_cities):
            location[i] = coordinate_system[i]

        initial_route = [i for i in range(num_cities)]
        route = [random.sample(initial_route, num_cities) for _ in range(num_population)]

        return location, route

    def __evaluator(self, location, route, iteration, loop=0):

        evaluation = []

        for i in range(len(route)):

            _evaluation = []

            x_coordinate = [location[route[i][x]][0] for x in range(len(route[i]))]
            y_coordinate = [location[route[i][y]][1] for y in range(len(route[i]))]

            # Compute objective function (total route length)
            for j in range(len(route[i])):

                if j == len(route[i]) - 1:

                    dist = np.sqrt((x_coordinate[j] - x_coordinate[0])**2 + (y_coordinate[j] - y_coordinate[0])**2)

                else:

                    dist = np.sqrt((x_coordinate[j] - x_coordinate[j+1])**2 + (y_coordinate[j] - y_coordinate[j+1])**2)

                _evaluation.append(dist)

            evaluation.append(sum(_evaluation))

        best_value = min(evaluation)
        draw_population_index = evaluation.index(best_value)

        self.__visualizer(location, route[draw_population_index], best_value, iteration, loop=loop+1)

        return evaluation, best_value, route[draw_population_index]

    def __selection(self, route, evaluation, tournament_num_select, tournament_size, num_elite, ascending=False):

        population_select = []
        population_elite = []

        while True:

            select = random.sample(evaluation, tournament_size)
            select.sort(reverse=ascending)

            for i in range(tournament_num_select):

                value = select[i]
                index = evaluation.index(value)
                population_select.append(route[index])

            if len(population_select) >= len(route) / 2:
                break

        evaluation_sort = copy.deepcopy(evaluation)
        evaluation_sort.sort(reverse=ascending)

        for i in range(num_elite):

            value = evaluation_sort[i]
            index = evaluation.index(value)
            population_elite.append(route[index])

        return population_select, population_elite

    def __crossover(self, population_select, probability_co):

        population_co = random.sample(population_select, 2)
        population_1 = population_co[0]
        population_2 = population_co[1]

        conf_probability = random.randint(0, 100)

        if conf_probability <= probability_co:

            population_1_new = []
            cut_index = random.randint(1, len(population_1)-2)
            population_1_new.extend(population_1[:cut_index])

            for i in range(len(population_1)):

                if population_2[i] not in population_1_new:
                    population_1_new.append(population_2[i])

            population_2_new = []
            population_2_new.extend(population_1[cut_index:])

            for j in range(len(population_1)):

                if population_2[j] not in population_2_new:
                    population_2_new.append(population_2[j])

            return population_1_new, population_2_new

        else:

            return population_1, population_2

    def __mutation(self, population, probability_mu, num_cities):

        conf_probability = random.randint(0, 100)

        if conf_probability <= probability_mu:

            select_num = [i for i in range(num_cities)]
            select_index = random.sample(select_num, 2)

            a = population[select_index[0]]
            b = population[select_index[1]]
            population[select_index[1]] = a
            population[select_index[0]] = b

        return population

    def __visualizer(self, location, route, best_value, i, loop=0):

        x_coordinate = [location[route[i]][0] for i in range(len(route))]
        y_coordinate = [location[route[i]][1] for i in range(len(route))]
        x_coordinate.append(location[route[0]][0])
        y_coordinate.append(location[route[0]][1])

        fig = plt.figure()
        plt.scatter(x_coordinate, y_coordinate)
        plt.plot(x_coordinate, y_coordinate, label=best_value)
        plt.title("Generation: {}".format(loop))
        plt.legend()

        plt.savefig("./result/ga/25a/" + str(i) + "/tsp{0:03d}.png".format(loop))
        plt.close(fig)

def main():

    df = pd.read_csv('./hw2_data/25cities_a.csv', header=None)
    cities = df.values.tolist()
    num_cities = len(cities)

    num_population = 100
    num_generation = 500

    tournament_size = 20
    tournament_num_select = 5
    num_elite = 1

    probability_co = 50

    probability_mu = 3

    genetic_algorithm = GeneticAlgorithm()

    best_total_dist_list = []
    best_route_list = []
    time_list = []

    for i in range(10):
        start = time.time()

        best_total_dist, best_route, best_value_list = genetic_algorithm.genetic(cities, num_cities, num_population, num_generation, tournament_num_select, tournament_size, num_elite, probability_co, probability_mu, i+1)

        fig = plt.figure()
        plt.plot(np.arange(0, len(best_value_list)), best_value_list)
        plt.xlabel("generation")
        plt.ylabel("total travel distance")

        plt.savefig("./result/ga/25a/" + str(i+1) + "/result.png")
        plt.close(fig)

        elapsed_time = time.time() - start

        best_total_dist_list.append(best_total_dist)
        best_route_list.append(best_route)
        time_list.append(elapsed_time)

    best_total_dist_list = pd.DataFrame(best_total_dist_list)
    best_route_list = pd.DataFrame(best_route_list)
    time_list = pd.DataFrame(time_list)

    best_total_dist_list.to_csv("./result/ga/25a/total_dist.csv")
    best_route_list.to_csv("./result/ga/25a/route.csv")
    time_list.to_csv("./result/ga/25a/time.csv")

if __name__ == '__main__':
    main()