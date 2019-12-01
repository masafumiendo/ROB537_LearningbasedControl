"""
Author: Masafumi Endo
Date: 2019/10/19
Version: 2.0
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class SimulatedAnnealing:

    def __init__(self, T_start, T_stop, num_cities, cooling_rate):

        self.T_start = T_start
        self.T_stop = T_stop
        self.num_cities = num_cities
        self.cooling_rate = cooling_rate

        self.best_route = None
        self.best_total_dist = float("Inf")

    # Public method
    def simulated_annealing(self, cities, iteration):

        epoch = 0
        T = self.T_start

        location, route, total_dist, = self.__initializer(cities)
        best_value_list = []

        best_total_dist_prev = self.best_total_dist
        best_value_list.append(self.best_total_dist)

        while T >= self.T_stop:

            route_candidate = list(route)
            i = random.randint(2, self.num_cities - 1)
            j = random.randint(0, self.num_cities - i)
            route_candidate[j:(j+i)] = reversed(route_candidate[j:(j+i)])

            route, total_dist = self.__selection(T, location, route_candidate, route, total_dist)
            T = self.cooling_rate * T

            self.__visualizer(location, T, iteration, epoch=epoch)

            epoch += 1
            best_total_dist_prev = self.best_total_dist
            best_value_list.append(self.best_total_dist)

        return self.best_total_dist, self.best_route, best_value_list

    # Private method
    def __initializer(self, cities):

        x_coordinate = [row[0] for row in cities]
        y_coordinate = [row[1] for row in cities]

        coordinate_system = [[x_coordinate[i], y_coordinate[i]] for i in range(self.num_cities)]

        location = {}

        for i in range(self.num_cities):
            location[i] = coordinate_system[i]

        initial_route = [i for i in range(self.num_cities)]
        route = self.best_route = random.sample(initial_route, self.num_cities)
        total_dist = self.best_total_dist = self.__total_dist(location, route)

        return location, route, total_dist

    def __selection(self, T, location, route_candidate, route, total_dist):

        total_dist_candidate = self.__total_dist(location, route_candidate)

        if total_dist_candidate < total_dist:
            route, total_dist = route_candidate, total_dist_candidate

            if total_dist_candidate < self.best_total_dist:
                self.best_route, self.best_total_dist = route_candidate, total_dist_candidate

        else:
            if random.random() < np.exp(- abs(total_dist_candidate - total_dist) / T):
                route, total_dist = route_candidate, total_dist_candidate
            else:
                route, total_dist = route, total_dist

        return route, total_dist

    def __total_dist(self, location, route):

        x_coordinate = [location[route[x]][0] for x in range(len(route))]
        y_coordinate = [location[route[y]][1] for y in range(len(route))]

        list_total_dist = []

        for i in range(len(route)):

            if i == len(route) - 1:
                _total_dist = np.sqrt((x_coordinate[i] - x_coordinate[0])**2 + (y_coordinate[i] - y_coordinate[0])**2)

            else:
                _total_dist = np.sqrt((x_coordinate[i] - x_coordinate[i+1])**2 + (y_coordinate[i] - y_coordinate[i+1])**2)

            list_total_dist.append(_total_dist)

        total_dist = sum(list_total_dist)

        return total_dist

    def __visualizer(self, location, T, i, epoch=0):

        x_coordinate = [location[self.best_route[i]][0] for i in range(len(self.best_route))]
        y_coordinate = [location[self.best_route[i]][1] for i in range(len(self.best_route))]
        x_coordinate.append(location[self.best_route[0]][0])
        y_coordinate.append(location[self.best_route[0]][1])

        fig = plt.figure()
        plt.scatter(x_coordinate, y_coordinate)
        plt.plot(x_coordinate, y_coordinate, label=self.best_total_dist)
        plt.title("Temperature: {}".format(T))
        plt.legend()

        plt.savefig("./result/sa/25a/" + str(i) + "/tsp{0:03f}.png".format(epoch))
        plt.close(fig)

def main():

    df = pd.read_csv('./hw2_data/25cities_a.csv', header=None)
    cities = df.values.tolist()
    num_cities = len(cities)

    T_start = 10000
    T_stop = 0.001
    cooling_rate = 0.95

    best_total_dist_list = []
    best_route_list = []
    time_list = []

    for i in range(10):
        start = time.time()

        simulated_annealing = SimulatedAnnealing(T_start, T_stop, num_cities, cooling_rate)
        best_total_dist, best_route, best_value_list = simulated_annealing.simulated_annealing(cities, i+1)

        fig = plt.figure()
        plt.plot(np.arange(0, len(best_value_list)), best_value_list)
        plt.xlabel("iteration")
        plt.ylabel("total travel distance")

        plt.savefig("./result/sa/25a/" + str(i+1) + "/result.png")
        plt.close(fig)

        elapsed_time = time.time() - start

        best_total_dist_list.append(best_total_dist)
        best_route_list.append(best_route)
        time_list.append(elapsed_time)

    best_total_dist_list = pd.DataFrame(best_total_dist_list)
    best_route_list = pd.DataFrame(best_route_list)
    time_list = pd.DataFrame(time_list)

    best_total_dist_list.to_csv("./result/sa/25a/total_dist.csv")
    best_route_list.to_csv("./result/sa/25a/route.csv")
    time_list.to_csv("./result/sa/25a/time.csv")

if __name__ == '__main__':
    main()