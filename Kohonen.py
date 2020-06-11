import math
import numpy as np
from itertools import permutations 
import random
import matplotlib.pyplot as plt

class Kohonen:
    def __init__(self , alpha , No , start , end , radius , propartional , topology):
        if topology == 'line':
            self.neurons = generate_line(No , start , end , propartional)
        elif topology == 'circle':
            self.neurons = generate_circle((0,0) , No , 1 , propartional)     
        elif topology == '5_on_5':
            self.neurons = generate_5_5(2 , propartional)
        self.alpha = alpha
        self.radius = radius

    def best_match(self,chosen):
        winner = self.neurons[0]
        for i in self.neurons:
            if euclidean_dist(chosen , i) < euclidean_dist(winner , chosen):
                winner = i
        return winner

    def update(self , neuron , chosen):
        for i in range(len(neuron)):
            neuron[i] += self.alpha * (chosen[i] - neuron[i])

    def update_neighbors(self , chosen , bmu):
        for neuron in self.neurons:
            if euclidean_dist(chosen , neuron) <= self.radius and neuron != bmu:
                self.update(neuron,chosen)

    def train(self , data , iterations):
        plt.scatter([row[0] for row in self.neurons] , [row[1] for row in self.neurons] , c='green' , label='Neurons')
        for i in range(len(self.neurons) - 1):
            plt.plot([self.neurons[i][0],self.neurons[i + 1][0]],[self.neurons[i][1],self.neurons[i + 1][1]],'-ok',c='green')
        plt.scatter([row[0] for row in data] , [row[1] for row in data] , c='blue', label='Training data')
        plt.legend(loc="upper left")
        plt.title('Kohonen init situation')
        plt.show()
        init_alpha = self.alpha
        init_radius = self.radius
        i = 0
        while i <= iterations:
            rand_chosen = random.choice(data)
            bmu = self.best_match(rand_chosen)
            self.update(bmu , rand_chosen)
            self.update_neighbors(rand_chosen , bmu)
            self.alpha = init_alpha * np.exp(- i / iterations)
            self.radius = init_radius * np.exp(- i / iterations)
            if i % 1000 == 0:
                draw(self.neurons , data , rand_chosen , bmu , i)
            i += 1

def draw(neurons , data , rand_chosen , bmu , i):
    plt.scatter([row[0] for row in neurons] , [row[1] for row in neurons] ,c='green' , label='Neurons')
    for ij in range(len(neurons) - 1):
        plt.plot([neurons[ij][0],neurons[ij + 1][0]],[neurons[ij][1],neurons[ij + 1][1]],'-ok',c='green')
    plt.scatter([row[0] for row in data] , [row[1] for row in data] , c='blue', label='Training data')
    plt.scatter(rand_chosen[0],rand_chosen[1],c='red' , label='Random data point')
    plt.scatter(bmu[0],bmu[1],c='yellow',label='BMU')
    plt.legend(loc="upper left")
    plt.title('Kohonen simulation , Iteration No. ' + str(i))
    plt.show()

def euclidean_dist(chosen , neuron):
    return np.sqrt((chosen[0] - neuron[0])**2 + (chosen[1] - neuron[1])**2)

def generate_circle(center, num_points, radius , range_=None ,propartional=False):
    arc = (2 * math.pi) / num_points # what is the angle between two of the points
    points = []
    if range_:
        radius = random.choice(range_)
    for p in range(num_points):
        px = (0 * math.cos(arc * p)) - (radius * math.sin(arc * p))
        py = (radius * math.cos(arc * p)) + (0 * math.sin(arc * p))
        px += center[0]
        py += center[1]
        points.append([px,py])
    for i in range(num_points):
        if(points[i][0] == 0 and points[i][1] == 0 and propartional):
            points[i][0] += 0.5
            points[i][1] -= 0.5
    return points

def generate_line(No , start , end, propartional=False):
    neurons = []
    neurons_X = np.random.uniform(low=start, high=end, size=No)
    neurons_Y = [0] * No
    for i , j in zip(neurons_X , neurons_Y):
        if(i == 0 and j == 0 and propartional):
            i += 0.5
            j -= 0.5
        neurons.append([i,j])
    return neurons

def generate_5_5(density, propartional=False):
    points = []
    points.append([-2,-2])
    points.append([-2,-1])
    points.append([-2,0])
    points.append([-2,1])
    points.append([-2,2])
    points.append([-1,-2])
    points.append([-1,-1])
    points.append([-1,0])
    points.append([-1,1])
    points.append([-1,2])
    points.append([0,0])
    points.append([0,1])
    points.append([0,2])
    points.append([0,-1])
    points.append([0,-2])
    points.append([1,-2])
    points.append([1,-1])
    points.append([1,0])
    points.append([1,1])
    points.append([1,2])
    points.append([2,-2])
    points.append([2,-1])
    points.append([2,0])
    points.append([2,1])
    points.append([2,2])
    # Normalize 
    for point in points:
        if(not propartional):
            point[0] /= density
            point[1] /= density 
        else :
            point[0] /= density * 2
            point[1] /= density * 2
    return points

def main():
    net = Kohonen(0.5 , 30 , -2 , 2 , 2 , True , topology='line')
    circle = generate_circle((0,0) , 60 , 3 , np.arange(2.1,3.9,0.1))
    net.train(circle , 5000)

if __name__ == '__main__':
    main()
