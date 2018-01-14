import matplotlib.pyplot as plt
import numpy as np

N_CITIES = 20  # DNA大小
CROSS_RATE = 0.1
MUTATE_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 500


class GA(object):
    def __init__(self,DNA_size, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size


        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

        '''
            1,如果传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本；而shuffle只是对一个矩阵进行洗牌，无返回值。
            2,如果传入一个整数，它会返回一个洗牌后的arange。
        '''

    def translateDNA(self,DNA,city_position): # 解码：返回DNA序列代表的城市顺序的x,y坐标
        line_x = np.empty_like(DNA,dtype=np.float64)
        line_y = np.empty_like(DNA,dtype=np.float64)
        for i,d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i,:] = city_coord[:,0]
            line_y[i,:] = city_coord[:,1]

        return line_x,line_y

    def get_fitness(self,lx,ly):    # 计算适应度：距离越大，适应度越小
        total_distance = np.empty((lx.shape[0]),dtype=np.float64)
        for i,(xs,ys) in enumerate(zip(lx,ly)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys)))) # 计算距离
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    def select(self,fitness):
        idx = np.random.choice(np.arange(self.pop_size),size=self.pop_size,replace=True,p=fitness/fitness.sum())
        return self.pop[idx]

    def crossover(self,parent,pop): # 交叉：不能简单的把两个互换，必须保持所有点都要访问
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0,self.pop_size,size=1)
            cross_points = np.random.randint(0,2,size=self.DNA_size).astype(np.bool)
            keep_city = parent[~cross_points]
            swap_city = pop[i_,np.isin(pop[i_].ravel(),keep_city,invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self,child): # 变异：不能简单改变某一个值，必须保持所有点都要访问
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child


    def evolve(self,fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent,pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop

class TravelSalesPerson(object):
    def __init__(self,n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self,lx,ly,total_d):
        plt.cla()   # 清洗掉当前的图
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)

if __name__ == "__main__":
    ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

    env = TravelSalesPerson(N_CITIES)

    for generation in range(N_GENERATIONS):
        lx, ly = ga.translateDNA(ga.pop, env.city_position)
        fitness,total_distance = ga.get_fitness(lx,ly)

        ga.evolve(fitness)

        best_inx = np.argmax(fitness)
        print('Gen',generation,'| best fit:%.2f' % fitness[best_inx],)
        env.plotting(lx[best_inx],ly[best_inx],total_distance[best_inx])

    plt.ioff()
    plt.show()