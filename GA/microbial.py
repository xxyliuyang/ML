import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10
POP_SIZE = 20
CROSS_RATE = 0.6
MUTATION_RATE = 0.01
N_GENERATIONS = 200
X_BOUND = [0, 5]


def F(x): return np.sin(10*x)*x + np.cos(2*x)*x


class MGA(object):
    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        # 初始化
        self.pop = np.random.randint(*DNA_bound, size=(1, self.DNA_size)).repeat(pop_size, axis=0)

    def translateDNA(self, pop):
        return pop.dot(2 ** np.arange(self.DNA_size)[::-1]) / float(2 ** self.DNA_size - 1) * X_BOUND[1]

    def get_fitness(self, product):
        return product  #不需要负数转化了，因为没有根据概率进行选择

    def crossover(self, loser_winner):      # 交叉：把适应度高的个体，交叉点对应的的值给适应度低的个体
        cross_idx = np.empty((self.DNA_size,)).astype(np.bool)
        for i in range(self.DNA_size):
            cross_idx[i] = True if np.random.rand() < self.cross_rate else False
        loser_winner[0, cross_idx] = loser_winner[1, cross_idx]
        return loser_winner

    def mutate(self, loser_winner):         # 变异：对适应度低的进行
        mutation_idx = np.empty((self.DNA_size,)).astype(np.bool)
        for i in range(self.DNA_size):
            mutation_idx[i] = True if np.random.rand() < self.mutate_rate else False  # 变异位置
        loser_winner[0, mutation_idx] = ~loser_winner[0, mutation_idx].astype(np.bool)
        return loser_winner

    def evolve(self, n):    # 进化过程中，每次悬着两个，对于适应度低的那个，利用使用度高的进行交叉和变异，保留适应度高的
        for _ in range(n):
            sub_pop_idx = np.random.choice(np.arange(0, self.pop_size), size=2, replace=False)
            sub_pop = self.pop[sub_pop_idx]
            product = F(self.translateDNA(sub_pop))
            fitness = self.get_fitness(product)
            loser_winner_idx = np.argsort(fitness)
            loser_winner = sub_pop[loser_winner_idx]    # 排序：第一个是适应度小的，第二个是适应度大的
            loser_winner = self.crossover(loser_winner)
            loser_winner = self.mutate(loser_winner)
            self.pop[sub_pop_idx] = loser_winner

        DNA_prod = self.translateDNA(self.pop)
        pred = F(DNA_prod)
        return DNA_prod, pred

if __name__ == "__main__":
    plt.ion()
    x = np.linspace(*X_BOUND, 200)
    plt.plot(x, F(x))

    ga = MGA(DNA_size=DNA_SIZE, DNA_bound=[0, 1], cross_rate=CROSS_RATE, mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)

    for _ in range(N_GENERATIONS):
        DNA_prod, pred = ga.evolve(5)

        if 'sca' in globals(): sca.remove()
        sca = plt.scatter(DNA_prod, pred, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    plt.ioff();plt.show()