import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # 编码长度
POP_SIZE = 100           # 种群大小
CROSS_RATE = 0.8         # 交叉概率
MUTATION_RATE = 0.003    # 遗传概率
N_GENERATIONS = 200      # 迭代次数
X_BOUND = [0, 5]         # x的取值范围


def F(x):   # 计算函数的值
    return np.sin(10*x)*x + np.cos(2*x)*x

def translateDNA(pop):  # 转换DNA到十进制，并归一化到x得区间内
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * X_BOUND[1]
    '''
        这里要注意两个地方：
        1，权重weight=np.arange(DNA_SIZE)[::-1]),这里的数组切片处理容易让人迷惑，[开始：结束：步长],这的负号代表的是从后向前
        2，这里的十进制计算使用pop与weight的内积计算得到，巧妙的运用的矩阵的内积
    '''

def get_fitness(pred):  # 计算适应度值
    return pred + 1e-3 - np.min(pred)

def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    return pop[idx]
    '''
        这里的选择是调用choice方法进行，p代表的选择的概率大小（轮盘选择法），size是选择的个数，我们选择保持种群大小不变
    '''

def crossover(parent, pop):    # 交叉过程
    if np.random.rand() < CROSS_RATE:   # 产生0-1的随机值
        i_ = np.random.randint(0,POP_SIZE,size=1)   # 选择另外一个交叉个体
        cross_points = np.random.randint(0,2,size=DNA_SIZE).astype(np.bool) # 选择交叉点，这里假设每个基因都可以交叉
        parent[cross_points] = pop[i_,cross_points] # 完成交叉
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0    # python中的三元运算符，等价于:child[popint] = np.where(child[point]==0],1,0)
    return child


if __name__ == "__main__":
    pop = np.random.randint(2,size=(POP_SIZE,DNA_SIZE)) # 初始化种群
    plt.ion()   # 打开plt的交互模式
    x = np.linspace(*X_BOUND,200)
    plt.plot(x,F(x))

    for _ in range(N_GENERATIONS):  # 开始遗传算法的迭代
        F_values = F(translateDNA(pop))
        if 'sca' in globals():  # python中locals 和globals两个内置函数，基于字典的访问局部和全局变量的方式
            sca.remove()
        sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5)     # 画出当前的种群数据
        plt.pause(0.05)

        # 遗传算法部分
        fitness = get_fitness(F_values)
        print(_,'Most fitted DNA:',pop[np.argmax(fitness),:])
        pop = select(pop,fitness)
        pop_copy = pop.copy()

        for parent in pop:
            child = crossover(parent,pop_copy)
            child = mutate(child)
            parent[:] = child
    plt.ioff()  # 关闭plt的交互模式，这是plt交互模式的使用方法
    plt.show()