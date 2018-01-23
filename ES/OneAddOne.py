import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1             # DNA大小
DNA_BOUND = [0, 5]       # DNA的边界
N_GENERATIONS = 200      # 迭代次数
MUT_STRENGTH = 5.        # 变异强度

def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(pre):
    return pre.flatten()

def make_kid(parent):   # 生成孩子
    kid = parent + MUT_STRENGTH*np.random.randn(DNA_SIZE)
    kid = np.clip(kid,*DNA_BOUND)   # 边界化，不能超过边界

    return kid

def kill_bad(parent, kid):  # 抛弃掉不好的，同时根据结果修改变异强度
    global MUT_STRENGTH
    fp = get_fitness(F(parent))[0]
    fk = get_fitness(F(kid))[0]
    p_target = 1/5

    if fp < fk:
        parent = kid    # 抛弃不好的
        ps = 1
    else:
        ps = 0

    MUT_STRENGTH *= np.exp(1/np.sqrt(DNA_SIZE+1)*(ps-p_target)/(1-p_target))    # 修改变异强度
    return parent


if __name__ == "__main__":
    parent = 5*np.random.rand(DNA_SIZE)
    plt.ion()

    x = np.linspace(*DNA_BOUND,200)

    for _ in range(N_GENERATIONS):
        kid = make_kid(parent)
        py,ky = F(parent),F(kid)
        parent = kill_bad(parent,kid)

        plt.cla()
        plt.scatter(parent,py,s=200,lw=0,c='red',alpha=0.5)
        plt.scatter(kid,ky,s=200,lw=0,c='blue',alpha=0.5)
        plt.text(0,-7,'Mutation strength=%.2f' % MUT_STRENGTH)
        plt.plot(x, F(x))
        plt.pause(0.05)

    plt.ioff()
    plt.show()