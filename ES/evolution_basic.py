import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1             # DNA大小
DNA_BOUND = [0, 5]       # DNA取值边界
N_GENERATIONS = 200      # 迭代次数
POP_SIZE = 100           # 种群大小
N_KID = 50               # 每次迭代生成孩子个数

def F(x):   # 定义函数
    return np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(pred):  # 适应度
    return pred.flatten()

def make_kid(pop, n_kid):   # 产生孩子的过程
    kids = {'DNA':np.empty((n_kid,DNA_SIZE))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])

    for kv,ks in zip(kids['DNA'],kids['mut_strength']):
        p1,p2 = np.random.choice(np.arange(POP_SIZE),size=2,replace=True)
        cp = np.random.randint(0,2,DNA_SIZE,dtype=np.bool)
        # 交叉过程
        kv[cp] = pop['DNA'][p1,cp]
        kv[~cp] = pop['DNA'][p2,~cp]
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]
        # 变异过程
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5),0.) # 变异程度也可以改变，保持大于0
        kv += ks * np.random.rand(*kv.shape)
        kv[:] = np.clip(kv,*DNA_BOUND)

    return kids

def kill_bad(pop, kids): # 抛弃不好的
    for key in ['DNA','mut_strength']:  # 合并父亲和儿子
        pop[key] = np.vstack((pop[key],kids[key]))

    # 利用适应度选择出好的，排序，选出前面的POP_SIZE
    fitness = get_fitness(F(pop['DNA']))
    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][-POP_SIZE:]

    for key in ['DNA','mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop


if __name__ == "__main__":
    pop = dict(DNA = 5*np.random.rand(1,DNA_SIZE).repeat(POP_SIZE,axis=0),
               mut_strength=np.random.rand(POP_SIZE,DNA_SIZE))  # 初始化DNA和变异强度

    plt.ion()
    x = np.linspace(*DNA_BOUND,200)
    plt.plot(x,F(x))

    for _ in range(N_GENERATIONS):  # 进化过程
        if 'sca' in globals():
            sca.remove()
        sca = plt.scatter(pop['DNA'],F(pop['DNA']),s=200,lw=0,c='red',alpha=0.5)    # 当前种群数据
        plt.pause(0.05)

        # 进化过程
        kids = make_kid(pop,N_KID)
        pop = kill_bad(pop,kids)

    plt.ioff()
    plt.show()