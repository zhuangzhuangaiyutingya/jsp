import numpy as np
import math
import matplotlib.pyplot as plt
import random
import copy
import random

def createInd(J):
    '''
    初始化操作，一次初始化一个个体，机器矩阵从1开始
    J: 机器顺序矩阵，J[i, j]表示加工i工件的第j个操作的机器号。大小为n*m
    T: 加工时间矩阵，T[i, j]表示工件j再机器i上的加工时间。大小为m*n
    '''
    n = J.shape[0]  # 工件数
    # m = J.shape[1]  # 机器数
    s = []
    Ji = J.copy()
    while not np.all(Ji == 0):
        I = np.random.randint(0, n)
        M = Ji[I, 0]
        if M != 0:
            s.append(I)
            b = np.roll(Ji[I, :], -1)
            b[-1] = 0
            Ji[I, :] = b
    return s


def createPop(Jm, popSize):  # 创建初始种群
    pop = []
    for i in range(popSize):  # 在0到popsize-1进行循环
        pop.append(createInd(Jm))
    return pop

def decode(J, P, s):  # 解码函数
    n, m = J.shape
    #   T为每个机器的调度信息三维列表，记录了该机器上执行的工序的开始时间、工件号、工序编号和结束时间
    T = [[[0]] for _ in range(m)]  # 创建一个三维列表（或数组）T，其结构为 m 个元素，每个元素都是一个包含单个元素 0 的二维列表
    C = np.zeros((n, m))  # C记录每个工件（工件序号为行数）完成每个工序后的时间（工序数为列数）
    k = np.zeros(n, dtype=int)  # k为工件工序完成情况矩阵（列数：工件标号，元素代表当前完成工序数目）

    for job in s:  # s为初始化的第一条染色体
        machine = J[job, k[job]] - 1
        process_time = P[job, k[job]]
        last_job_finish = C[job, k[job] - 1] if k[job] > 0 else 0  # 如果k[job] > 0赋值左边的，否则赋值0

        # 寻找机器上的第一个合适空闲时间段
        start_time = max(last_job_finish, T[machine][-1][-1])  # 默认在最后一个任务后开始
        insert_index = len(T[machine])  # 默认插入位置在末尾
        for i in range(1, len(T[machine])):
            gap_start = max(T[machine][i - 1][-1], last_job_finish)
            mm = machine;
            qq = T[machine][i - 1][-1];
            gap_end = T[machine][i][0]
            if gap_end - gap_start >= process_time:
                start_time = gap_start  # 找到合适的起始时间
                insert_index = i  # 更新插入位置
                break

        end_time = start_time + process_time
        C[job, k[job]] = end_time
        T[machine].insert(insert_index, [start_time, job, k[job], end_time])
        k[job] += 1

    # 根据T矩阵构建M矩阵
    M = [[] for _ in range(m)]
    for machine in range(m):
        for entry in T[machine][1:]:
            M[machine].append(entry[1])  # M返回`每台机器先后加工工件的工件号`

    return T, M, C

def jisuan_time(T):
    wait_time = 0
    for i in T:
        for j in range(1, len(i)):
            wait_time += i[j][0] - i[j - 1][-1]
    return  wait_time

def getP(c,t):
    p = math.exp(-c / t);
    return p

def rao_dong(pop,n,m):
    # index1, index2 = random.sample(range(n * m), 2)
    # pop[i][index1], pop[i][index2] = pop[i][index2], pop[i][index1]
    num_index = 30;
    indices = random.sample(range(n * m), num_index*2)
    # pop[index1], pop[index2] = pop[index2], pop[index1]
    # pop[index3], pop[index4] = pop[index4], pop[index3]
    # pop[index5], pop[index6] = pop[index6], pop[index5]

    for i in range(num_index):
        index1 = indices[2 * i];
        index2 = indices[2 * i + 1];
        pop[index1], pop[index2] = pop[index2], pop[index1];
    return pop

def drawGantt(timeList):  # 绘制甘特图
    T = timeList.copy()
    # 创建一个新的图形
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots(figsize=(10, 6))

    # 颜色映射字典，为每个工件分配一个唯一的颜色
    color_map = {}
    for machine_schedule in T:
        for task_data in machine_schedule[1:]:
            job_idx, operation_idx = task_data[1], task_data[2]
            if job_idx not in color_map:
                # 为新工件分配一个随机颜色
                color_map[job_idx] = (random.random(), random.random(), random.random())

    # 遍历机器
    for machine_idx, machine_schedule in enumerate(T):
        for task_data in machine_schedule[1:]:
            start_time, job_idx, operation_idx, end_time = task_data
            color = color_map[job_idx]  # 获取工件的颜色

            # 绘制甘特图条形，使用工件的颜色
            ax.barh(machine_idx, end_time - start_time, left=start_time, height=0.4, color=color)

            # 在色块内部标注工件-工序
            label = f'{job_idx}-{operation_idx}'
            ax.text((start_time + end_time) / 2, machine_idx, label, ha='center', va='center', color='white',
                    fontsize=10)

    # 设置Y轴标签为机器名称
    ax.set_yticks(range(len(T)))
    ax.set_yticklabels([f'Machine {i + 1}' for i in range(len(T))])

    # 设置X轴标签
    plt.xlabel("时间")

    # 添加标题
    plt.title("JSP问题甘特图")

    # 创建图例，显示工件颜色
    legend_handles = []
    for job_idx, color in color_map.items():
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f'Job {job_idx}'))
    plt.legend(handles=legend_handles, title='工件')

    # # 显示图形
    # plt.show()

J = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 2, 4, 9, 3, 1, 6, 5, 7, 8],
    [1, 0, 3, 2, 8, 5, 7, 6, 9, 4],
    [1, 2, 0, 4, 6, 8, 7, 3, 9, 5],
    [2, 0, 1, 5, 3, 4, 8, 7, 9, 6],
    [2, 1, 5, 3, 8, 9, 0, 6, 4, 7],
    [1, 0, 3, 2, 6, 5, 9, 8, 7, 4],
    [2, 0, 1, 5, 4, 6, 8, 9, 7, 3],
    [0, 1, 3, 5, 2, 9, 6, 7, 4, 8],
    [1, 0, 2, 6, 8, 9, 5, 3, 4, 7]]) + 1
P = np.array([
    [29, 78,  9, 36, 49, 11, 62, 56, 44, 21],
    [43, 90, 75, 11, 69, 28, 46, 46, 72, 30],
    [91, 85, 39, 74, 90, 10, 12, 89, 45, 33],
    [81, 95, 71, 99,  9, 52, 85, 98, 22, 43],
    [14,  6, 22, 61, 26, 69, 21, 49, 72, 53],
    [84,  2, 52, 95, 48, 72, 47, 65,  6, 25],
    [46, 37, 61, 13, 32, 21, 32, 89, 30, 55],
    [31, 86, 46, 74, 32, 88, 19, 48, 36, 79],
    [76, 69, 76, 51, 85, 11, 40, 89, 26, 74],
    [85, 13, 61,  7, 64, 76, 47, 52, 90, 45]])

n = J.shape[0]  # 工件数
m = J.shape[1]  # 机器数
ct = 1000;  # 初始化
ct_min=1e-14
alpha=0.98;  # 温度的下降率
group_size = 1000;
pop_ind_group = createPop(J,group_size)
C_g = [0]*group_size;
k = 0;
for i in pop_ind_group:
    _ ,_ ,C_gi = decode(J,P,i)
    C_g[k] = C_gi.max();
    k += 1;

# 找到最小值
min_value = min(C_g)
# 找到最小值的索引
min_index = C_g.index(min_value)

pop_ind = pop_ind_group[min_index];

T0, _, C0 = decode(J, P, pop_ind)
wt0 =  jisuan_time(T0);
T_best = T0;
C_best = C0;
wt_best = wt0;
pop_best = pop_ind;
chistory = []

while ct>ct_min:
    for i in range(100):
        pop_new = rao_dong(pop_ind,n,m)
        T_new, _, C_new = decode(J, P, pop_new);
        Cn = C_new.max()
        Cb = C_best.max()
        wt_new = jisuan_time(T_new);
        delta = Cn - Cb;
        if delta<0:
            C0 = C_new;
            T0 = T_new;
            wt0 = wt_new;
            pop_ind = pop_new;
        else:
            Pp = getP(delta,ct);
            R =  random.random();
            if Pp > R:
                C0 = C_new;
                T0 = T_new;
                wt0 = wt_new;
                pop_ind = pop_new;
        Ci = C0.max()
        if Ci < Cb:
            C_best = C0;
            T_best = T0;
            wt_best = wt0;
            pop_best = pop_ind;
    ct = ct*alpha;
    chistory.append(C_best.max())
    print('温度为{}时，最优处理时间{}'.format(ct, C_best.max()))
    print('温度为{}时，平均机器等待时间={}'.format(ct, wt_best / m))
print('最优处理时间={}'.format(C_best.max()))
print('机器等待时间={}'.format(wt_best / m))
print(C_best)
print(T_best)
plt.plot(chistory)
drawGantt(T_best)
plt.show()
