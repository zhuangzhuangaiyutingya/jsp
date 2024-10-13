import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import random
import copy
import random
from mpl_toolkits.mplot3d import Axes3D

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

def fit_fun(x ,J , P):  # 适应函数

    # 对数组进行排序，得到排序后的索引
    sorted_indices = np.argsort(x)

    # 创建一个新的数组用于存放映射后的数值
    mapped_data = np.zeros_like(x)

    # 按照升序将数据分为10组，分别赋值1到10
    for i in range(10):
        mapped_data[0, sorted_indices[0, i * 10:(i + 1) * 10]] = i + 1

    # 将 mapped_data 转换为整数类型
    mapped_data = mapped_data.astype(int)
    # 输出原始数据和映射后的数据

    # 将二维的 mapped_data 转换为一维列表
    mapped_data_1d = mapped_data.flatten().tolist()
    mapped_data_1d_minus_1 = [x - 1 for x in mapped_data_1d]

    time,_,fitness = decode(J , P ,mapped_data_1d_minus_1)
    return time, fitness.max()

def drawGantt(timeList):
    T = timeList.copy()
    # 创建一个新的图形
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots(figsize=(12, 6))

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

def jisuan_time(T):
    wait_time = 0
    for i in T:
        for j in range(1, len(i)):
            wait_time += i[j][0] - i[j - 1][-1]
    return wait_time


class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim,J,P):
        self.__pos = np.random.uniform(-x_max, x_max, (1, dim))  # 粒子的位置
        self.__vel = np.random.uniform(-max_vel, max_vel, (1, dim))  # 粒子的速度
        self.__bestPos = np.zeros((1, dim))  # 粒子最好的位置
        self.j = J
        self.p = P
        self.time, self.__fitnessValue = fit_fun(self.__pos,self.j,self.p)  # 适应度函数值

    def set_pos(self, value):
        self.__pos = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, value):
        self.__bestPos = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, value):
        self.__vel = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue

    def get_time_value(self):
        return self.time


class PSO:
    def __init__(self, dim, size, iter_num, x_max, max_vel, tol,J,P, best_fitness_value=float('Inf'), best_time = [], C1=2, C2=2, W=1):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.tol = tol  # 截至条件
        self.best_fitness_value = best_fitness_value
        self.best_position = np.zeros( dim)  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.j = J
        self.p = P
        self.best_time = best_time;

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim,self.j,self.p) for i in range(self.size)]
        #


    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, value):
        self.best_position = value

    def get_bestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part):
        W = 0.9 - (0.9 - 0.4) * (self.iter_num/ self.max_vel)  # 线性递减惯性权重
        vel_value = self.W * part.get_vel() + self.C1 * np.random.rand() * (part.get_best_pos() - part.get_pos()) \
                    + self.C2 * np.random.rand() * (self.get_bestPosition() - part.get_pos())
        vel_value[vel_value > self.max_vel] = self.max_vel
        vel_value[vel_value < -self.max_vel] = -self.max_vel
        part.set_vel(vel_value)

    # 更新位置
    def update_pos(self, part):
        pos_value = part.get_pos() + part.get_vel()
        part.set_pos(pos_value)
        Ti, value = fit_fun(part.get_pos(), self.j, self.p)
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos(pos_value)
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            self.set_bestPosition(pos_value)
            self.best_time = Ti

    def update_ndim(self):

        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            print('第{}次最佳适应值为{}'.format(i, self.get_bestFitnessValue()))
            if self.get_bestFitnessValue() < self.tol:
                break

        return self.fitness_val_list, self.get_bestPosition(), self.best_time

if __name__ == '__main__':
    # test 香蕉函数
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
        [1, 0, 2, 6, 8, 9, 5, 3, 4, 7]
    ]) + 1
    P = np.array([
        [29, 78, 9, 36, 49, 11, 62, 56, 44, 21],
        [43, 90, 75, 11, 69, 28, 46, 46, 72, 30],
        [91, 85, 39, 74, 90, 10, 12, 89, 45, 33],
        [81, 95, 71, 99, 9, 52, 85, 98, 22, 43],
        [14, 6, 22, 61, 26, 69, 21, 49, 72, 53],
        [84, 2, 52, 95, 48, 72, 47, 65, 6, 25],
        [46, 37, 61, 13, 32, 21, 32, 89, 30, 55],
        [31, 86, 46, 74, 32, 88, 19, 48, 36, 79],
        [76, 69, 76, 51, 85, 11, 40, 89, 26, 74],
        [85, 13, 61, 7, 64, 76, 47, 52, 90, 45]
    ])
    n = J.shape[0]  # 工件数
    m = J.shape[1]  # 机器数
    pop_size = 200  # 粒子个数
    pso = PSO(m*n, pop_size, 2000, 30, 60, 1e-4, J, P, C1=2, C2=2, W=1)
    fit_var_list, best_pos, Tmax = pso.update_ndim()
    wt_best = jisuan_time(Tmax)
    print("最优位置:" + str(best_pos))
    print("最优解:" + str(fit_var_list[-1]))
    print('机器等待时间={}'.format(wt_best / m))
    print(Tmax)
    plt.plot(range(len(fit_var_list)), fit_var_list, alpha=0.5)
    drawGantt(Tmax)
    plt.show()

