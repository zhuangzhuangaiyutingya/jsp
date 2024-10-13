import numpy as np
import random
import matplotlib.pyplot as plt

# 绘制甘特图
def plot_gantt_chart(schedule, processing_data):
    num_jobs = processing_data.shape[0]
    num_machines = processing_data.shape[1] // 2
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab20.colors  # 颜色映射
    for job_id, job_schedule in enumerate(schedule):
        for op_id, (start_time, machine) in enumerate(job_schedule):
            duration = processing_data[job_id, 2 * op_id + 1]
            ax.barh(y=machine, width=duration, left=start_time, height=0.8,
                    color=colors[job_id % len(colors)], edgecolor='black')
            ax.text(start_time + duration / 2, machine, f'Job {job_id + 1}-{op_id + 1}',
                    va='center', ha='center', color='black', fontsize=8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Gantt Chart')
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'Machine {i}' for i in range(num_machines)])
    ax.invert_yaxis()  # 机器编号从上到下
    plt.tight_layout()
    plt.show()

# 调度工件
def schedule_jobs(individual, processing_data):
    num_jobs = processing_data.shape[0]
    num_machines = processing_data.shape[1] // 2

    machine_available_time = [0] * num_machines
    job_next_operation_idx = [0] * num_jobs
    job_available_time = [0] * num_jobs
    job_schedule = [[] for _ in range(num_jobs)]

    for operation_job_id in individual:
        op_idx = job_next_operation_idx[operation_job_id]
        if op_idx >= num_machines:
            continue  # 该作业的所有工序已调度

        machine = int(processing_data[operation_job_id, 2 * op_idx])
        duration = processing_data[operation_job_id, 2 * op_idx + 1]
        start_time = max(machine_available_time[machine], job_available_time[operation_job_id])

        job_schedule[operation_job_id].append((start_time, machine))
        machine_available_time[machine] = start_time + duration
        job_available_time[operation_job_id] = start_time + duration

        job_next_operation_idx[operation_job_id] += 1

    makespan = max(job_available_time)
    return job_schedule, makespan

# 遗传算法850  0.8  0.35
def genetic_algorithm(processing_data, population_size=500, generations=20000, crossover_rate=0.8, mutation_rate=0.35):
    num_jobs = processing_data.shape[0]
    num_machines = processing_data.shape[1] // 2

    # 初始化种群
    operations = []
    for job_id in range(num_jobs):
        operations.extend([job_id] * num_machines)

    population = [random.sample(operations, len(operations)) for _ in range(population_size)]

    best_makespan_history = []
    global_best_individual = None
    global_best_makespan = float('inf')

    # 初始化动态绘图
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Makespan")
    ax.set_title("Best Makespan Over Generations")
    line, = ax.plot([], [], label="Best Makespan")
    ax.legend()


    for gen in range(generations):
        # 计算适应度
        fitness = []
        for individual in population:
            _, makespan = schedule_jobs(individual, processing_data)
            fitness.append(makespan)

        # 找出这一代的最佳个体
        best_makespan_gen = min(fitness)
        best_individual_gen = population[fitness.index(best_makespan_gen)]
        best_makespan_history.append(best_makespan_gen)

        # 更新全局最优解
        if best_makespan_gen < global_best_makespan:
            global_best_makespan = best_makespan_gen
            global_best_individual = best_individual_gen

        # 选择（锦标赛选择）
        tournament_size = 3
        selected = []
        for _ in range(population_size):
            participants = random.sample(list(zip(population, fitness)), tournament_size)
            winner = min(participants, key=lambda x: x[1])
            selected.append(winner[0])

        # 交叉
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            if i + 1 < population_size:
                parent2 = selected[i + 1]
            else:
                parent2 = selected[0]
            if random.random() < crossover_rate:
                child1, child2 = [], []
                counts1 = {job_id: 0 for job_id in range(num_jobs)}
                counts2 = {job_id: 0 for job_id in range(num_jobs)}
                for pos in range(len(parent1)):
                    gene1 = parent1[pos] if random.random() < 0.5 else parent2[pos]
                    gene2 = parent2[pos] if random.random() < 0.5 else parent1[pos]

                    if counts1[gene1] < num_machines:
                        child1.append(gene1)
                        counts1[gene1] += 1
                    else:
                        for job_id in counts1:
                            if counts1[job_id] < num_machines:
                                child1.append(job_id)
                                counts1[job_id] += 1
                                break

                    if counts2[gene2] < num_machines:
                        child2.append(gene2)
                        counts2[gene2] += 1
                    else:
                        for job_id in counts2:
                            if counts2[job_id] < num_machines:
                                child2.append(job_id)
                                counts2[job_id] += 1
                                break
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])

        # 变异
        for i in range(len(new_population)):
            if random.random() < mutation_rate:
                idx1, idx2 = random.sample(range(num_jobs * num_machines), 2)
                new_population[i][idx1], new_population[i][idx2] = new_population[i][idx2], new_population[i][idx1]

        population = new_population

        if (gen + 1) % 100 == 0:
            print(f"Generation {gen + 1}: Best makespan = {global_best_makespan}")
            ax.clear()
            ax.set_xlabel("Generation")
            ax.set_ylabel("Makespan")
            ax.set_title("Best Makespan Over Generations")
            ax.plot(range(len(best_makespan_history)), best_makespan_history, label="Best Makespan")
            ax.legend()
            plt.draw()
            plt.pause(0.1)

    plt.ioff()

    print(f"Global best makespan: {global_best_makespan}")
    print(f"Best individual: {global_best_individual}")
    best_schedule, _ = schedule_jobs(global_best_individual, processing_data)
    plot_gantt_chart(best_schedule, processing_data)

    return global_best_individual, global_best_makespan

# ft10 数据集
if __name__ == "__main__":
    processing_data = np.array([
        [0, 29, 1, 78, 2, 9, 3, 36, 4, 49, 5, 11, 6, 62, 7, 56, 8, 44, 9, 21],
        [0, 43, 2, 90, 4, 75, 9, 11, 3, 69, 1, 28, 6, 46, 5, 46, 7, 72, 8, 30],
        [1, 91, 0, 85, 3, 39, 2, 74, 8, 90, 5, 10, 7, 12, 6, 89, 9, 45, 4, 33],
        [1, 81, 2, 95, 0, 71, 4, 99, 6, 9, 8, 52, 7, 85, 3, 98, 9, 22, 5, 43],
        [2, 14, 0, 6, 1, 22, 5, 61, 3, 26, 4, 69, 8, 21, 7, 49, 9, 72, 6, 53],
        [2, 84, 1, 2, 5, 52, 3, 95, 8, 48, 9, 72, 0, 47, 6, 65, 4, 6, 7, 25],
        [1, 46, 0, 37, 3, 61, 2, 13, 6, 32, 5, 21, 9, 32, 8, 89, 7, 30, 4, 55],
        [2, 31, 0, 86, 1, 46, 5, 74, 4, 32, 6, 88, 8, 19, 9, 48, 7, 36, 3, 79],
        [0, 76, 1, 69, 3, 76, 5, 51, 2, 85, 9, 11, 6, 40, 7, 89, 4,26, 8, 74],
        [1, 85, 0, 13, 2, 61, 6, 7, 8, 64, 9, 76, 5, 47, 3, 52, 4, 90, 7, 45]
    ])

    genetic_algorithm(processing_data)
