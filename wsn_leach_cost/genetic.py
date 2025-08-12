# -*- coding: utf-8 -*-
from wsn_simulation import WSNSimulation
from module_cost import set_modules_cost
from utilities import GLOBAL_CONFIG, MODULES, DEFAULT_PARAM_VALUES
import time
from calculate_cost import calculate_total_cost_with_simulation
import random
from deap import base, creator, tools, algorithms
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import itertools
import functools

# 使用非交互式后端避免GUI问题
matplotlib.use('Agg')

def compare_individuals(ind1, ind2):
    """多标准排序：
    1. 总代价最小
    2. 模块代价总和最小
    3. 参数偏离常规值最小
    """
    cost1, cost2 = ind1.fitness.values[0], ind2.fitness.values[0]
    if cost1 < cost2:
        return -1
    elif cost1 > cost2:
        return 1

    # 模块代价总和
    n_modules = len(all_modules)
    selected_modules1 = [all_modules[i] for i in range(n_modules) if ind1[i] == 1]
    selected_modules2 = [all_modules[i] for i in range(n_modules) if ind2[i] == 1]

    module_cost1 = sum(MODULES[m]['cost'] for m in selected_modules1)
    module_cost2 = sum(MODULES[m]['cost'] for m in selected_modules2)

    if module_cost1 < module_cost2:
        return -1
    elif module_cost1 > module_cost2:
        return 1

    # 参数偏离程度
    param1 = ind1[n_modules:]
    param2 = ind2[n_modules:]

    param_keys = list(DEFAULT_PARAM_VALUES.keys())
    deviation1 = sum(abs(p - DEFAULT_PARAM_VALUES[k]) for p, k in zip(param1, param_keys))
    deviation2 = sum(abs(p - DEFAULT_PARAM_VALUES[k]) for p, k in zip(param2, param_keys))

    if deviation1 < deviation2:
        return -1
    elif deviation1 > deviation2:
        return 1

    return 0  # 完全相等

def solution_sort_key(sol):
    # 计算模块代价总和
    module_cost = sum(MODULES[m]['cost'] for m in sol['modules'])

    # 参数偏离度计算
    param_keys = list(DEFAULT_PARAM_VALUES.keys())
    param_values = [
        sol['warning_energy'],
        sol['preventive_check_days'],
        sol['frequency_heartbeat'] if sol['frequency_heartbeat'] is not None else DEFAULT_PARAM_VALUES["frequency_heartbeat"],
        sol['heartbeat_loss_threshold'] if sol['heartbeat_loss_threshold'] is not None else DEFAULT_PARAM_VALUES["heartbeat_loss_threshold"]
    ]
    param_deviation = sum(abs(p - DEFAULT_PARAM_VALUES[k]) for p, k in zip(param_values, param_keys))

    return (sol['total_cost'], module_cost, param_deviation)

# 设置模块成本
set_modules_cost()

# 获取所有模块名称
all_modules = list(MODULES.keys())
print(f"Total modules available: {len(all_modules)}")
application = "animal_room"  #  animal_room,  electricity_meter

# 定义适应度函数（单目标：最小化总成本）
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 单目标最小化

# 定义个体（二进制列表表示模块是否被选择 + 仿真参数）
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate_individual(individual):
    """评估个体的适应度并返回适应度值和成本明细"""
    n_modules = len(all_modules)
    # 提取模块选择部分
    selected_modules = [all_modules[i] for i in range(n_modules) if individual[i] == 1]
    
    # 提取仿真参数部分
    param_start_idx = n_modules
    simulation_params = {
        "warning_energy": individual[param_start_idx],
        "preventive_check_days": int(round(individual[param_start_idx + 1])) if individual[param_start_idx + 1] > 0 else 1,
        "frequency_heartbeat": max(60, int(round(individual[param_start_idx + 2]))) if "heartbeat" in selected_modules else None,
        "heartbeat_loss_threshold": int(round(individual[param_start_idx + 3])) if "heartbeat" in selected_modules else None
    }
    
    # 创建仿真实例
    sim = WSNSimulation(GLOBAL_CONFIG, selected_modules, simulation_params, application)
    
    # 计算总成本
    total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost = calculate_total_cost_with_simulation(
        sim, selected_modules=selected_modules
    )
    
    # 计算预算超出部分
    budget = GLOBAL_CONFIG["budget"]
    total_budget_cost = base_cost + module_cost + check_cost
    budget_exceed = max(0, total_budget_cost - budget)

    # 创建成本明细字典
    cost_breakdown = {
        "total_cost": total_cost,
        "base_cost": base_cost,
        "module_cost": module_cost,
        "check_cost":check_cost,
        "fault_cost": fault_cost,
        "data_loss_cost": data_loss_cost,
    }
    
    # 保持硬约束，预算超支直接返回无穷大
    if budget_exceed > 0:
        return (float('inf'),), cost_breakdown
    
    return (total_cost,), cost_breakdown

def calculate_genetic_diversity(population):
    """计算种群的基因多样性（基于汉明距离）"""
    if len(population) < 2:
        return 0.0
    
    n_modules = len(all_modules)
    diversity = 0.0
    count = 0
    
    # 计算所有个体两两之间的汉明距离（仅模块部分）
    for i, j in itertools.combinations(range(len(population)), 2):
        ind1 = population[i][:n_modules]
        ind2 = population[j][:n_modules]
        # 计算汉明距离（不同基因的数量）
        hamming_dist = sum(g1 != g2 for g1, g2 in zip(ind1, ind2))
        diversity += hamming_dist
        count += 1
    
    # 返回平均汉明距离
    return diversity / count if count > 0 else 0.0

def genetic_algorithm_optimization(num_processes=None):
    """执行遗传算法优化（预算约束下的单目标优化）"""
    # 初始化工具箱
    toolbox = base.Toolbox()
    n_modules = len(all_modules)
    
    # 定义参数范围
    warning_energy_range = (0.0, 50.0)
    check_day_min = round(GLOBAL_CONFIG["frequency_sampling"] / (60 * 60 * 24))
    if check_day_min <= 0:
        check_day_min = 1
    preventive_check_days_range = (check_day_min, 180)
    frequency_heartbeat_max = GLOBAL_CONFIG["frequency_sampling"]
    frequency_heartbeat_min=frequency_heartbeat_max/60
    if frequency_heartbeat_min<=0:
        frequency_heartbeat_min=1
    frequency_heartbeat_range = (frequency_heartbeat_min, frequency_heartbeat_max)
    heartbeat_loss_threshold_range = (3, 15)
    
    # 个体生成函数
    def create_individual():
        # 模块选择部分（二进制）
        modules_part = [random.randint(0, 1) for _ in range(n_modules)]
        
        # 仿真参数部分
        params_part = [
            random.uniform(*warning_energy_range),
            random.uniform(*preventive_check_days_range),
            random.uniform(*frequency_heartbeat_range),
            random.uniform(*heartbeat_loss_threshold_range)
        ]
        
        return modules_part + params_part
    
    # 注册个体和种群生成器
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 注册评估函数
    toolbox.register("evaluate", evaluate_individual)
    
    # 注册选择算子（锦标赛选择）
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # 注册多种交叉算子
    toolbox.register("mate_two_point", tools.cxTwoPoint)
    toolbox.register("mate_uniform", tools.cxUniform, indpb=0.5)
    
    # 自定义变异算子 - 增强变异强度
    def mutMixed(individual, indpb, stagnation_count):
        n = len(individual)
        
        # 基于停滞计数增加变异强度
        mutation_intensity = 1.0 + stagnation_count * 0.2
        
        # 模块部分：按位翻转（增强变异概率）
        for i in range(n_modules):
            # 动态调整变异概率
            adjusted_indpb = min(0.5, indpb * (1.0 + stagnation_count * 0.1))
            if random.random() < adjusted_indpb:
                individual[i] = 1 - individual[i]
        
        # 参数部分：增强变异
        param_indices = range(n_modules, n)
        for i in param_indices:
            # 动态调整变异概率
            adjusted_indpb = min(0.5, indpb * (1.0 + stagnation_count * 0.1))
            if random.random() < adjusted_indpb:
                # 50%概率进行大范围变异（探索） - 增加探索比例
                if random.random() < 0.5:
                    if i == n_modules:  # warning_energy
                        individual[i] = random.uniform(*warning_energy_range)
                    elif i == n_modules + 1:  # preventive_check_days
                        individual[i] = random.randint(*preventive_check_days_range)
                    elif i == n_modules + 2:  # frequency_heartbeat
                        individual[i] = random.uniform(*frequency_heartbeat_range)
                    elif i == n_modules + 3:  # heartbeat_loss_threshold
                        individual[i] = random.randint(*heartbeat_loss_threshold_range)
                # 50%概率进行大范围扰动（利用）
                else:  
                    if i == n_modules:  # warning_energy
                        perturbation = random.gauss(0, 10 * mutation_intensity)  # 增加标准差
                        individual[i] = max(warning_energy_range[0], 
                                          min(warning_energy_range[1], 
                                          individual[i] + perturbation))
                    elif i == n_modules + 1:  # preventive_check_days
                        perturbation = random.randint(-15, 15) * mutation_intensity  # 增加扰动范围
                        individual[i] = max(preventive_check_days_range[0], 
                                          min(preventive_check_days_range[1], 
                                          individual[i] + perturbation))
                    elif i == n_modules + 2:  # frequency_heartbeat
                        perturbation = random.uniform(0.5, 1.5)  # 扩大乘法扰动范围
                        new_val = individual[i] * perturbation
                        individual[i] = max(frequency_heartbeat_range[0], 
                                          min(frequency_heartbeat_range[1], 
                                          new_val))
                    elif i == n_modules + 3:  # heartbeat_loss_threshold
                        perturbation = random.randint(-3, 3) * mutation_intensity  # 增加扰动范围
                        individual[i] = max(heartbeat_loss_threshold_range[0], 
                                          min(heartbeat_loss_threshold_range[1], 
                                          individual[i] + perturbation))
        return individual,
    
    # 注册变异算子
    # 注意：现在mutMixed需要额外的stagnation_count参数
    # 我们将在主循环中部分应用这个函数
    
    # 设置种群规模和迭代次数
    pop_size = 150
    ngen = 100
    
    # 创建初始种群
    pop = toolbox.population(n=pop_size)
    
    # 设置多进程评估
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
        
    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)
    toolbox.register("map", pool.map)
    
    # 评估初始种群
    print(f"Evaluating initial population (using {num_processes} processes)...")
    # 获取评估结果（适应度值和成本明细）
    results = list(toolbox.map(toolbox.evaluate, pop))
    for ind, res in zip(pop, results):
        fitness_value, cost_breakdown = res
        ind.fitness.values = fitness_value
        # 显式存储成本明细到个体
        ind.cost_breakdown = cost_breakdown
    
    # 设置遗传算法参数
    cxpb = 0.85  # 交叉概率
    base_mutpb = 0.25
    min_mutpb = 0.15
    max_mutpb = 0.45
    
    # 创建Hall of Fame记录历史最优解
    hof = tools.HallOfFame(10)
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [],
        'avg_total_cost': [],
        'feasible_count': [],
        'mutation_rate': [],
        'diversity': [],
        'genetic_diversity': []  # 新增基因多样性指标
    }
    
    # 记录初始代
    total_costs = [ind.fitness.values[0] for ind in pop]
    feasible_count = sum(1 for cost in total_costs if cost != float('inf'))
    min_cost = min(total_costs)
    
    convergence_data['min_total_cost'].append(min_cost)
    convergence_data['avg_total_cost'].append(
        sum(c for c in total_costs if c != float('inf')) / feasible_count if feasible_count > 0 else float('inf')
    )
    convergence_data['feasible_count'].append(feasible_count)
    convergence_data['mutation_rate'].append(base_mutpb)
    
    # 计算初始多样性和基因多样性
    feasible_costs = [c for c in total_costs if c != float('inf')]
    diversity = np.std(feasible_costs) if feasible_costs else 0
    genetic_diversity = calculate_genetic_diversity(pop)
    
    convergence_data['diversity'].append(diversity)
    convergence_data['genetic_diversity'].append(genetic_diversity)
    
    # 运行遗传算法
    print("Starting GA optimization...")
    start_time = time.time()
    
    # 添加停滞检测和响应机制
    stagnation_count = 0
    stagnation_threshold = 10  # 连续10代无改进触发响应
    last_improvement = min_cost
    
    for gen in range(ngen):
        # 计算当前代的基因多样性
        current_genetic_diversity = calculate_genetic_diversity(pop)
        
        # 自适应调整变异概率（基于多样性和停滞情况）
        # 主要基于基因多样性进行调整
        if current_genetic_diversity < 0.5 * convergence_data['genetic_diversity'][0] or stagnation_count > 0:
            # 基因多样性下降或停滞时大幅增加变异率
            mutpb = min(max_mutpb, base_mutpb * (1.5 + stagnation_count * 0.2))
        else:
            mutpb = base_mutpb
            
        convergence_data['mutation_rate'].append(mutpb)
        
        # 选择父代个体进行繁殖
        parents = toolbox.select(pop, len(pop))
        
        # 复制选中的个体
        offspring = [toolbox.clone(ind) for ind in parents]
        
        # 应用交叉操作
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                # 随机选择交叉算子（50%两点交叉，50%均匀交叉）- 增加均匀交叉比例
                if random.random() < 0.5:
                    toolbox.mate_two_point(child1, child2)
                else:
                    toolbox.mate_uniform(child1, child2)
                
                # 交叉后增加小变异（概率基于停滞情况）
                extra_mut_prob = 0.3 + stagnation_count * 0.05
                if random.random() < extra_mut_prob:
                    # 使用增强的变异函数，传入当前停滞计数
                    mutMixed(child1, 0.3, stagnation_count)
                    del child1.fitness.values
                if random.random() < extra_mut_prob:
                    mutMixed(child2, 0.3, stagnation_count)
                    del child2.fitness.values
                else:
                    del child1.fitness.values
                    del child2.fitness.values
        
        # 应用变异操作 - 使用增强的变异函数，传入当前停滞计数
        for mutant in offspring:
            if random.random() < mutpb:
                mutMixed(mutant, mutpb, stagnation_count)
                del mutant.fitness.values
        
        # 评估所有新个体（那些被修改的）
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            results = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, res in zip(invalid_ind, results):
                fitness_value, cost_breakdown = res
                ind.fitness.values = fitness_value
                ind.cost_breakdown = cost_breakdown
        
        # 合并父代和子代
        combined_pop = pop + offspring
        
        # 动态平衡选择策略
        # 精英比例随停滞增加而减少（初始80%，每代停滞减少3%）
        elite_ratio = max(0.5, 0.8 - stagnation_count * 0.03)
        elite_size = int(elite_ratio * pop_size)
        
        # 按适应度排序
        sorted_pop = sorted(combined_pop, key=functools.cmp_to_key(compare_individuals))

        elite = sorted_pop[:elite_size]
        
        # 选择非精英部分（保持多样性）
        non_elite = sorted_pop[elite_size:]
        random.shuffle(non_elite)
        
        # 构建新一代种群
        pop[:] = elite + non_elite[:pop_size - elite_size]
        
        # 更新Hall of Fame
        hof.update(pop)
        
        # 智能重启机制（基于停滞程度和基因多样性）
        restart_triggered = False
        if stagnation_count > stagnation_threshold // 2 or current_genetic_diversity < 0.3 * convergence_data['genetic_diversity'][0]:
            # 部分重启：替换30%最差个体
            num_replace = max(10, int(0.3 * pop_size))
            pop.sort(key=functools.cmp_to_key(compare_individuals))
            
            # 保留历史最优解
            best_to_keep = tools.selBest(pop, pop_size - num_replace)
            
            # 生成新个体
            new_individuals = toolbox.population(n=num_replace)
            results = toolbox.map(toolbox.evaluate, new_individuals)
            for ind, res in zip(new_individuals, results):
                fitness_value, cost_breakdown = res
                ind.fitness.values = fitness_value
                ind.cost_breakdown = cost_breakdown
            
            pop[:] = best_to_keep + new_individuals
            stagnation_count = 0  # 重置停滞计数器
            print(f"Gen {gen+1}: Partial restart ({num_replace} new individuals)")
            restart_triggered = True
        
        # 记录当前代数据
        total_costs = [ind.fitness.values[0] for ind in pop]
        feasible_count = sum(1 for cost in total_costs if cost != float('inf'))
        current_min = min(total_costs)
        
        # 更新停滞计数器（考虑浮点误差）
        if current_min < last_improvement - 0.001:
            stagnation_count = 0
            last_improvement = current_min
        elif not restart_triggered:  # 如果触发了重启就不增加停滞计数
            stagnation_count += 1
        
        convergence_data['min_total_cost'].append(current_min)
        convergence_data['avg_total_cost'].append(
            sum(c for c in total_costs if c != float('inf')) / feasible_count if feasible_count > 0 else float('inf')
        )
        convergence_data['feasible_count'].append(feasible_count)
        
        # 计算并记录种群多样性和基因多样性
        feasible_costs = [c for c in total_costs if c != float('inf')]
        diversity = np.std(feasible_costs) if feasible_costs else 0
        genetic_diversity = calculate_genetic_diversity(pop)
        
        convergence_data['diversity'].append(diversity)
        convergence_data['genetic_diversity'].append(genetic_diversity)
        
        # 打印进度（增加基因多样性显示）
        print(f"Gen {gen+1}/{ngen}: Min Cost={current_min:.2f}, "
              f"Feasible={feasible_count}/{pop_size}, "
              f"Diversity={diversity:.1f}, "
              f"GeneticDiv={genetic_diversity:.1f}, "
              f"MutPb={mutpb:.3f}, "
              f"Stagnation={stagnation_count}/{stagnation_threshold}")
        
        # 早停机制
        if stagnation_count >= stagnation_threshold:
            print(f"Early stopping at generation {gen+1} due to convergence.")
            break
    
    elapsed = time.time() - start_time
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 从Hall of Fame获取最佳个体
    feasible_individuals = [ind for ind in hof if ind.fitness.values[0] != float('inf')]
    if feasible_individuals:
        best_individual = min(feasible_individuals, key=functools.cmp_to_key(compare_individuals))
    else:
        best_individual = min(pop, key=functools.cmp_to_key(compare_individuals))
    
    return best_individual, pop, elapsed, num_processes, convergence_data

def plot_convergence(convergence_data, elapsed_time, num_procs):
    """分别绘制各类收敛图"""
    
    # 1. 总成本收敛图
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_data['min_total_cost'], 'b-', label='Min Total Cost')
    plt.plot(convergence_data['avg_total_cost'], 'r--', label='Avg Total Cost')
    plt.xlabel('Generation')
    plt.ylabel('Total Cost')
    plt.title(f'Total Cost Convergence\n(Time: {elapsed_time:.2f}s, Processes: {num_procs})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ga_convergence_total_cost.png')
    print("Saved: ga_convergence_total_cost.png")
    plt.close()

    # 2. 可行解数量变化
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_data['feasible_count'], 'g-', label='Feasible Solutions')
    plt.xlabel('Generation')
    plt.ylabel('Number of Solutions')
    plt.title('Feasible Solutions Evolution')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ga_convergence_feasible_count.png')
    print("Saved: ga_convergence_feasible_count.png")
    plt.close()

    # 3. 变异率变化
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_data['mutation_rate'], 'm-', label='Mutation Rate')
    plt.xlabel('Generation')
    plt.ylabel('Mutation Probability')
    plt.title('Adaptive Mutation Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ga_convergence_mutation_rate.png')
    print("Saved: ga_convergence_mutation_rate.png")
    plt.close()

    # 4. 种群多样性
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_data['diversity'], 'c-', label='Fitness Diversity (Std Dev)')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Fitness Diversity Over Generations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ga_convergence_diversity.png')
    print("Saved: ga_convergence_diversity.png")
    plt.close()

    # 5. 基因多样性
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_data['genetic_diversity'], 'y-', label='Genetic Diversity (Avg Hamming Dist)')
    plt.xlabel('Generation')
    plt.ylabel('Genetic Diversity')
    plt.title('Genetic Diversity Over Generations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ga_convergence_genetic_diversity.png')
    print("Saved: ga_convergence_genetic_diversity.png")
    plt.close()


def extract_solution_info(individual):
    """从个体中提取解决方案信息"""
    n_modules = len(all_modules)
    
    # 提取模块选择
    selected_modules = [all_modules[i] for i in range(n_modules) if individual[i] == 1]
    
    # 提取仿真参数
    param_start_idx = n_modules
    warning_energy = individual[param_start_idx]
    preventive_check_days = int(round(individual[param_start_idx + 1]))
    frequency_heartbeat = int(round(individual[param_start_idx + 2])) if "heartbeat" in selected_modules else None
    heartbeat_loss_threshold = int(round(individual[param_start_idx + 3])) if "heartbeat" in selected_modules else None
    
    # 使用显式存储的成本明细
    cost_info = individual.cost_breakdown
    
    return {
        "modules": selected_modules,
        "warning_energy": warning_energy,
        "preventive_check_days": preventive_check_days,
        "frequency_heartbeat": frequency_heartbeat,
        "heartbeat_loss_threshold": heartbeat_loss_threshold,
        "total_cost": cost_info["total_cost"],
        "base_cost": cost_info["base_cost"],
        "module_cost": cost_info["module_cost"],
        "check_cost": cost_info["check_cost"],
        "fault_cost": cost_info["fault_cost"],
        "data_loss_cost": cost_info["data_loss_cost"],
        "budget": GLOBAL_CONFIG["budget"]
    }

if __name__ == "__main__":
    # 运行遗传算法优化
    print("Starting multi-process GA optimization...")
    
    try:
        best_individual, population, elapsed_time, num_procs, convergence_data = genetic_algorithm_optimization()
        
        # 绘制收敛图
        plot_convergence(convergence_data, elapsed_time, num_procs)
        
        # 提取最佳解决方案
        best_solution = extract_solution_info(best_individual)
        
        # 写入结果文件
        with open("ga_optimization_results.txt", "w") as f:
            f.write("Genetic Algorithm Optimization Results (Budget-Constrained)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Optimization Time: {elapsed_time:.2f} seconds | Processes Used: {num_procs}\n")
            f.write(f"Budget: {best_solution['budget']}\n")
            f.write(f"Total Cost: {best_solution['total_cost']:.2f}\n\n")
            
            # 最佳解决方案
            f.write("Best Solution:\n")
            f.write("=" * 80 + "\n")
            f.write(f"Modules: {best_solution['modules']}\n")
            f.write(f"Warning Energy: {best_solution['warning_energy']:.2f}%\n")
            f.write(f"Preventive Check Days: {best_solution['preventive_check_days']} days\n")
            if best_solution['frequency_heartbeat'] is not None:
                f.write(f"Heartbeat Frequency: {best_solution['frequency_heartbeat']} seconds\n")
                f.write(f"Heartbeat Loss Threshold: {best_solution['heartbeat_loss_threshold']}\n")
            f.write("\nCost Breakdown:\n")
            f.write(f"  Base Cost:      {best_solution['base_cost']:.2f}\n")
            f.write(f"  Module Cost:    {best_solution['module_cost']:.2f}\n")
            f.write(f"  Fault Cost:     {best_solution['fault_cost']:.2f}\n")
            f.write(f"  Data Loss Cost: {best_solution['data_loss_cost']:.2f}\n")
            f.write(f"  Total Cost:     {best_solution['total_cost']:.2f}\n")
            f.write("-" * 80 + "\n\n")
            
            # 其他可行解（最多前10个）
            feasible_solutions = []
            for ind in population:
                if ind.fitness.values[0] != float('inf'):
                    solution = extract_solution_info(ind)
                    feasible_solutions.append(solution)
            
            feasible_solutions.sort(key=solution_sort_key)
            
            f.write(f"Other Feasible Solutions ({len(feasible_solutions)} total):\n")
            f.write("=" * 80 + "\n")
            for i, sol in enumerate(feasible_solutions[:10]):
                f.write(f"Solution {i+1} (Cost: {sol['total_cost']:.2f}):\n")
                f.write(f"  Modules: {sol['modules']}\n")
                f.write(f"  Warning Energy: {sol['warning_energy']:.2f}%\n")
                f.write(f"  Preventive Check Days: {sol['preventive_check_days']} days\n")
                if sol['frequency_heartbeat'] is not None:
                    f.write(f"  Heartbeat Frequency: {sol['frequency_heartbeat']} seconds\n")
                    f.write(f"  Heartbeat Loss Threshold: {sol['heartbeat_loss_threshold']}\n")
                f.write(f"  Base Cost:      {sol['base_cost']:.2f}\n")
                f.write(f"  Module Cost:    {sol['module_cost']:.2f}\n")
                f.write(f"  Check Cost:     {sol['check_cost']:.2f}\n") 
                f.write(f"  Fault Cost:     {sol['fault_cost']:.2f}\n")
                f.write(f"  Data Loss Cost: {sol['data_loss_cost']:.2f}\n")
                f.write(f"  Total Cost:     {sol['total_cost']:.2f}\n")
                f.write("-" * 80 + "\n")

        
        # 打印最佳解决方案
        print("\nBest Solution Found:")
        print(f"  Modules: {best_solution['modules']}")
        print(f"  Warning Energy: {best_solution['warning_energy']:.2f}%")
        print(f"  Preventive Check Days: {best_solution['preventive_check_days']} days")
        if best_solution['frequency_heartbeat'] is not None:
            print(f"  Heartbeat Frequency: {best_solution['frequency_heartbeat']} seconds")
            print(f"  Heartbeat Loss Threshold: {best_solution['heartbeat_loss_threshold']}")
        print("  Cost Breakdown:")
        print(f"    Base Cost:      {best_solution['base_cost']:.2f}")
        print(f"    Module Cost:    {best_solution['module_cost']:.2f}")
        print(f"    Check Cost:    {best_solution['check_cost']:.2f}")
        print(f"    Fault Cost:     {best_solution['fault_cost']:.2f}")
        print(f"    Data Loss Cost: {best_solution['data_loss_cost']:.2f}")
        print(f"    Total Cost:     {best_solution['total_cost']:.2f}")
        print(f"  Budget: {best_solution['budget']}")
                
        # 打印执行时间
        print(f"\nGA optimization completed! Time: {elapsed_time:.2f} seconds | Processes: {num_procs}")
        print(f"Feasible solutions found: {len([ind for ind in population if ind.fitness.values[0] != float('inf')])}")
    
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

        sys.exit(1)
