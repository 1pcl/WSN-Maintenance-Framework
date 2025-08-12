# -*- coding: utf-8 -*-
from wsn_simulation import WSNSimulation
from module_cost import set_modules_cost
from utilities import GLOBAL_CONFIG, MODULES, DEFAULT_PARAM_VALUES
import time
from calculate_cost import calculate_total_cost_with_simulation
import random
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import itertools
from deap import base, creator, tools, algorithms
import functools
import os

# 使用非交互式后端避免GUI问题
matplotlib.use('Agg')

# 设置模块成本
set_modules_cost()

# 获取所有模块名称
all_modules = list(MODULES.keys())
print(f"Total modules available: {len(all_modules)}")
application = "animal_room"  #  animal_room, electricity_meter

# 定义适应度函数（单目标：最小化总成本）
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 单目标最小化

# 定义个体（二进制列表表示模块是否被选择 + 仿真参数）
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

# ==================== 通用函数 ====================

def evaluate_individual(individual):
    """评估个体的适应度，返回完整结果字典"""
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
    
    # 返回完整结果字典
    result = {
        "fitness": (float('inf'),) if budget_exceed > 0 else (total_cost,),
        "total_cost": total_cost,
        "base_cost": base_cost,
        "module_cost": module_cost,
        "check_cost": check_cost,
        "fault_cost": fault_cost,
        "data_loss_cost": data_loss_cost,
        "feasible": budget_exceed <= 0,
        "individual": individual,  # 保留原始个体
        "selected_modules": selected_modules,
        "simulation_params": simulation_params
    }
    
    return result

def solution_info_sort_key(sol_info):
    """解决方案信息排序函数，直接使用solution_info字典"""
    # 计算模块代价总和
    module_cost = sum(MODULES[m]['cost'] for m in sol_info['modules'])
    
    # 参数偏离度计算
    param_keys = list(DEFAULT_PARAM_VALUES.keys())
    param_values = [
        sol_info['warning_energy'],
        sol_info['preventive_check_days'],
        sol_info['frequency_heartbeat'] if sol_info['frequency_heartbeat'] is not None else DEFAULT_PARAM_VALUES["frequency_heartbeat"],
        sol_info['heartbeat_loss_threshold'] if sol_info['heartbeat_loss_threshold'] is not None else DEFAULT_PARAM_VALUES["heartbeat_loss_threshold"]
    ]
    param_deviation = sum(abs(p - DEFAULT_PARAM_VALUES[k]) for p, k in zip(param_values, param_keys))

    return (sol_info['total_cost'], module_cost, param_deviation)

def extract_solution_info(solution_data):
    """从解决方案数据中提取信息"""
    return {
        "modules": solution_data["selected_modules"],
        "warning_energy": solution_data["simulation_params"]["warning_energy"],
        "preventive_check_days": solution_data["simulation_params"]["preventive_check_days"],
        "frequency_heartbeat": solution_data["simulation_params"]["frequency_heartbeat"],
        "heartbeat_loss_threshold": solution_data["simulation_params"]["heartbeat_loss_threshold"],
        "total_cost": solution_data["total_cost"],
        "base_cost": solution_data["base_cost"],
        "module_cost": solution_data["module_cost"],
        "check_cost": solution_data["check_cost"],
        "fault_cost": solution_data["fault_cost"],
        "data_loss_cost": solution_data["data_loss_cost"],
        "budget": GLOBAL_CONFIG["budget"],
        "encoded_solution": solution_data["individual"]  # 原始个体
    }

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

# ==================== 遗传算法部分 ====================

def genetic_algorithm_optimization(pop_size=120, ngen=50, num_processes=128):
    """遗传算法优化"""
    # 初始化工具箱
    toolbox = base.Toolbox()
    n_modules = len(all_modules)
    
    # 定义参数范围
    warning_energy_range = (0.0, 50.0)
    check_day_min = round(GLOBAL_CONFIG["frequency_sampling"]/(60*60*24))
    if check_day_min <= 0:
        check_day_min = 1        
    preventive_check_days_range = (check_day_min, 180)
    frequency_heartbeat_max = GLOBAL_CONFIG["frequency_sampling"]
    frequency_heartbeat_min = max(1, frequency_heartbeat_max/60)
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
        
        return creator.Individual(modules_part + params_part)
    
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
    
    # 自定义变异算子
    def mutMixed(individual, indpb):
        """自定义变异操作，处理混合类型基因"""
        n = len(individual)
        
        # 模块部分：按位翻转
        for i in range(n_modules):
            if random.random() < indpb:
                individual[i] = 1 - individual[i]  # 翻转
        
        # 全局参数部分
        if random.random() < indpb:
            individual[n_modules] = random.uniform(*warning_energy_range)
        if random.random() < indpb:
            individual[n_modules+1] = random.uniform(*preventive_check_days_range)
        
        # 心跳模块参数（如果模块被选择）
        if "heartbeat" in [all_modules[i] for i in range(n_modules) if individual[i] == 1]:
            if random.random() < indpb:
                individual[n_modules+2] = random.uniform(*frequency_heartbeat_range)
            if random.random() < indpb:
                individual[n_modules+3] = random.uniform(*heartbeat_loss_threshold_range)
        
        return individual,
    
    # 注册变异算子
    toolbox.register("mutate", mutMixed, indpb=0.2)
    
    # 设置种群规模和迭代次数
    pop = toolbox.population(n=pop_size)
    
    # 创建进程池
    print(f"Creating multiprocessing pool with {num_processes} workers for GA")
    pool = multiprocessing.Pool(processes=num_processes)
    toolbox.register("map", pool.map)
    
    # 评估初始种群
    print(f"Evaluating initial population (using {num_processes} processes)...")
    results = list(toolbox.map(toolbox.evaluate, pop))
    for ind, res in zip(pop, results):
        fitness_value = res["fitness"]
        ind.fitness.values = fitness_value
        # 存储完整结果
        ind.result = res
    
    # 创建Hall of Fame记录历史最优解
    hof = tools.HallOfFame(10)
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [],
        'avg_total_cost': [],
        'feasible_count': [],
        'mutation_rate': [],
        'genetic_diversity': []
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
    convergence_data['mutation_rate'].append(0.2)
    convergence_data['genetic_diversity'].append(calculate_genetic_diversity(pop))
    
    # 运行遗传算法
    print("Starting GA optimization...")
    start_time = time.time()
    
    # 遗传算法参数
    cxpb = 0.85  # 交叉概率
    mutpb = 0.25  # 变异概率
    stagnation_count = 0
    stagnation_threshold = 15
    
    for gen in range(1, ngen + 1):
        # 选择下一代
        offspring = toolbox.select(pop, len(pop))
        
        # 克隆选中个体
        offspring = list(map(toolbox.clone, offspring))
        
        # 应用交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                # 随机选择交叉算子
                if random.random() < 0.5:
                    toolbox.mate_two_point(child1, child2)
                else:
                    toolbox.mate_uniform(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # 评估新个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            results = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, res in zip(invalid_ind, results):
                fitness_value = res["fitness"]
                ind.fitness.values = fitness_value
                ind.result = res
        
        # 替换种群
        pop[:] = offspring
        
        # 更新Hall of Fame
        hof.update(pop)
        
        # 记录当前代数据
        total_costs = [ind.fitness.values[0] for ind in pop]
        feasible_count = sum(1 for cost in total_costs if cost != float('inf'))
        current_min = min(total_costs)
        
        # 更新停滞计数器
        if current_min < min_cost - 0.001:
            stagnation_count = 0
            min_cost = current_min
        else:
            stagnation_count += 1
        
        convergence_data['min_total_cost'].append(current_min)
        convergence_data['avg_total_cost'].append(
            sum(c for c in total_costs if c != float('inf')) / feasible_count if feasible_count > 0 else float('inf')
        )
        convergence_data['feasible_count'].append(feasible_count)
        convergence_data['mutation_rate'].append(mutpb)
        convergence_data['genetic_diversity'].append(calculate_genetic_diversity(pop))
        
        # 打印进度
        print(f"Gen {gen}/{ngen}: Min Cost={current_min:.2f}, "
              f"Feasible={feasible_count}/{pop_size}, "
              f"GeneticDiv={convergence_data['genetic_diversity'][-1]:.1f}, "
              f"Stagnation={stagnation_count}/{stagnation_threshold}")
        
        # 早停机制
        if stagnation_count >= stagnation_threshold:
            print(f"Early stopping at generation {gen} due to convergence.")
            break
    
    elapsed = time.time() - start_time
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 从Hall of Fame获取最佳个体
    feasible_individuals = [ind for ind in hof if ind.fitness.values[0] != float('inf')]
    if feasible_individuals:
        best_individual = min(feasible_individuals, key=lambda ind: ind.fitness.values[0])
    else:
        best_individual = min(pop, key=lambda ind: ind.fitness.values[0])
    
    return best_individual.result, pop, elapsed, num_processes, convergence_data

# ==================== 模拟退火部分 ====================

def neighbor_function(current_solution, temperature, initial_temp, ga_diversity=1.0):
    """生成邻域解，支持自适应扰动"""
    n_modules = len(all_modules)
    neighbor = creator.Individual(current_solution[:])  # 复制当前解
    
    # 定义参数范围
    warning_energy_range = (0.0, 50.0)
    check_day_min = round(GLOBAL_CONFIG["frequency_sampling"]/(60*60*24))
    if check_day_min <= 0:
        check_day_min = 1        
    preventive_check_days_range = (check_day_min, 180)
    frequency_heartbeat_max = GLOBAL_CONFIG["frequency_sampling"]
    frequency_heartbeat_min = max(1, frequency_heartbeat_max/60)
    frequency_heartbeat_range = (frequency_heartbeat_min, frequency_heartbeat_max)
    heartbeat_loss_threshold_range = (3, 15)
    
    # 温度影响因子：高温时扰动大，低温时扰动小
    perturbation_factor = max(0.1, min(1.0, temperature / initial_temp))
    
    # 结合GA多样性调整扰动强度
    perturbation_factor *= ga_diversity
    
    # 随机选择扰动类型
    perturbation_type = random.choice([
        "flip_module", 
        "adjust_warning_energy",
        "adjust_preventive_days",
        "adjust_heartbeat_freq",
        "adjust_heartbeat_threshold"
    ])
    
    # 模块部分：翻转一个随机模块
    if perturbation_type == "flip_module":
        idx = random.randint(0, n_modules - 1)
        neighbor[idx] = 1 - neighbor[idx]
    
    # 参数部分：调整warning_energy
    elif perturbation_type == "adjust_warning_energy":
        param_idx = n_modules
        std_dev = 5 * perturbation_factor
        perturbation = random.gauss(0, std_dev)
        neighbor[param_idx] = max(warning_energy_range[0], 
                                 min(warning_energy_range[1], 
                                 neighbor[param_idx] + perturbation))
    
    # 参数部分：调整preventive_check_days
    elif perturbation_type == "adjust_preventive_days":
        param_idx = n_modules + 1
        perturbation_range = int(10 * perturbation_factor)
        perturbation = random.randint(-perturbation_range, perturbation_range)
        neighbor[param_idx] = max(preventive_check_days_range[0], 
                                 min(preventive_check_days_range[1], 
                                 neighbor[param_idx] + perturbation))
    
    # 参数部分：调整frequency_heartbeat
    elif perturbation_type == "adjust_heartbeat_freq":
        param_idx = n_modules + 2
        perturbation = random.uniform(0.8, 1.2)  # 乘法扰动
        new_val = neighbor[param_idx] * perturbation
        neighbor[param_idx] = max(frequency_heartbeat_range[0], 
                                 min(frequency_heartbeat_range[1], 
                                 new_val))
    
    # 参数部分：调整heartbeat_loss_threshold
    elif perturbation_type == "adjust_heartbeat_threshold":
        param_idx = n_modules + 3
        perturbation_range = int(2 * perturbation_factor)
        perturbation = random.randint(-perturbation_range, perturbation_range)
        neighbor[param_idx] = max(heartbeat_loss_threshold_range[0], 
                                 min(heartbeat_loss_threshold_range[1], 
                                 neighbor[param_idx] + perturbation))
    
    return neighbor

def simulated_annealing_optimization(initial_solution=None, num_processes=128, max_iter=50, ga_diversity=1.0):
    """模拟退火优化 - 减少迭代次数至50"""
    # 初始化工具箱
    n_modules = len(all_modules)
    
    # 定义参数范围
    warning_energy_range = (0.0, 50.0)
    check_day_min = round(GLOBAL_CONFIG["frequency_sampling"]/(60*60*24))
    if check_day_min <= 0:
        check_day_min = 1        
    preventive_check_days_range = (check_day_min, 180)
    frequency_heartbeat_max = GLOBAL_CONFIG["frequency_sampling"]
    frequency_heartbeat_min = max(1, frequency_heartbeat_max/60)
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
        
        return creator.Individual(modules_part + params_part)
    
    # 模拟退火参数
    initial_temp = 1000.0
    final_temp = 0.1
    cooling_rate = 0.95
    max_stagnation = 20
    
    # 创建初始解
    if initial_solution is not None:
        current_solution = creator.Individual(initial_solution[:])
        print("Using provided initial solution for SA")
    else:
        current_solution = create_individual()
    
    # 评估初始解
    current_result = evaluate_individual(current_solution)
    current_cost = current_result["fitness"][0]
    
    # 记录最佳解
    best_solution = creator.Individual(current_solution[:])
    best_cost = current_cost
    best_result = current_result
    best_temp = initial_temp  # 初始化最优温度为初始温度
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [best_cost],
        'current_cost': [current_cost],
        'temperature': [initial_temp],
        'feasible_count': [1 if current_cost != float('inf') else 0]
    }
    
    # 记录历史可行解
    feasible_solutions = []
    if current_result["feasible"]:
        feasible_solutions.append({
            "result": current_result,
            "cost": current_cost
        })
    
    print("Starting simulated annealing optimization...")
    start_time = time.time()
    
    # 初始温度
    temperature = initial_temp
    iteration = 0
    stagnation_count = 0
    
    # 创建进程池
    print(f"Creating multiprocessing pool with {num_processes} workers for SA")
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 定义批量生成邻域解的函数
    def generate_neighbors(current, neighbors_per_iteration):
        """一次生成多个邻域解"""
        return [neighbor_function(current, temperature, initial_temp, ga_diversity) 
                for _ in range(neighbors_per_iteration)]
    
    # 主循环
    while temperature > final_temp and iteration < max_iter and stagnation_count < max_stagnation:
        # 一次生成多个邻域解（数量等于进程数）
        neighbors = generate_neighbors(current_solution, num_processes)
        
        # 并行评估所有邻域解
        results = pool.map(evaluate_individual, neighbors)
        
        # 找出最佳邻域解
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_neighbor_result = None
        for neighbor, result in zip(neighbors, results):
            neighbor_cost = result["fitness"][0]
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
                best_neighbor_result = result
        
        # 使用最佳邻域解进行Metropolis准则判断
        neighbor = best_neighbor
        neighbor_result = best_neighbor_result
        neighbor_cost = best_neighbor_cost
        
        # 更新可行解列表
        if neighbor_result["feasible"]:
            feasible_solutions.append({
                "result": neighbor_result,
                "cost": neighbor_cost
            })
        
        # 计算成本差
        cost_difference = neighbor_cost - current_cost
        
        # Metropolis准则：决定是否接受新解
        accept = False
        if cost_difference < 0:
            # 新解更好，总是接受
            accept = True
        elif temperature > 0:
            # 以一定概率接受较差的解
            acceptance_prob = math.exp(-cost_difference / temperature)
            if random.random() < acceptance_prob:
                accept = True
        
        # 如果接受新解，更新当前状态
        if accept:
            current_solution = creator.Individual(neighbor[:])
            current_cost = neighbor_cost
            current_result = neighbor_result
            
            # 更新最佳解
            if current_cost < best_cost:
                best_solution = creator.Individual(neighbor[:])
                best_cost = current_cost
                best_result = neighbor_result
                stagnation_count = 0  # 重置停滞计数
                best_temp = temperature  # 更新最优温度
                print(f"Iter {iteration}: New best cost = {best_cost:.2f}, Temp = {temperature:.2f}")
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1
        
        # 重新加热机制：当停滞计数达到阈值时增加温度
        if stagnation_count >= max_stagnation//2:
            # 使用最优温度进行重新加热
            new_temperature = best_temp * 1.1
            
            # 设置温度上限为初始温度
            if new_temperature > initial_temp:
                new_temperature = initial_temp
            
            # 设置温度下限
            min_temperature = initial_temp * 0.1
            temperature = max(new_temperature, min_temperature)
            
            stagnation_count = 0
            print(f"Iter {iteration}: Reheating to {temperature:.2f} (based on best temp {best_temp:.2f})")
        
        # 记录收敛数据
        convergence_data['min_total_cost'].append(best_cost)
        convergence_data['current_cost'].append(current_cost)
        convergence_data['temperature'].append(temperature)
        convergence_data['feasible_count'].append(len(feasible_solutions))
        
        # 降温
        temperature *= cooling_rate
        iteration += 1
        
        # 每10次迭代打印进度
        if iteration % 10 == 0:
            print(f"Iter {iteration}: Best cost = {best_cost:.2f}, Current cost = {current_cost:.2f}, "
                  f"Temp = {temperature:.2f}, Stagnation = {stagnation_count}, "
                  f"Feasible = {len(feasible_solutions)}")
    
    elapsed = time.time() - start_time
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 从可行解中选择最佳解
    if feasible_solutions:
        # 使用多标准排序
        feasible_solutions.sort(key=lambda x: x["cost"])
        best_feasible = feasible_solutions[0]["result"]
    else:
        # 如果没有可行解，使用当前最佳解
        best_feasible = best_result
    
    # 填充收敛数据结构
    convergence_data_full = {
        'min_total_cost': convergence_data['min_total_cost'],
        'current_cost': convergence_data['current_cost'],
        'avg_total_cost': convergence_data['min_total_cost'],  # 模拟退火没有平均值
        'feasible_count': convergence_data['feasible_count'],
        'mutation_rate': [cooling_rate] * len(convergence_data['min_total_cost']),
        'genetic_diversity': [0] * len(convergence_data['min_total_cost']),
        'temperature': convergence_data['temperature']
    }
    
    print(f"SA completed in {iteration} iterations. Best cost: {best_feasible['total_cost']:.2f}")
    return best_feasible, feasible_solutions, elapsed, num_processes, convergence_data_full

# ==================== 混合优化部分 ====================

def hybrid_ga_sa_optimization(num_processes=128):
    """结合遗传算法和模拟退火的混合优化"""
    # 第一阶段：遗传算法全局搜索
    print("="*80)
    print("Starting Genetic Algorithm phase...")
    print("="*80)
    
    ga_best, ga_population, ga_elapsed, ga_num_procs, ga_convergence = genetic_algorithm_optimization(
        pop_size=120,  # 用户指定种群规模
        ngen=50,       # 用户指定迭代次数
        num_processes=num_processes
    )
    
    # 计算GA的多样性
    ga_genetic_diversity = ga_convergence['genetic_diversity'][-1] if ga_convergence['genetic_diversity'] else 1.0
    print(f"GA phase completed. Best cost: {ga_best['total_cost']:.2f}, Diversity: {ga_genetic_diversity:.2f}")
    
    # 提取GA的最佳解作为SA的起点
    sa_start_solution = ga_best["individual"]
    
    # 第二阶段：模拟退火局部优化
    print("\n" + "="*80)
    print("Starting Simulated Annealing phase...")
    print("="*80)
    
    sa_best_feasible, sa_feasible_solutions, sa_elapsed, sa_num_procs, sa_convergence = simulated_annealing_optimization(
        initial_solution=sa_start_solution,
        num_processes=num_processes,
        max_iter=50,  # 用户指定迭代次数
        ga_diversity=ga_genetic_diversity
    )
    
    # 合并结果
    elapsed_time = ga_elapsed + sa_elapsed
    convergence_data = {
        "ga_min_cost": ga_convergence["min_total_cost"],
        "sa_min_cost": sa_convergence["min_total_cost"],
        "ga_feasible": ga_convergence["feasible_count"],
        "sa_feasible": sa_convergence["feasible_count"],
        "ga_diversity": ga_convergence["genetic_diversity"]
    }
    
    return sa_best_feasible, sa_feasible_solutions, elapsed_time, num_processes, convergence_data

def plot_hybrid_convergence(convergence_data, elapsed_time, num_procs):
    """绘制混合优化的收敛图"""
    # 确保目录存在
    os.makedirs("results", exist_ok=True)
    
    # 1. 组合收敛曲线
    plt.figure(figsize=(12, 8))
    
    # GA阶段
    ga_generations = len(convergence_data["ga_min_cost"])
    ga_x = range(ga_generations)
    plt.plot(ga_x, convergence_data["ga_min_cost"], 'b-', label='GA Min Cost', linewidth=2)
    
    # SA阶段
    sa_iterations = len(convergence_data["sa_min_cost"])
    sa_x = range(ga_generations, ga_generations + sa_iterations)
    plt.plot(sa_x, convergence_data["sa_min_cost"], 'r-', label='SA Min Cost', linewidth=2)
    
    # 标记阶段转换
    plt.axvline(x=ga_generations-1, color='g', linestyle='--', linewidth=2, label='GA to SA Transition')
    
    plt.xlabel('Evaluation Sequence', fontsize=12)
    plt.ylabel('Total Cost', fontsize=12)
    plt.title(f'Hybrid GA-SA Optimization Convergence\n(Time: {elapsed_time:.2f}s, Processes: {num_procs})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results/hybrid_convergence.png', dpi=300)
    print("Saved: results/hybrid_convergence.png")
    plt.close()
    
    # 2. 可行解数量变化
    plt.figure(figsize=(12, 8))
    
    # GA阶段
    plt.plot(ga_x, convergence_data["ga_feasible"], 'g-', label='GA Feasible', linewidth=2)
    
    # SA阶段
    plt.plot(sa_x, convergence_data["sa_feasible"], 'm-', label='SA Feasible', linewidth=2)
    
    plt.axvline(x=ga_generations-1, color='b', linestyle='--', linewidth=2, label='Phase Transition')
    plt.xlabel('Evaluation Sequence', fontsize=12)
    plt.ylabel('Feasible Solutions', fontsize=12)
    plt.title('Feasible Solutions Evolution', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results/hybrid_feasible_count.png', dpi=300)
    print("Saved: results/hybrid_feasible_count.png")
    plt.close()
    
    # 3. 多样性变化
    if "ga_diversity" in convergence_data:
        plt.figure(figsize=(12, 8))
        plt.plot(ga_x, convergence_data["ga_diversity"], 'c-', label='GA Diversity', linewidth=2)
        plt.axvline(x=ga_generations-1, color='r', linestyle='--', linewidth=2, label='Phase Transition')
        plt.xlabel('GA Generation', fontsize=12)
        plt.ylabel('Genetic Diversity', fontsize=12)
        plt.title('Genetic Diversity in GA Phase', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('results/hybrid_diversity.png', dpi=300)
        print("Saved: results/hybrid_diversity.png")
        plt.close()

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 直接使用128个进程
    num_processes = 128
    
    print("="*80)
    print(f"Starting Hybrid GA-SA Optimization with {num_processes} processes")
    print(f"Application: {application}")
    print("="*80)
    
    try:
        # 运行混合优化
        start_time = time.time()
        best_feasible, feasible_solutions, elapsed_time, num_procs, convergence_data = hybrid_ga_sa_optimization(num_processes)
        total_time = time.time() - start_time
        
        # 绘制混合收敛图
        plot_hybrid_convergence(convergence_data, total_time, num_procs)
        
        # 提取最佳解决方案
        best_solution = extract_solution_info(best_feasible)
        
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
        
        # 写入结果文件
        with open("results/hybrid_optimization_results.txt", "w") as f:
            f.write("Hybrid GA-SA Optimization Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Optimization Time: {total_time:.2f} seconds\n")
            f.write(f"Processes Used: {num_procs}\n")
            f.write(f"Application: {application}\n")
            f.write(f"Budget: {GLOBAL_CONFIG['budget']}\n")
            f.write(f"Best Total Cost: {best_solution['total_cost']:.2f}\n\n")
            
            # 最佳解决方案
            f.write("Optimal Configuration:\n")
            f.write("=" * 80 + "\n")
            f.write(f"Selected Modules: {best_solution['modules']}\n\n")
            f.write("Global Parameters:\n")
            f.write(f"  warning_energy: {best_solution['warning_energy']:.2f}%\n")
            f.write(f"  preventive_check_days: {best_solution['preventive_check_days']} days\n")
            
            if best_solution['frequency_heartbeat'] is not None:
                f.write("\nModule Parameters (Heartbeat):\n")
                f.write(f"  frequency_heartbeat: {best_solution['frequency_heartbeat']} seconds\n")
                f.write(f"  heartbeat_loss_threshold: {best_solution['heartbeat_loss_threshold']}\n")
            
            f.write("\nCost Breakdown:\n")
            f.write("=" * 80 + "\n")
            f.write(f"Base Cost:         {best_solution['base_cost']:.2f}\n")
            f.write(f"Module Cost:       {best_solution['module_cost']:.2f}\n")
            f.write(f"Check Cost:        {best_solution['check_cost']:.2f}\n")
            f.write(f"Fault Cost:        {best_solution['fault_cost']:.2f}\n")
            f.write(f"Data Loss Cost:    {best_solution['data_loss_cost']:.2f}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Cost:        {best_solution['total_cost']:.2f}\n\n")
            
            # 其他可行解
            if feasible_solutions:
                solutions_info = []
                for sol in feasible_solutions:
                    sol_info = extract_solution_info(sol["result"])
                    solutions_info.append(sol_info)
                
                solutions_info.sort(key=solution_info_sort_key)
                
                f.write(f"Alternative Feasible Solutions ({len(feasible_solutions)} found):\n")
                f.write("=" * 80 + "\n")
                for i, sol in enumerate(solutions_info[:5]):  # 只显示前5个
                    f.write(f"\nSolution #{i+1} (Total Cost: {sol['total_cost']:.2f}):\n")
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
            else:
                f.write("No alternative feasible solutions found.\n")
        
        # 打印最佳解决方案
        print("\n" + "="*80)
        print("Optimal Configuration Found:")
        print("="*80)
        print(f"Selected Modules: {best_solution['modules']}")
        print("\nGlobal Parameters:")
        print(f"  Warning Energy: {best_solution['warning_energy']:.2f}%")
        print(f"  Preventive Check Days: {best_solution['preventive_check_days']} days")
        
        if best_solution['frequency_heartbeat'] is not None:
            print("\nModule Parameters (Heartbeat):")
            print(f"  Frequency: {best_solution['frequency_heartbeat']} seconds")
            print(f"  Loss Threshold: {best_solution['heartbeat_loss_threshold']}")
        
        print("\nCost Breakdown:")
        print(f"  Base Cost:         {best_solution['base_cost']:.2f}")
        print(f"  Module Cost:       {best_solution['module_cost']:.2f}")
        print(f"  Check Cost:        {best_solution['check_cost']:.2f}")
        print(f"  Fault Cost:        {best_solution['fault_cost']:.2f}")
        print(f"  Data Loss Cost:    {best_solution['data_loss_cost']:.2f}")
        print("-"*40)
        print(f"Total Cost:        {best_solution['total_cost']:.2f}")
        print(f"Budget:            {best_solution['budget']}")
        
        # 打印执行时间
        print("\n" + "="*80)
        print(f"Hybrid optimization completed in {total_time:.2f} seconds")
        print(f"Using {num_procs} processes")
        print(f"Feasible solutions found: {len(feasible_solutions) if feasible_solutions else 0}")
        print("Results saved to 'results' directory")
        print("="*80)
    
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

        sys.exit(1)
