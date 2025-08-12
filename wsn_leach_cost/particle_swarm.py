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
import sys
import itertools
import math
import functools
from deap import base, creator

# 使用非交互式后端避免GUI问题
matplotlib.use('Agg')

# 设置模块成本
set_modules_cost()

# 获取所有模块名称
all_modules = list(MODULES.keys())
print(f"Total modules available: {len(all_modules)}")
application = "animal_room"  #  animal_room, electricity_meter

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
    selected_modules1 = [all_modules[i] for i in range(n_modules) if ind1[i] >= 0.5]
    selected_modules2 = [all_modules[i] for i in range(n_modules) if ind2[i] >= 0.5]

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
    deviation1 = 0
    deviation2 = 0
    
    # 总是存在的参数
    deviation1 += abs(param1[0] - DEFAULT_PARAM_VALUES['warning_energy'])
    deviation1 += abs(param1[1] - DEFAULT_PARAM_VALUES['preventive_check_days'])
    deviation2 += abs(param2[0] - DEFAULT_PARAM_VALUES['warning_energy'])
    deviation2 += abs(param2[1] - DEFAULT_PARAM_VALUES['preventive_check_days'])
    
    # 只有当heartbeat模块被选中时才计算相关参数
    if "heartbeat" in selected_modules1:
        deviation1 += abs(param1[2] - DEFAULT_PARAM_VALUES['frequency_heartbeat'])
        deviation1 += abs(param1[3] - DEFAULT_PARAM_VALUES['heartbeat_loss_threshold'])
    if "heartbeat" in selected_modules2:
        deviation2 += abs(param2[2] - DEFAULT_PARAM_VALUES['frequency_heartbeat'])
        deviation2 += abs(param2[3] - DEFAULT_PARAM_VALUES['heartbeat_loss_threshold'])

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
    param_deviation = 0
    param_deviation += abs(param_values[0] - DEFAULT_PARAM_VALUES['warning_energy'])
    param_deviation += abs(param_values[1] - DEFAULT_PARAM_VALUES['preventive_check_days'])
    if "heartbeat" in sol['modules']:
        param_deviation += abs(param_values[2] - DEFAULT_PARAM_VALUES['frequency_heartbeat'])
        param_deviation += abs(param_values[3] - DEFAULT_PARAM_VALUES['heartbeat_loss_threshold'])

    return (sol['total_cost'], module_cost, param_deviation)

# 定义适应度函数（单目标：最小化总成本）
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 单目标最小化

# 定义个体（二进制列表表示模块是否被选择 + 仿真参数）
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate_individual(individual):
    """评估个体的适应度"""
    n_modules = len(all_modules)
    
    try:
        # 提取模块选择部分 - 使用0.5作为阈值将连续值转换为二进制
        selected_modules = [all_modules[i] for i in range(n_modules) if individual[i] >= 0.5]
        
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
        
        # 创建成本分解字典
        cost_breakdown = {
            "total_cost": total_cost,
            "base_cost": base_cost,
            "module_cost": module_cost,
            "check_cost": check_cost,  # 新增检查成本
            "fault_cost": fault_cost,
            "data_loss_cost": data_loss_cost
        }
        
        # 保持硬约束，预算超支返回实际值但标记为不可行
        return (total_cost,), cost_breakdown, budget_exceed
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        return (float('inf'),), {
            "total_cost": float('inf'),
            "base_cost": 0,
            "module_cost": 0,
            "check_cost": 0,
            "fault_cost": 0,
            "data_loss_cost": 0
        }, float('inf')  # 标记为严重不可行

def calculate_genetic_diversity(population_positions):
    """计算种群的基因多样性（基于汉明距离）"""
    if len(population_positions) < 2:
        return 0.0
    
    n_modules = len(all_modules)
    diversity = 0.0
    count = 0
    
    # 计算所有个体两两之间的汉明距离（仅模块部分）
    for i, j in itertools.combinations(range(len(population_positions)), 2):
        # 将连续值转换为二进制
        ind1 = [1 if x >= 0.5 else 0 for x in population_positions[i][:n_modules]]
        ind2 = [1 if x >= 0.5 else 0 for x in population_positions[j][:n_modules]]
        
        # 计算汉明距离（不同基因的数量）
        hamming_dist = sum(g1 != g2 for g1, g2 in zip(ind1, ind2))
        diversity += hamming_dist
        count += 1
    
    # 返回平均汉明距离
    return diversity / count if count > 0 else 0.0

class Particle:
    """粒子群优化中的粒子类"""
    def __init__(self, position):
        self.position = position
        self.velocity = [0.0] * len(position)
        self.best_position = position[:]
        self.best_fitness = float('inf')
        self.fitness = float('inf')
        self.cost_breakdown = None  # 存储成本分解信息
        self.budget_exceed = float('inf')  # 存储预算超支值
    
    def update_position(self, bounds):
        """更新粒子位置，考虑边界约束"""
        for i in range(len(self.position)):
            # 更新位置
            self.position[i] += self.velocity[i]
            
            # 应用边界约束
            if i < len(bounds):
                min_bound, max_bound = bounds[i]
                self.position[i] = max(min_bound, min(max_bound, self.position[i]))
    
    def update_best(self):
        """更新粒子的最佳位置"""
        # 优先选择预算不超支的解
        if self.budget_exceed == 0 and self.best_budget_exceed > 0:
            self.best_position = self.position[:]
            self.best_fitness = self.fitness
            self.best_budget_exceed = self.budget_exceed
        elif self.budget_exceed == 0 and self.best_budget_exceed == 0:
            # 两者都不超支，选择总成本更小的
            if self.fitness < self.best_fitness:
                self.best_position = self.position[:]
                self.best_fitness = self.fitness
                self.best_budget_exceed = self.budget_exceed
        elif self.budget_exceed > 0 and self.best_budget_exceed > 0:
            # 两者都超支，选择超支更小的
            if self.budget_exceed < self.best_budget_exceed:
                self.best_position = self.position[:]
                self.best_fitness = self.fitness
                self.best_budget_exceed = self.budget_exceed
        else:
            # 当前解超支但历史解不超支，不更新
            pass

def particle_swarm_optimization(num_processes=None):
    """执行粒子群优化（预算约束下的单目标优化）"""
    # 参数范围
    n_modules = len(all_modules)
    
    # 定义参数范围
    warning_energy_range = (0.0, 50.0)
    check_day_min = round(GLOBAL_CONFIG["frequency_sampling"]/(60*60*24))
    if check_day_min <= 0:
        check_day_min = 1        
    preventive_check_days_range = (check_day_min, 180)
    frequency_heartbeat_max = GLOBAL_CONFIG["frequency_sampling"]
    frequency_heartbeat_min = max(60, frequency_heartbeat_max / 60)  # 确保最小值合理
    frequency_heartbeat_range = (frequency_heartbeat_min, frequency_heartbeat_max)
    heartbeat_loss_threshold_range = (3, 15)
    
    # 定义搜索空间边界
    bounds = []
    # 模块选择部分（二进制，但PSO中我们使用连续值并通过sigmoid转换）
    for _ in range(n_modules):
        bounds.append((0, 1))  # 二进制变量边界
    
    # 仿真参数部分
    bounds.append(warning_energy_range)
    bounds.append(preventive_check_days_range)
    bounds.append(frequency_heartbeat_range)
    bounds.append(heartbeat_loss_threshold_range)
    
    # 粒子群参数
    pop_size = 50
    ngen = 300
    base_w = 0.729
    c1 = 1.49445
    c2 = 1.49445
    base_v_max = 0.2
    
    # 创建初始种群
    def create_particle():
        # 模块选择部分（二进制）
        modules_part = [random.uniform(0, 1) for _ in range(n_modules)]
        
        # 仿真参数部分
        params_part = [
            random.uniform(*warning_energy_range),
            random.uniform(*preventive_check_days_range),
            random.uniform(*frequency_heartbeat_range),
            random.uniform(*heartbeat_loss_threshold_range)
        ]
        
        particle = Particle(modules_part + params_part)
        particle.best_budget_exceed = float('inf')  # 初始化历史最佳预算超支值
        return particle
    
    population = [create_particle() for _ in range(pop_size)]
    
    # 设置多进程评估
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 评估初始种群
    print(f"Evaluating initial population (using {num_processes} processes)...")
    positions = [p.position for p in population]
    results = pool.map(evaluate_individual, positions)
    
    # 更新粒子适应度、成本分解和预算超支值
    for i, p in enumerate(population):
        fitness, cost_breakdown, budget_exceed = results[i]
        p.fitness = fitness[0]
        p.cost_breakdown = cost_breakdown
        p.budget_exceed = budget_exceed
        p.update_best()
    
    # 全局最佳粒子
    global_best_particle = min(
        population,
        key=lambda p: (p.budget_exceed, p.fitness)  # 优先选不超支的解
    )
    global_best_position = global_best_particle.best_position[:]
    global_best_fitness = global_best_particle.best_fitness
    global_best_budget_exceed = global_best_particle.best_budget_exceed
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [],
        'avg_total_cost': [],
        'feasible_count': [],
        'diversity': [],
        'genetic_diversity': [],
        'inertia_weight': [],
        'velocity_max': []
    }
    
    # 记录初始代
    total_costs = [p.fitness for p in population]
    feasible_count = sum(1 for p in population if p.budget_exceed == 0)
    min_cost = min(total_costs)
    
    convergence_data['min_total_cost'].append(min_cost)
    convergence_data['avg_total_cost'].append(
        sum(c for c in total_costs) / pop_size)
    convergence_data['feasible_count'].append(feasible_count)
    
    # 计算初始多样性和基因多样性
    feasible_costs = [p.fitness for p in population if p.budget_exceed == 0]
    diversity = np.std(feasible_costs) if feasible_costs else 0
    genetic_diversity = calculate_genetic_diversity(positions)
    
    convergence_data['diversity'].append(diversity)
    convergence_data['genetic_diversity'].append(genetic_diversity)
    convergence_data['inertia_weight'].append(base_w)
    convergence_data['velocity_max'].append(base_v_max)
    
    # 运行PSO优化
    print("Starting PSO optimization...")
    start_time = time.time()
    
    # 添加停滞检测和响应机制
    stagnation_count = 0
    stagnation_threshold = 15  # 连续15代无改进触发响应
    last_improvement = min_cost
    
    # 自适应参数
    w = base_w
    v_max = base_v_max
    
    for gen in range(ngen):
        # 自适应调整参数（基于停滞情况）
        if stagnation_count > 0:
            # 增加探索性：降低惯性权重，增加最大速度
            w = max(0.4, base_w * (1 - stagnation_count * 0.05))
            v_max = min(0.5, base_v_max * (1 + stagnation_count * 0.1))
        else:
            w = base_w
            v_max = base_v_max
        
        # 更新每个粒子
        for p in population:
            # 更新速度
            for i in range(len(p.velocity)):
                r1 = random.random()
                r2 = random.random()
                
                # 速度更新公式
                cognitive = c1 * r1 * (p.best_position[i] - p.position[i])
                social = c2 * r2 * (global_best_position[i] - p.position[i])
                p.velocity[i] = w * p.velocity[i] + cognitive + social
                
                # 限制速度范围
                min_bound, max_bound = bounds[i]
                range_size = max_bound - min_bound
                p.velocity[i] = max(-v_max * range_size, min(v_max * range_size, p.velocity[i]))
            
            # 更新位置
            p.update_position(bounds)
        
        # 评估所有粒子
        positions = [p.position for p in population]
        results = pool.map(evaluate_individual, positions)
        
        # 更新粒子适应度、成本分解和预算超支值
        for i, p in enumerate(population):
            fitness, cost_breakdown, budget_exceed = results[i]
            p.fitness = fitness[0]
            p.cost_breakdown = cost_breakdown
            p.budget_exceed = budget_exceed
            p.update_best()
            
            # 更新全局最佳（使用全局比较）
            # 优先选择预算不超支的解
            if p.best_budget_exceed < global_best_budget_exceed:
                global_best_particle = p
                global_best_position = p.best_position[:]
                global_best_fitness = p.best_fitness
                global_best_budget_exceed = p.best_budget_exceed
            elif p.best_budget_exceed == global_best_budget_exceed:
                if p.best_fitness < global_best_fitness:
                    global_best_particle = p
                    global_best_position = p.best_position[:]
                    global_best_fitness = p.best_fitness
                    global_best_budget_exceed = p.best_budget_exceed
        
        # 记录当前代数据
        total_costs = [p.fitness for p in population]
        feasible_count = sum(1 for p in population if p.budget_exceed == 0)
        current_min = min(total_costs)
        
        # 更新停滞计数器
        if current_min < last_improvement - 0.001:
            stagnation_count = 0
            last_improvement = current_min
        else:
            stagnation_count += 1
        
        convergence_data['min_total_cost'].append(current_min)
        convergence_data['avg_total_cost'].append(
            sum(c for c in total_costs) / pop_size)
        convergence_data['feasible_count'].append(feasible_count)
        
        # 计算并记录种群多样性和基因多样性
        feasible_costs = [p.fitness for p in population if p.budget_exceed == 0]
        diversity = np.std(feasible_costs) if feasible_costs else 0
        genetic_diversity = calculate_genetic_diversity(positions)
        
        convergence_data['diversity'].append(diversity)
        convergence_data['genetic_diversity'].append(genetic_diversity)
        convergence_data['inertia_weight'].append(w)
        convergence_data['velocity_max'].append(v_max)
        
        # 打印进度
        print(f"Gen {gen+1}/{ngen}: Min Cost={current_min:.2f}, "
              f"Feasible={feasible_count}/{pop_size}, "
              f"Diversity={diversity:.1f}, "
              f"GeneticDiv={genetic_diversity:.1f}, "
              f"Stagnation={stagnation_count}/{stagnation_threshold}, "
              f"w={w:.3f}, v_max={v_max:.3f}")
        
        # 早停机制
        if stagnation_count >= stagnation_threshold:
            print(f"Early stopping at generation {gen+1} due to convergence.")
            break
        
        # 智能重启机制（基于停滞程度）
        if stagnation_count > stagnation_threshold // 2:
            # 部分重启：替换30%最差粒子
            population.sort(key=lambda p: (p.budget_exceed, p.fitness))
            num_replace = max(5, int(0.3 * pop_size))
            
            # 保留历史最优解
            best_to_keep = population[:-num_replace]
            
            # 生成新粒子
            new_particles = [create_particle() for _ in range(num_replace)]
            
            # 评估新粒子
            new_positions = [p.position for p in new_particles]
            new_results = pool.map(evaluate_individual, new_positions)
            
            for i, p in enumerate(new_particles):
                fitness, cost_breakdown, budget_exceed = new_results[i]
                p.fitness = fitness[0]
                p.cost_breakdown = cost_breakdown
                p.budget_exceed = budget_exceed
                p.update_best()
            
            population = best_to_keep + new_particles
            stagnation_count = 0  # 重置停滞计数器
            print(f"Gen {gen+1}: Partial restart ({num_replace} new particles)")
            
            # 更新全局最佳
            global_best_particle = min(
                population,
                key=lambda p: (p.budget_exceed, p.fitness)
            )
            global_best_position = global_best_particle.best_position[:]
            global_best_fitness = global_best_particle.best_fitness
            global_best_budget_exceed = global_best_particle.best_budget_exceed
    
    elapsed = time.time() - start_time
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 从最终种群选择真正的最佳可行解
    feasible_population = [p for p in population if p.budget_exceed == 0]
    if feasible_population:
        true_best = min(feasible_population, key=lambda p: p.fitness)
    else:
        # 如果没有可行解，选择超支最小的解
        true_best = min(population, key=lambda p: p.budget_exceed)
    
    # 创建最佳个体
    best_individual = creator.Individual(true_best.position)
    best_individual.fitness.values = (true_best.fitness,)
    best_individual.cost_breakdown = true_best.cost_breakdown
    
    # 创建最终种群并设置适应度值
    final_population = []
    for p in population:
        ind = creator.Individual(p.position)
        ind.fitness.values = (p.fitness,)
        ind.cost_breakdown = p.cost_breakdown
        final_population.append(ind)
    
    return best_individual, final_population, elapsed, num_processes, convergence_data

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
    plt.savefig('pso_convergence_total_cost.png')
    print("Saved: pso_convergence_total_cost.png")
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
    plt.savefig('pso_convergence_feasible_count.png')
    print("Saved: pso_convergence_feasible_count.png")
    plt.close()

    # 3. 种群多样性
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_data['diversity'], 'c-', label='Fitness Diversity (Std Dev)')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Fitness Diversity Over Generations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pso_convergence_diversity.png')
    print("Saved: pso_convergence_diversity.png")
    plt.close()

    # 4. 基因多样性
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_data['genetic_diversity'], 'y-', label='Genetic Diversity (Avg Hamming Dist)')
    plt.xlabel('Generation')
    plt.ylabel('Genetic Diversity')
    plt.title('Genetic Diversity Over Generations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pso_convergence_genetic_diversity.png')
    print("Saved: pso_convergence_genetic_diversity.png")
    plt.close()
    
    # 5. 自适应参数变化
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_data['inertia_weight'], 'm-', label='Inertia Weight (w)')
    plt.plot(convergence_data['velocity_max'], 'k-', label='Max Velocity (v_max)')
    plt.xlabel('Generation')
    plt.ylabel('Parameter Value')
    plt.title('Adaptive Parameters Over Generations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pso_convergence_parameters.png')
    print("Saved: pso_convergence_parameters.png")
    plt.close()

def extract_solution_info(individual):
    """从个体中提取解决方案信息"""
    n_modules = len(all_modules)
    
    try:
        # 提取模块选择 - 使用0.5作为阈值将连续值转换为二进制
        selected_modules = [all_modules[i] for i in range(n_modules) if individual[i] >= 0.5]
        
        # 提取仿真参数
        param_start_idx = n_modules
        warning_energy = individual[param_start_idx]
        preventive_check_days = int(round(individual[param_start_idx + 1]))
        frequency_heartbeat = int(round(individual[param_start_idx + 2])) if "heartbeat" in selected_modules else None
        heartbeat_loss_threshold = int(round(individual[param_start_idx + 3])) if "heartbeat" in selected_modules else None
        
        # 获取适应度值和成本分解
        total_cost = individual.fitness.values[0] if hasattr(individual, 'fitness') else float('inf')
        
        # 使用存储的成本分解信息
        if hasattr(individual, 'cost_breakdown'):
            cost_info = individual.cost_breakdown
        else:
            cost_info = {
                "total_cost": total_cost,
                "base_cost": 0,
                "module_cost": 0,
                "check_cost": 0,
                "fault_cost": 0,
                "data_loss_cost": 0
            }
        
        return {
            "modules": selected_modules,
            "warning_energy": warning_energy,
            "preventive_check_days": preventive_check_days,
            "frequency_heartbeat": frequency_heartbeat,
            "heartbeat_loss_threshold": heartbeat_loss_threshold,
            "total_cost": cost_info["total_cost"],
            "base_cost": cost_info["base_cost"],
            "module_cost": cost_info["module_cost"],
            "check_cost": cost_info["check_cost"],  # 新增检查成本
            "fault_cost": cost_info["fault_cost"],
            "data_loss_cost": cost_info["data_loss_cost"],
            "budget": GLOBAL_CONFIG["budget"]
        }
    except Exception as e:
        print(f"Error extracting solution info: {e}")
        return {
            "modules": [],
            "warning_energy": 0.0,
            "preventive_check_days": 1,
            "frequency_heartbeat": None,
            "heartbeat_loss_threshold": None,
            "total_cost": float('inf'),
            "base_cost": 0,
            "module_cost": 0,
            "check_cost": 0,
            "fault_cost": 0,
            "data_loss_cost": 0,
            "budget": GLOBAL_CONFIG["budget"]
        }

if __name__ == "__main__":
    # 运行粒子群优化
    print("Starting multi-process PSO optimization...")
    
    try:
        best_individual, population, elapsed_time, num_procs, convergence_data = particle_swarm_optimization()
        
        # 绘制收敛图
        plot_convergence(convergence_data, elapsed_time, num_procs)
        
        # 提取最佳解决方案
        best_solution = extract_solution_info(best_individual)
        
        # 写入结果文件
        with open("pso_optimization_results.txt", "w") as f:
            f.write("Particle Swarm Optimization Results (Budget-Constrained)\n")
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
            f.write(f"  Check Cost:     {best_solution['check_cost']:.2f}\n")  # 新增检查成本
            f.write(f"  Fault Cost:     {best_solution['fault_cost']:.2f}\n")
            f.write(f"  Data Loss Cost: {best_solution['data_loss_cost']:.2f}\n")
            f.write(f"  Total Cost:     {best_solution['total_cost']:.2f}\n")
            f.write("-" * 80 + "\n\n")
            
            # 其他可行解（最多前10个）
            feasible_solutions = []
            for ind in population:
                solution = extract_solution_info(ind)
                # 只考虑预算不超支的解
                total_budget_cost = solution['base_cost'] + solution['module_cost'] + solution['check_cost']
                if total_budget_cost <= solution['budget']:
                    feasible_solutions.append(solution)
            
            # 按总成本排序
            feasible_solutions.sort(key=lambda sol: sol['total_cost'])
            
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
                f.write("  Cost Breakdown:\n")
                f.write(f"    Base Cost:      {sol['base_cost']:.2f}\n")
                f.write(f"    Module Cost:    {sol['module_cost']:.2f}\n")
                f.write(f"    Check Cost:     {sol['check_cost']:.2f}\n")  # 新增检查成本
                f.write(f"    Fault Cost:     {sol['fault_cost']:.2f}\n")
                f.write(f"    Data Loss Cost: {sol['data_loss_cost']:.2f}\n")
                f.write(f"    Total Cost:     {sol['total_cost']:.2f}\n")
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
        print(f"    Check Cost:     {best_solution['check_cost']:.2f}")  # 新增检查成本
        print(f"    Fault Cost:     {best_solution['fault_cost']:.2f}")
        print(f"    Data Loss Cost: {best_solution['data_loss_cost']:.2f}")
        print(f"    Total Cost:     {best_solution['total_cost']:.2f}")
        print(f"  Budget: {best_solution['budget']}")
                
        # 打印执行时间
        print(f"\nPSO optimization completed! Time: {elapsed_time:.2f} seconds | Processes: {num_procs}")
        print(f"Feasible solutions found: {len(feasible_solutions)}")
    
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

        sys.exit(1)
