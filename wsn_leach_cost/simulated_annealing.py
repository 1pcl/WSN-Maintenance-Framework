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
from deap import base, creator
import functools

# 使用非交互式后端避免GUI问题
matplotlib.use('Agg')

# 设置模块成本
set_modules_cost()

# 获取所有模块名称
all_modules = list(MODULES.keys())
print(f"Total modules available: {len(all_modules)}")
application = "animal_room"  # parking_lot, animal_room, None，electricity_meter

# 定义适应度函数（单目标：最小化总成本）
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 单目标最小化

# 定义个体（二进制列表表示模块是否被选择 + 仿真参数）
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

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
        "check_cost": check_cost,  # 包含检查代价
        "fault_cost": fault_cost,
        "data_loss_cost": data_loss_cost,
        "feasible": budget_exceed <= 0,
        "individual": individual,  # 保留原始个体
        "selected_modules": selected_modules,
        "simulation_params": simulation_params
    }
    
    return result

def solution_sort_key(sol_result):
    """解决方案排序函数，使用完整结果字典"""
    # 处理None值（当找不到匹配的解决方案时）
    if sol_result is None:
        return (float('inf'), float('inf'), float('inf'))  # 返回极大值确保排在最后
    
    # 计算模块代价总和
    module_cost = sum(MODULES[m]['cost'] for m in sol_result["selected_modules"])

    # 参数偏离度计算
    param_keys = list(DEFAULT_PARAM_VALUES.keys())
    params = sol_result["simulation_params"]
    param_values = [
        params["warning_energy"],
        params["preventive_check_days"],
        params["frequency_heartbeat"] if params["frequency_heartbeat"] is not None else DEFAULT_PARAM_VALUES["frequency_heartbeat"],
        params["heartbeat_loss_threshold"] if params["heartbeat_loss_threshold"] is not None else DEFAULT_PARAM_VALUES["heartbeat_loss_threshold"]
    ]
    param_deviation = sum(abs(p - DEFAULT_PARAM_VALUES[k]) for p, k in zip(param_values, param_keys))

    return (sol_result["total_cost"], module_cost, param_deviation)

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

def neighbor_function(current_solution, temperature, initial_temp, stagnation_count=0):
    """生成邻域解，支持自适应扰动和停滞响应"""
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
    
    # 基于停滞计数增加扰动强度
    mutation_intensity = 1.0 + stagnation_count * 0.2
    
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
        # 增加扰动强度：基于停滞计数增加翻转的模块数量
        num_flips = min(n_modules, 1 + stagnation_count // 5)
        for _ in range(num_flips):
            idx = random.randint(0, n_modules - 1)
            neighbor[idx] = 1 - neighbor[idx]
    
    # 参数部分：调整warning_energy
    elif perturbation_type == "adjust_warning_energy":
        param_idx = n_modules
        # 增强扰动：基于停滞计数增加标准差
        std_dev = 5 * perturbation_factor * mutation_intensity
        perturbation = random.gauss(0, std_dev)
        neighbor[param_idx] = max(warning_energy_range[0], 
                                 min(warning_energy_range[1], 
                                 neighbor[param_idx] + perturbation))
    
    # 参数部分：调整preventive_check_days
    elif perturbation_type == "adjust_preventive_days":
        param_idx = n_modules + 1
        # 增强扰动：基于停滞计数增加扰动范围
        perturbation_range = int(10 * mutation_intensity)
        perturbation = random.randint(-perturbation_range, perturbation_range)
        neighbor[param_idx] = max(preventive_check_days_range[0], 
                                 min(preventive_check_days_range[1], 
                                 neighbor[param_idx] + perturbation))
    
    # 参数部分：调整frequency_heartbeat
    elif perturbation_type == "adjust_heartbeat_freq":
        param_idx = n_modules + 2
        # 增强扰动：基于停滞计数增加乘法扰动范围
        min_factor = max(0.5, 0.8 - stagnation_count * 0.05)
        max_factor = min(1.5, 1.2 + stagnation_count * 0.05)
        perturbation = random.uniform(min_factor, max_factor)  # 乘法扰动
        new_val = neighbor[param_idx] * perturbation
        neighbor[param_idx] = max(frequency_heartbeat_range[0], 
                                 min(frequency_heartbeat_range[1], 
                                 new_val))
    
    # 参数部分：调整heartbeat_loss_threshold
    elif perturbation_type == "adjust_heartbeat_threshold":
        param_idx = n_modules + 3
        # 增强扰动：基于停滞计数增加扰动范围
        perturbation_range = int(2 * mutation_intensity)
        perturbation = random.randint(-perturbation_range, perturbation_range)
        neighbor[param_idx] = max(heartbeat_loss_threshold_range[0], 
                                 min(heartbeat_loss_threshold_range[1], 
                                 neighbor[param_idx] + perturbation))
    
    return neighbor

def simulated_annealing_optimization(num_processes=None):
    """执行模拟退火优化（预算约束下的单目标优化）"""
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
    
    # 模拟退火参数
    initial_temp = 1000.0
    final_temp = 0.1
    cooling_rate = 0.91  # 调整冷却率以适应更少的迭代次数
    max_iter = 117  # 15000次评估 / 128核 ≈ 117次迭代
    max_stagnation = 23  # 最大迭代次数的20%
    reheat_threshold = 11  # 重新加热阈值
    
    # 创建初始解
    current_solution = create_individual()
    
    # ✅ 评估初始解（单次评估）
    current_result = evaluate_individual(current_solution)
    current_cost = current_result["fitness"][0]
    
    # 记录最佳解
    best_solution = creator.Individual(current_solution[:])
    best_cost = current_cost
    best_result = current_result
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [best_cost],
        'current_cost': [current_cost],
        'temperature': [initial_temp],
        'feasible_count': [1 if current_cost != float('inf') else 0],
        'stagnation_count': 0
    }
    
    # 记录历史可行解和已见解
    feasible_solutions = []
    seen_solutions = set()
    
    if current_result["feasible"]:
        # 将个体转换为元组以便哈希
        ind_tuple = tuple(current_solution)
        seen_solutions.add(ind_tuple)
        feasible_solutions.append({
            "result": current_result,
            "cost": current_cost
        })
    
    print("Starting simulated annealing optimization...")
    print(f"Using {num_processes} processes for parallel evaluation")
    print(f"Target evaluations: ~15000 | Max iterations: {max_iter}")
    start_time = time.time()
    
    # 初始温度
    temperature = initial_temp
    iteration = 0
    stagnation_count = 0
    
    # 设置多进程评估
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 主循环
    while temperature > final_temp and iteration < max_iter and stagnation_count < max_stagnation:
        # 生成多个邻域解（数量等于进程数）
        neighbors = []
        for _ in range(num_processes):
            neighbor = neighbor_function(current_solution, temperature, initial_temp, stagnation_count)
            neighbors.append(neighbor)
        
        # 并行评估所有邻域解
        results = pool.map(evaluate_individual, neighbors)
        
        # 处理结果并更新可行解列表
        for res in results:
            if res["feasible"]:
                # 检查是否已见过该解
                ind_tuple = tuple(res["individual"])
                if ind_tuple not in seen_solutions:
                    seen_solutions.add(ind_tuple)
                    feasible_solutions.append({
                        "result": res,
                        "cost": res["fitness"][0]
                    })
        
        # 从邻域解中找出最佳候选解
        candidate_result = min(results, key=lambda res: res["fitness"][0])
        candidate_solution = creator.Individual(candidate_result["individual"])
        candidate_cost = candidate_result["fitness"][0]
        
        # 计算成本差
        cost_difference = candidate_cost - current_cost
        
        # Metropolis准则：决定是否接受新解
        accept = False
        if cost_difference < 0:
            # 新解更好，总是接受
            accept = True
        elif temperature > 0:
            # 以一定概率接受较差的解（避免局部最优）
            acceptance_prob = math.exp(-cost_difference / temperature)
            if random.random() < acceptance_prob:
                accept = True
        
        # 如果接受新解，更新当前状态
        if accept:
            current_solution = creator.Individual(candidate_solution[:])
            current_cost = candidate_cost
            current_result = candidate_result
            
            # 更新最佳解
            if current_cost < best_cost:
                best_solution = creator.Individual(candidate_solution[:])
                best_cost = current_cost
                best_result = candidate_result
                stagnation_count = 0  # 重置停滞计数
                print(f"Iter {iteration}: New best cost = {best_cost:.2f}, Temp = {temperature:.2f}")
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1
        
        # # 重新加热机制：当停滞计数达到阈值时增加温度
        # if stagnation_count >= reheat_threshold:
        #     # 重新加热：将温度提高到初始温度的50%
        #     temperature = max(temperature, initial_temp * 0.5)
        #     stagnation_count = 0
        #     print(f"Iter {iteration}: Reheating to {temperature:.2f}")
        
        # 重新加热机制：当停滞计数达到阈值时增加温度
        if stagnation_count >= reheat_threshold:
            # 设定温度升温倍数，比如升温10%
            reheating_factor = 1.1

            # 计算新的温度（温和升温）
            new_temperature = temperature * reheating_factor

            # 设置温度下限，防止升温过低
            min_temperature = initial_temp * 0.1

            # 最终温度取升温后的温度和下限中较大值
            temperature = max(new_temperature, min_temperature)

            stagnation_count = 0
            print(f"Iter {iteration}: Reheating to {temperature:.2f}")

        # 记录收敛数据
        convergence_data['min_total_cost'].append(best_cost)
        convergence_data['current_cost'].append(current_cost)
        convergence_data['temperature'].append(temperature)
        convergence_data['feasible_count'].append(len(feasible_solutions))
        
        # 降温
        temperature *= cooling_rate
        iteration += 1
        
        # 打印进度
        print(f"Iter {iteration}: Best cost = {best_cost:.2f}, Current cost = {current_cost:.2f}, "
              f"Temp = {temperature:.2f}, Stagnation = {stagnation_count}, "
              f"Feasible = {len(feasible_solutions)}, Evals = {iteration * num_processes}")
    
    elapsed = time.time() - start_time
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 从可行解中选择最佳解
    if feasible_solutions:
        # 使用多标准排序
        feasible_solutions.sort(key=lambda x: solution_sort_key(x["result"]))
        best_feasible = feasible_solutions[0]["result"]
    else:
        # 如果没有可行解，使用当前最佳解
        best_feasible = best_result
    
    # 填充收敛数据结构以保持兼容
    convergence_data_full = {
        'min_total_cost': convergence_data['min_total_cost'],
        'current_cost': convergence_data['current_cost'],
        'avg_total_cost': convergence_data['min_total_cost'],  # 模拟退火没有平均值，用最小值替代
        'feasible_count': convergence_data['feasible_count'],
        'mutation_rate': [cooling_rate] * len(convergence_data['min_total_cost']),
        'diversity': [0] * len(convergence_data['min_total_cost']),  # 模拟退火没有多样性概念
        'genetic_diversity': [0] * len(convergence_data['min_total_cost']),  # 模拟退火没有基因多样性
        'temperature': convergence_data['temperature']  # 添加温度数据
    }
    
    print(f"SA completed in {iteration} iterations. Total evaluations: {iteration * num_processes}")
    print(f"Best cost: {best_feasible['total_cost']:.2f}, Time: {elapsed:.2f}s")
    return best_feasible, feasible_solutions, elapsed, num_processes, convergence_data_full

def plot_convergence(convergence_data, elapsed_time, num_procs):
    """分别绘制模拟退火的各类收敛图"""
    iterations = range(len(convergence_data['min_total_cost']))
    
    # 1. 总成本收敛图
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, convergence_data['min_total_cost'], 'b-', label='Min Total Cost')
    
    # 确保current_cost存在才绘制
    if 'current_cost' in convergence_data:
        plt.plot(iterations, convergence_data['current_cost'], 'g--', label='Current Cost')
    else:
        print("Warning: 'current_cost' not found in convergence_data")
    
    plt.xlabel('Iteration')
    plt.ylabel('Total Cost')
    plt.title(f'Total Cost Convergence\n(Time: {elapsed_time:.2f}s, Processes: {num_procs})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sa_convergence_total_cost.png')
    print("Saved: sa_convergence_total_cost.png")
    plt.close()

    # 2. 温度变化曲线
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, convergence_data['temperature'], 'm-', label='Temperature')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.title('Temperature Schedule')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sa_convergence_temperature.png')
    print("Saved: sa_convergence_temperature.png")
    plt.close()

    # 3. 可行解数量变化
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, convergence_data['feasible_count'], 'g-', label='Feasible Solutions')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Solutions')
    plt.title('Cumulative Feasible Solutions')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sa_convergence_feasible_count.png')
    print("Saved: sa_convergence_feasible_count.png")
    plt.close()

    # 4. 冷却率变化（恒定）
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, convergence_data['mutation_rate'], 'c-', label='Cooling Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Cooling Rate')
    plt.title('Cooling Rate (Constant)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sa_convergence_cooling_rate.png')
    print("Saved: sa_convergence_cooling_rate.png")
    plt.close()

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
        "check_cost": solution_data["check_cost"],  # 包含检查代价
        "fault_cost": solution_data["fault_cost"],
        "data_loss_cost": solution_data["data_loss_cost"],
        "budget": GLOBAL_CONFIG["budget"],
        "encoded_solution": solution_data["individual"]  # 原始个体
    }

if __name__ == "__main__":
    # 运行模拟退火优化
    print("Starting simulated annealing optimization...")
    
    try:
        # 获取CPU核心数
        num_processes = multiprocessing.cpu_count()
        print(f"Detected {num_processes} CPU cores")
        
        # 运行优化
        best_feasible, feasible_solutions, elapsed_time, num_procs, convergence_data = simulated_annealing_optimization(num_processes)
        
        # 绘制收敛图
        plot_convergence(convergence_data, elapsed_time, num_procs)
        
        # 提取最佳解决方案
        best_solution = extract_solution_info(best_feasible)
        
        # 写入结果文件
        with open("sa_optimization_results.txt", "w") as f:
            f.write("Simulated Annealing Optimization Results (Budget-Constrained)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Optimization Time: {elapsed_time:.2f} seconds | Processes Used: {num_procs}\n")
            f.write(f"Total Evaluations: {len(convergence_data['min_total_cost']) * num_procs}\n")
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
            f.write(f"  Check Cost:     {best_solution['check_cost']:.2f}\n")  # 添加检查代价
            f.write(f"  Fault Cost:     {best_solution['fault_cost']:.2f}\n")
            f.write(f"  Data Loss Cost: {best_solution['data_loss_cost']:.2f}\n")
            f.write(f"  Total Cost:     {best_solution['total_cost']:.2f}\n")
            f.write("-" * 80 + "\n\n")
            
            # 其他可行解
            if feasible_solutions:
                # 提取解决方案信息用于排序
                solutions_info = []
                for sol in feasible_solutions:
                    sol_info = extract_solution_info(sol["result"])
                    solutions_info.append(sol_info)
                
                # 修复排序问题：使用新的排序函数
                solutions_info.sort(key=solution_info_sort_key)
                
                f.write(f"Other Feasible Solutions ({len(feasible_solutions)} total):\n")
                f.write("=" * 80 + "\n")
                for i, sol in enumerate(solutions_info[:10]):
                    f.write(f"Solution {i+1} (Cost: {sol['total_cost']:.2f}):\n")
                    f.write(f"  Modules: {sol['modules']}\n")
                    f.write(f"  Warning Energy: {sol['warning_energy']:.2f}%\n")
                    f.write(f"  Preventive Check Days: {sol['preventive_check_days']} days\n")
                    if sol['frequency_heartbeat'] is not None:
                        f.write(f"  Heartbeat Frequency: {sol['frequency_heartbeat']} seconds\n")
                        f.write(f"  Heartbeat Loss Threshold: {sol['heartbeat_loss_threshold']}\n")
                    f.write(f"  Base Cost:      {sol['base_cost']:.2f}\n")
                    f.write(f"  Module Cost:    {sol['module_cost']:.2f}\n")
                    f.write(f"  Check Cost:     {sol['check_cost']:.2f}\n")  # 添加检查代价
                    f.write(f"  Fault Cost:     {sol['fault_cost']:.2f}\n")
                    f.write(f"  Data Loss Cost: {sol['data_loss_cost']:.2f}\n")
                    f.write(f"  Total Cost:     {sol['total_cost']:.2f}\n")
                    f.write("-" * 80 + "\n")
            else:
                f.write("No other feasible solutions found.\n")
        
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
        print(f"    Check Cost:     {best_solution['check_cost']:.2f}")  # 添加检查代价
        print(f"    Fault Cost:     {best_solution['fault_cost']:.2f}")
        print(f"    Data Loss Cost: {best_solution['data_loss_cost']:.2f}")
        print(f"    Total Cost:     {best_solution['total_cost']:.2f}")
        print(f"  Budget: {best_solution['budget']}")
                
        # 打印执行时间
        print(f"\nSA optimization completed! Time: {elapsed_time:.2f} seconds | Processes: {num_procs}")
        print(f"Total evaluations: {len(convergence_data['min_total_cost']) * num_procs}")
        print(f"Feasible solutions found: {len(feasible_solutions) if feasible_solutions else 0}")
    
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)