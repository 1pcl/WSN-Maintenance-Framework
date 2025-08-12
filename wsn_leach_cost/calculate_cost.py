import random
from collections import defaultdict, deque
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import weibull_min
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import concurrent.futures
from deap import base, creator, tools, algorithms
import json
from PIL import Image  # 确保安装了Pillow库
from utilities import MODULES,FAULT_TYPES,GLOBAL_CONFIG

# ======================== 成本计算函数 ========================

def calculate_hardware_cost(nodeNum, hardCost):
    """计算年度硬件成本 （一次性的） - 考虑节点能力和中继节点"""
    # 确保节点数量是整数
    nodeNum = int(round(nodeNum))
        
    return nodeNum*hardCost

#开发成本
def calculate_development_cost(peopleNum,monthly_salary,development_cycle):
    return peopleNum*monthly_salary*development_cycle

#部署成本 installation
def calculate_installation_cost(nodeNum,installation_per_cost):
    return nodeNum*installation_per_cost

def calculate_total_cost_with_simulation(wsn_sim, selected_modules):
    """计算系统总成本"""
    # 从全局配置中获取参数
    sensorNum = GLOBAL_CONFIG["sensor_num"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    development_cycle = GLOBAL_CONFIG["development_cycle"]
    installation_per_cost = GLOBAL_CONFIG["installation_per_cost"]
    sensorCost = GLOBAL_CONFIG["per_sensor_cost"]
    per_packet_cost = GLOBAL_CONFIG["per_packet_cost"]
    per_check_cost=GLOBAL_CONFIG["per_check_cost"]
    
    # 计算固定成本：硬件成本+开发成本+部署成本
    developmentCost = calculate_development_cost(developmentPeople, monthly_salary, development_cycle)  # 开发成本
    installationCost = calculate_installation_cost(sensorNum, installation_per_cost)  # 部署成本
    baseCost = sensorCost * 1.5  # 基站成本
    hardwareCost = calculate_hardware_cost(sensorNum, sensorCost) + calculate_hardware_cost(1, baseCost)  # 硬件成本
    base_cost = developmentCost + installationCost + hardwareCost
    
    # 维护模块成本
    module_cost = sum(MODULES[module]["cost"] for module in selected_modules)
    
    # 获取仿真结果
    loss_data_count,node_fault,check_count=wsn_sim.run_simulation(visualize=False) #计算代价的时候不需要画图
    frames_per_round = GLOBAL_CONFIG["frames_per_round"]

    # fault_count = sum(fault["count"] for fault in FAULT_TYPES.values())
    # 计算总故障次数（所有节点所有故障类型发生次数的总和）
    fault_count = 0
    for fault_timers in node_fault:  # 直接遍历故障计时器字典
        for timer in fault_timers.values():
            fault_count += timer["count"]

    data_loss_cost=loss_data_count*per_packet_cost+fault_count*per_packet_cost*frames_per_round
    
    # 计算故障维修成本（按故障类型分类统计）
    fault_cost = 0
    for fault_type, params in FAULT_TYPES.items():
        # 统计该故障类型在所有节点中的发生总次数
        type_count = 0
        for fault_timers in node_fault:  # 直接遍历故障计时器字典
            type_count += fault_timers[fault_type]["count"]
                # 累加该故障类型的总成本
        fault_cost += type_count * params["cost"]

    # 计算故障成本
    # fault_cost = sum(fault["cost"]*fault["count"] for fault in FAULT_TYPES.values())

    
    # 总故障成本 = 故障维护成本 + 数据丢失成本
    all_failure_cost = fault_cost + data_loss_cost
    
    #检查成本
    check_cost=per_check_cost*check_count

    # 总成本 = 基础成本 + 模块成本 + 总故障成本 + 检查维护成本
    total_cost = base_cost + module_cost + all_failure_cost + check_cost
    
    return total_cost, base_cost, module_cost, check_cost,fault_cost,data_loss_cost
