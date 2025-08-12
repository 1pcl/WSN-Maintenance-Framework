# -*- coding: utf-8 -*-
from wsn_simulation import WSNSimulation
from module_cost import set_modules_cost
from utilities import GLOBAL_CONFIG,DEFAULT_PARAM_VALUES
import matplotlib.pyplot as plt
from calculate_cost import calculate_total_cost_with_simulation

# 设置模块成本
set_modules_cost()

# 不选择任何模块
selected_modules = []  # 空列表表示不选择任何模块
application = "electricity_meter"  # parking_lot, animal_room, None, electricity_meter

# 定义仿真参数
simulation_params = {
    "warning_energy": DEFAULT_PARAM_VALUES["warning_energy"],                 # 预警电量值
    "preventive_check_days": DEFAULT_PARAM_VALUES["preventive_check_days"],          # 预防性检查间隔（天）
    #以下只有有心跳模块的时候才会用到，所以使用遗传算法也要注意这个问题
    "frequency_heartbeat": DEFAULT_PARAM_VALUES["frequency_heartbeat"],         # 心跳频率（s/次）
    "heartbeat_loss_threshold": DEFAULT_PARAM_VALUES["heartbeat_loss_threshold"]         # 心跳包丢失阈值（个）
}

# 创建仿真实例
sim = WSNSimulation(GLOBAL_CONFIG, selected_modules, simulation_params, application)

# 计算总成本
total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost = calculate_total_cost_with_simulation(
    sim, selected_modules=selected_modules
)

# 打印结果
print("="*50)
print("无任何维护模块的结果")
print("="*50)
print(f"总成本: {total_cost:.2f}")
print(f"基础成本: {base_cost:.2f}")
print(f"模块成本: {module_cost:.2f}")
print(f"检查成本: {check_cost:.2f}")
print(f"故障处理成本: {fault_cost:.2f}")
print(f"数据丢失成本: {data_loss_cost:.2f}")
print("="*50)
print(f"预算: {GLOBAL_CONFIG['budget']}")
print(f"预算使用情况: {base_cost + module_cost:.2f} / {GLOBAL_CONFIG['budget']}")
print("="*50)

# 检查是否超出预算
if (base_cost + module_cost) > GLOBAL_CONFIG['budget']:
    print("警告: 超出预算!")
    over_budget = (base_cost + module_cost) - GLOBAL_CONFIG['budget']
    print(f"超出预算金额: {over_budget:.2f}")
else:
    available_budget = GLOBAL_CONFIG['budget'] - (base_cost + module_cost)
    print(f"剩余预算: {available_budget:.2f}")

# 保存结果到文件
with open("no_modules_simulation_results.txt", "w") as f:
    f.write("无任何维护模块仿真结果\n")
    f.write("="*50 + "\n")
    f.write(f"应用场景: {application}\n")
    f.write(f"选择的模块: 无\n")
    f.write("\n仿真参数 (未使用):\n")
    for param, value in simulation_params.items():
        f.write(f"  {param}: {value}\n")
    f.write("\n成本明细:\n")
    f.write(f"  基础成本: {base_cost:.2f}\n")
    f.write(f"  模块成本: {module_cost:.2f}\n")
    f.write(f"  故障处理成本: {fault_cost:.2f}\n")
    f.write(f"  数据丢失成本: {data_loss_cost:.2f}\n")
    f.write(f"  检查成本: {check_cost:.2f}\n")
    f.write(f"  总成本: {total_cost:.2f}\n")
    f.write("\n预算分析:\n")
    f.write(f"  预算总额: {GLOBAL_CONFIG['budget']}\n")
    f.write(f"  预算使用: {base_cost + module_cost:.2f}\n")
    if (base_cost + module_cost) > GLOBAL_CONFIG['budget']:
        over_budget = (base_cost + module_cost) - GLOBAL_CONFIG['budget']
        f.write(f"  超出预算: {over_budget:.2f}\n")
    else:
        available_budget = GLOBAL_CONFIG['budget'] - (base_cost + module_cost)
        f.write(f"  剩余预算: {available_budget:.2f}\n")

print("\n结果已保存到 no_modules_simulation_results.txt")