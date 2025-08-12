# -*- coding: utf-8 -*-
# 定义全局配置参数(输入参数)
GLOBAL_CONFIG = {
    #根据场景不同用户修改的参数
    #电表场景     
    # "sensor_num": 100,           # 传感器节点数量(不包括基站)
    # "area_size": 500,           # 网络区域大小（米），覆盖范围
    # "packet_size": 1000,                # 传感器数据包大小（字节）,每帧数据大小   
    # "per_packet_cost": 5,          # 一个数据包的价值 
    # "per_sensor_cost": 8,            # 单个节点硬件成本(元)
    # "budget": 500000,                 # 预算约束(元)
    # "development_cycle": 3,           # 开发周期（月）
    # "life_span": 60,                   # 仿真时间，要求至少寿命(月)
    # "per_wai_cost": 3,                # 单个硬件外围成本
    # "installation_per_cost": 15,      # 每个节点部署成本
    # "frequency_sampling":1800,            #采样/上传频率（s/次） 一个月
    # "per_check_cost": 500,                # 每次预防性检查的代价（元）
    # "cluster_head_percentage": 0.15,  # 簇头比例
    # "max_hops": 4,              # 扇区数量：簇头最大跳数限制(1-8),不得超过8，1指所有簇头直接连接基站
    # "frames_per_round":6,   #每轮帧数   frequency_sampling*frames_per_round=每轮时间（s）

    # 动物房场景
    "sensor_num": 50,           # 传感器节点数量(不包括基站)
    "area_size": 500,           # 网络区域大小（米），覆盖范围
    "packet_size": 3000,                # 传感器数据包大小（字节）,每帧数据大小   
    "per_packet_cost": 500,          # 一个数据包的价值 
    "per_sensor_cost": 20,            # 单个节点硬件成本(元)
    "budget": 500000,                 # 预算约束(元)
    "development_cycle": 8,           # 开发周期（月）
    "life_span": 24,                   # 仿真时间，要求至少寿命(月)
    "per_wai_cost": 8,                # 单个硬件外围成本
    "installation_per_cost": 5,      # 每个节点部署成本
    "frequency_sampling":900,            #采样/上传频率（s/次） 
    "per_check_cost": 200,                # 每次预防性检查的代价（元）
    "cluster_head_percentage": 0.18,  # 簇头比例
    "max_hops": 2,              # 扇区数量：簇头最大跳数限制(1-8),不得超过8，1指所有簇头直接连接基站
    "frames_per_round":4,   #每轮帧数   frequency_sampling*frames_per_round=每轮时间（s）
    
    # LEACH协议参数    
    "intra_cluster_multihop": 0,  # 簇内多跳 (0=关闭, 1=开启)
    "inter_cluster_multihop": 1,  # 簇间多跳 (0=关闭, 1=开启)
    "configuring_package_size":15, #配置包大小(字节)
    "TDMA_package_size":250,    #TDMA时隙包大小，字节
    
    # 能耗参数
    "energy_sampling":  5e-9,         #采样能耗（J）    采样能量 (E_sample) = 工作电压 (V) × 采样工作电流 (I_active) × 采样持续时间 (T_active)
    "energy_consumption_tx": 5e-8,    # 发送电路能耗系数 (J/bit)
    "energy_amplifier": 1e-10,        # 放大能耗系数 (J/bit/m²)
    "energy_consumption_rx": 5e-8,    # 接收电路能耗系数 (J/bit)
    "energy_aggregation": 5e-9,       # 数据聚合能耗系数 (J/bit/node)
    "initial_energy": 5000,      # 电量初始值，单位是J
    
    # 成本参数
    "development_people": 5,          # 开发团队人数
    "monthly_salary": 10000,          # 开发人员月薪
    "per_power_cost": 5,              # 单个无线充电硬件成本
    "per_activation_cost": 2,         # 单个激活硬件成本
    "maintenance_instruction_size":1,   #维护指令大小，单位字节
    "maintenance_noise_size":20,     #校准背景噪声指令大小，单位字节

    #有心跳的时候才有用
    "heartbeat_packet_size": 2,       # 心跳包大小（字节）（1位标志位，节点ID和电量）

    #rts_cts
    "rts_cts_size":20       #rts_cts包大小(包括发送方和接收方地址)
}

# 定义所有可用维护模块及其属性
MODULES = {
    "rts_cts": {
        "name": "RTS/CTS mechanism",
        "cycle": 0.01,
        "cost": 0,
        "prevents": ["hidden node failure"],
    },
    "heartbeat": {
        "name": "node heartbeat monitoring",
        "cycle": 0.005,
        "cost": 0,
        "prevents": ["power failure"],
    },
    "boot_update": {            
        "name": "program/firmware remote update",
        "cycle": 0.005,         #编写更新接口的开发周期
        "cost": 0,              #编写更新接口的模块代价
        "prevents": ["boot_fault"],
    },
    "wireless_power": {
        "name": "wireless charging module",
        "cycle": 0.006,
        "cost": 0,
        "prevents": ["power failure"],
    },
    "hardware_wai": {
        "name": "peripheral hardware maintenance",
        "cost": 0,
        "reduces": {"hardware failure": 0.85},  # 可减少85%的故障
    },
    "remote_restart": {
        "name": "remote restart",
        "cycle": 0.005,
        "cost": 0,
        "fixed_success": {"data failure": 0.7,"data loss failure":0.7}, 
    },
    "remote_reset": {
        "name": "remote restore factory settings",
        "cycle": 0.005,
        "cost": 0,
        "fixed_success": {"data failure": 0.8,"data loss failure":0.8},  
    },
    "noise": {
        "name": "recalibration of background noise",
        "cycle": 0.1,
        "cost": 0,
        "fixed_success": {"data failure": 0.6},  
    },
    "short_restart": {
        "name": "close restart module",
        "cycle": 0.005,
        "cost": 0,
        "fixed_success": {"communication failure": 0.8},  
    },
    "short_reset": {
        "name": "short-range restore factory Settings",
        "cycle": 0.005,
        "cost": 0,
        "fixed_success": {"communication failure": 0.8},  
    },
    "activation": {
        "name": "sensor reactivation",
        "cycle": 0.12,
        "cost": 0,
        "fixed_success": {"communication failure": 0.7},  
    },
}

# 定义可能的故障类型及其概率和对应恢复成本（不同的场景的故障概率可能不一样）
FAULT_TYPES = {
    "hidden node failure": {
        "probability": 0.003,       #月发生的故障概率
        "cost": 500,
    },
    "data failure": {
        "probability": 0.005,
        "cost": 500,
    },
    "data loss failure": {
        "probability": 0.005,
        "cost": 500,
    },
    "communication failure": {
        "probability": 0.008,
        "cost": 500,
    },
    "power failure": {
        "probability": 0,   #leach协议直接可以计算是否没有能量，不使用这里的probability
        "cost": 500,
    },
    "hardware failure": {
        "probability": 0.001,
        "cost": 510,  # 需要额外添加10元硬件成本
    },
    "boot_fault":{
        "probability": 0.0005,
        "cost": 500,     
    }
}

#常规值取中位数
DEFAULT_PARAM_VALUES = {
    "warning_energy": (0.0 + 50.0) / 2,
    "preventive_check_days": (round(GLOBAL_CONFIG["frequency_sampling"]/(60*60*24)) + 180) / 2,
    "frequency_heartbeat": (GLOBAL_CONFIG["frequency_sampling"]/60+GLOBAL_CONFIG["frequency_sampling"]) / 2,
    "heartbeat_loss_threshold": (3 + 15) / 2
}


