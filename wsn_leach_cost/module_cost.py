from utilities import MODULES,GLOBAL_CONFIG
from calculate_cost import calculate_hardware_cost,calculate_development_cost

#模块成本=硬件成本+开发成本
# 为每个模块定义独立计算函数
# def channel_cost():
#     """channel 模块的成本计算逻辑"""
#     channel_cycle = MODULES["channel"]["cycle"]
#     developmentPeople = GLOBAL_CONFIG["development_people"]
#     monthly_salary = GLOBAL_CONFIG["monthly_salary"]
#     developmentCost=calculate_development_cost(developmentPeople,monthly_salary,channel_cycle)
#     return developmentCost

def rts_cts_cost():
    """rts_cts 模块的成本计算逻辑"""
    rts_cts_cycle = MODULES["rts_cts"]["cycle"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,rts_cts_cycle)
    return developmentCost

def heartbeat_cost():
    """heartbeat 模块的成本计算逻辑"""
    heartbeat_cycle = MODULES["heartbeat"]["cycle"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,heartbeat_cycle)
    return developmentCost

def remote_restart_cost():
    """remote_restart 模块的成本计算逻辑"""
    remote_restart_cycle = MODULES["remote_restart"]["cycle"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,remote_restart_cycle)
    return developmentCost

def remote_reset_cost():
    """remote_reset 模块的成本计算逻辑"""
    remote_reset_cycle = MODULES["remote_reset"]["cycle"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,remote_reset_cycle)
    return developmentCost

def boot_update_cost():
    """boot_update 模块的成本计算逻辑"""
    boot_update_cycle = MODULES["boot_update"]["cycle"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,boot_update_cycle)
    return developmentCost

def noise_cost():
    """noise 模块的成本计算逻辑"""
    noise_cycle = MODULES["noise"]["cycle"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,noise_cycle)
    return developmentCost

def short_restart_cost():
    """short_restart 模块的成本计算逻辑"""
    short_restart_cycle = MODULES["short_restart"]["cycle"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,short_restart_cycle)
    return developmentCost

def short_reset_cost():
    """short_reset 模块的成本计算逻辑"""
    short_reset_cycle = MODULES["short_reset"]["cycle"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,short_reset_cycle)
    return developmentCost

def wireless_power_cost():
    """wireless_power 模块的成本计算逻辑"""
    wireless_power_cycle = MODULES["wireless_power"]["cycle"]
    sensor_num = GLOBAL_CONFIG["sensor_num"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    per_power_cost=GLOBAL_CONFIG["per_power_cost"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,wireless_power_cycle)
    hardCost=calculate_hardware_cost(sensor_num,per_power_cost)
    return developmentCost+hardCost

def activation_cost():
    """activation 模块的成本计算逻辑"""
    activation_cycle = MODULES["activation"]["cycle"]
    sensor_num = GLOBAL_CONFIG["sensor_num"]
    developmentPeople = GLOBAL_CONFIG["development_people"]
    monthly_salary = GLOBAL_CONFIG["monthly_salary"]
    per_activation_cost=GLOBAL_CONFIG["per_activation_cost"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,activation_cycle)
    hardCost=calculate_hardware_cost(sensor_num,per_activation_cost)
    return developmentCost+hardCost

def hardware_wai_cost():
    """hardware_wai 模块的成本计算逻辑"""
    sensor_num = GLOBAL_CONFIG["sensor_num"]
    per_wai_cost=GLOBAL_CONFIG["per_wai_cost"]
    hardCost=calculate_hardware_cost(sensor_num,per_wai_cost)
    return hardCost

#这个函数放在main那里就可以只执行一次，然后就只是对应的固定模块成本了
def set_modules_cost():
    # MODULES["channel"]["cost"]=channel_cost()
    MODULES["rts_cts"]["cost"]=rts_cts_cost()
    MODULES["heartbeat"]["cost"]=heartbeat_cost()
    MODULES["remote_restart"]["cost"]=remote_restart_cost()
    MODULES["remote_reset"]["cost"]=remote_reset_cost()
    MODULES["boot_update"]["cost"]=boot_update_cost()
    MODULES["noise"]["cost"]=noise_cost()
    MODULES["short_restart"]["cost"]=short_restart_cost()
    MODULES["short_reset"]["cost"]=short_reset_cost()
    MODULES["wireless_power"]["cost"]=wireless_power_cost()
    MODULES["activation"]["cost"]=activation_cost()
    MODULES["hardware_wai"]["cost"]=hardware_wai_cost()
     

