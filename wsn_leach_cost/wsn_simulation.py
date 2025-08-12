# -*- coding: utf-8 -*-
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from utilities import GLOBAL_CONFIG,FAULT_TYPES,MODULES
from matplotlib.lines import Line2D

class WSNSimulation:
    def __init__(self, config,selected_modules, sim_params=None,application=None):
        # 初始化配置参数
        self.config = config
        self.selected_modules=selected_modules
        self.application=application

        # 设置随机种子
        # self.seed = config["random_seed"]
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        self.month_time=30*24*60*60.0

        # 初始化网络参数
        self.area = config["area_size"]
        self.sensor_num = config["sensor_num"]
        self.initial_energy = config["initial_energy"]
        self.life_span = config["life_span"]
        # self.rounds_per_month = config["rounds_per_month"]
        self.frames_per_round = config["frames_per_round"]
        self.frequency_sampling=config["frequency_sampling"]
        self.rounds_per_month=(self.month_time)/(self.frequency_sampling*self.frames_per_round) #每月轮数
        self.cluster_head_percentage = config["cluster_head_percentage"]
        self.max_comm_range = self._distance(0,0,self.area,self.area)
        self.rounds_time=self.frequency_sampling*self.frames_per_round  #每轮时间

        # 初始化网络状态
        self.current_month = 0
        self.total_months = self.life_span
        # self.per_check_cost=config["per_check_cost"]

        # self.simulation_total_round=self.total_months *self.rounds_per_month
        
        # 能耗系数
        self.energy_tx = config["energy_consumption_tx"]
        self.energy_amp = config["energy_amplifier"]
        self.energy_rx = config["energy_consumption_rx"]
        self.energy_agg = config["energy_aggregation"]
        self.energy_sampling = config["energy_sampling"]
        
        # 数据包参数
        self.packet_size = config["packet_size"]
        self.max_hops = config["max_hops"]
        self.configuring_package_size=config["configuring_package_size"]
        self.TDMA_package_size=config["TDMA_package_size"]
        self.maintenance_noise_size=config["maintenance_noise_size"]
        self.maintenance_instruction_size=config["maintenance_instruction_size"]
        
        # 多跳参数
        self.intra_cluster_multihop = config["intra_cluster_multihop"]
        self.inter_cluster_multihop = config["inter_cluster_multihop"]
        self.number_of_sectors = float(config["max_hops"])

        #心跳参数
        self.heartbeat_size = config["heartbeat_packet_size"]
        # self.warning_energy=config["warning_energy"]
        # self.frequency_heartbeat=config["frequency_heartbeat"]
        # self.heartbeat_count_per_sampling=0


        # 使用传入的参数或默认值
        self.sim_params = sim_params if sim_params else {}
        # 设置动态参数
        self.warning_energy = self.sim_params.get("warning_energy", 5)
        self.preventive_check_days = self.sim_params.get("preventive_check_days", 30)   #这个要转换为轮数
        # 心跳相关参数（只有选择心跳模块时才有效）
        if "heartbeat" in self.selected_modules:
            self.frequency_heartbeat = self.sim_params.get("frequency_heartbeat", 60*30)
            self.heartbeat_loss_threshold = self.sim_params.get("heartbeat_loss_threshold", 5)
            self.heartbeat_count_per_sampling = float(self.frequency_heartbeat) / self.frequency_sampling   #每发送一个数据包平均发送的心跳数
            self.heartbeat_count_per_round=self.heartbeat_count_per_sampling*self.frames_per_round
        else:
            self.heartbeat_count_per_sampling = 0
            self.frequency_heartbeat = 0
            self.heartbeat_loss_threshold = 0
        # 预防性维护成本计数器
        self.preventive_maintenance_cost = 0
        # if("heartbeat" in self.selected_modules and (self.frequency_heartbeat)!=0):
        #     self.heartbeat_count_per_sampling=float(self.frequency_sampling)/(self.frequency_heartbeat)
        
        self.rts_cts_size = config["rts_cts_size"]
        self.rts_cts_flag=0
        if("rts_cts" in self.selected_modules):
            self.rts_cts_flag=1

        self.check_round=round(self.preventive_check_days*60*60.0*24/self.rounds_time)    #多少轮可维护性检查一次
        self.check_count=self.life_span*30.0/self.preventive_check_days

        # 计算基站位置（生成基站）
        self.base_station = self._create_base_station()
        
        # 生成节点
        self.nodes = self._generate_nodes()

        #初始化所有节点扇区(只有开启簇间多跳的时候)
        if(self.inter_cluster_multihop):
            self._initial_sector_division()

    
    def _create_base_station(self):
        """创建基站"""
        if(self.application=="electricity_meter"):
            # 建筑物参数（与绘图函数一致）
            building_width = self.area * 0.8
            center_x = self.area / 2
            building_height = self.area * 2.6  # 26层高度
            lobby_height = building_height / 26  # 大堂高度            
            # 基站位置：大堂层（1楼）的左侧，y坐标在大堂高度的一半处
            x = center_x - building_width * 0.25
            y = lobby_height * 0.5
            self.max_comm_range = self._distance(0,0,building_width,building_height)
        elif(self.application=="parking_lot"):
            x=self.area - 10
            y=self.area - 10
        elif(self.application=="animal_room"):
            x= (self.area / 4 )*3
            y= self.area + 25.0
        else:
            x=self.area + 25.0
            y=self.area / 2           
        return {
            "id": 0,
            "x": x,
            "y": y,
            "distance":0,
            "section":0
        }
    
    def _generate_nodes(self):
        """生成传感器节点"""
        nodes = []

        for i in range(1, self.sensor_num+1):
            if(self.application=="electricity_meter"):
                building_width = self.area * 0.8
                building_height = self.area * 2.6
                floor_height = building_height / 26  # 26层（25层住户+1层大堂）
                # 电梯/楼梯位置（建筑物中央）
                center_x = self.area / 2
                floor = (i - 1) // 4 + 1
                # 计算户号（每层4户）
                unit = (i - 1) % 4
                # 计算y坐标（楼层高度）
                y = (floor) * floor_height + 25  # +1表示从1楼开始
                # 计算x坐标（每层4户的位置）
                if unit == 0:  # 左侧第一户
                    x = center_x - building_width * 0.35
                elif unit == 1:  # 左侧第二户
                    x = center_x - building_width * 0.25
                elif unit == 2:  # 右侧第一户
                    x = center_x + building_width * 0.25
                else:  # 右侧第二户
                    x = center_x + building_width * 0.35
            elif(self.application=="parking_lot"):
                i_mode=i%8
                i_chu=math.floor((i-1)/8)
                if(i_mode==1):
                    y=10+((self.area-20) / 12)*0.5
                elif(i_mode==2):
                    y=10+((self.area-20) / 12)*2.5
                elif(i_mode==3):
                    y=10+((self.area-20) / 12)*3.5
                elif(i_mode==4):
                    y=10+((self.area-20) / 12)*5.5      
                elif(i_mode==5):
                    y=10+((self.area-20) / 12)*6.5
                elif(i_mode==6):
                    y=10+((self.area-20) / 12)*8.5
                elif(i_mode==7):
                    y=10+((self.area-20) / 12)*9.5      
                else:
                    y=10+((self.area-20) / 12)*11.5   
                x=(self.area / 25)*(i_chu) 
            elif(self.application=="animal_room"):
                i_chu=math.floor((i-1)/5)
                i_mode=(i-1)%5
                i_dex=(i-5)%10
                if(i_chu==0 or i_chu==9):
                    if(i_chu==0):
                        x=0
                        y=50+100*i_mode
                    else:
                        x=500
                        y=50+100*i_mode                        
                else:
                    if(i_chu==1 or i_chu==2):
                        x=80
                    elif(i_chu==3 or i_chu==4):
                        x=80+60
                    elif(i_chu==5 or i_chu==6):
                        x=80+60+60+100+60
                    else:
                        x=80+60+60+100+60+60
                    if(i_dex==1):
                        y=0
                    elif(i_dex==2):
                        y=90
                    elif(i_dex==3):
                        y=110
                    elif(i_dex==4):
                        y=190
                    elif(i_dex==5):
                        y=210
                    elif(i_dex==6):
                        y=290
                    elif(i_dex==7):
                        y=310
                    elif(i_dex==8):
                        y=390
                    elif(i_dex==9):
                        y=410
                    else:
                        y=490                                        
            else:
                x=round(np.random.uniform(0, self.area))
                y=round(np.random.uniform(0, self.area))                
            # 初始化故障计时器字典
            fault_timers = {}
            for fault_type in FAULT_TYPES:  #每个故障有独立的下一次故障的时间
                fault_timers[fault_type] = {
                    "next_time": -1,  # 下次故障时间（轮数）
                    "count":0,   #维护故障的次数
                    "last_fault_round":-1,   #这次出现故障的轮数
                    "fault_flag":0,
                    "fault_round":0 #故障维持的总轮数（用来计算故障时导致的数据丢失）（故障节点不参与传输数据，不会出现中间节点导致丢包的情况）
                }

            nodes.append({
                "id": i,
                "energy": self.initial_energy,
                "x": x, 
                "y": y,   
                "distance":self.max_comm_range,           
                "cluster_head": False,  #簇头标志（False=普通节点，True=簇头）
                "routing_information": [], #路由信息列表
                "cluster_info":[],  #簇内节点信息列表
                "sector":0, #扇区号
                "fault": False,     #是否故障
                "fault_timers": fault_timers,  # 故障计时器字典
            })
        return nodes
    
    def _initial_sector_division(self):
        """初始扇区划分"""

        # 1. 收集所有节点到基站的距离
        distanciasBS = []
        for node in self.nodes:
            dist=self._distance(node["x"],node["y"],self.base_station["x"],self.base_station["y"])
            distanciasBS.append(dist)
            node["energy"]=self._energy_consumption_tx(node["energy"],dist,self.configuring_package_size)

        # 2. 为每个节点分配扇区号
        for node in self.nodes:
            node["sector"]=self._setorizacaoCH(distanciasBS,distanciasBS[node["id"]-1],self.number_of_sectors)
            node["energy"]=self._energy_consumption_rx(node["energy"],self.configuring_package_size)

    #扇区划分：用于多跳路由的扇区划分(簇间)
    #将距离划分为divisor个扇区（1-8）
    #输入：所有距离列表（listaDistancias），需要划分扇区的目标距离（distancia），划分扇区的数量（divisor：1-8）
    #输出：根据目标距离所处的区间范围，返回对应的扇区编号（1-8）
    def _setorizacaoCH(self,listaDistancias,distancia,divisor):
        # Calculo entre o menor e o maior
        menor = self._menorLista(listaDistancias)#求最小距离
        maior = self._maiorLista(listaDistancias)#求最大距离

        valor = (maior - menor) / divisor#计算扇区宽度：扇区宽度 = (最大距离 - 最小距离) / 扇区数量

        #确定扇区归属
        if(distancia <= menor + 1*valor):
            return 1
        elif(distancia <= menor + 2*valor):
            return 2
        elif(distancia <= menor + 3*valor):
            return 3
        elif(distancia <= menor + 4*valor):
            return 4
        elif(distancia <= menor + 5*valor):
            return 5
        elif(distancia <= menor + 6*valor):
            return 6
        elif(distancia <= menor + 7*valor):
            return 7
        else:
            return 8
        
    #功能：根据节点到簇头的距离，将节点划分到不同的扇区（1-8）
    #输入：节点[8]列表（lista），扇区数量（divisor）
    def _setorizacao(self,lista,divisor):
        if(lista != []):
            # Vetor das Distâncias
            ordenado = []
            for k in lista:
                ordenado.append(k[3])
            # Calculo entre o menor e o maior
            ordenado.sort()
            valor = (ordenado[-1] - ordenado[0]) / divisor#计算扇区宽度
            # Setorização
            for k in lista:#为每个节点分配扇区
                if(k[3] <= ordenado[0] + 1*valor):
                    k[4] = 1
                elif(k[3] <= ordenado[0] + 2*valor):
                    k[4] = 2
                elif(k[3] <= ordenado[0] + 3*valor):
                    k[4] = 3
                elif(k[3] <= ordenado[0] + 4*valor):
                    k[4] = 4
                elif(k[3] <= ordenado[0] + 5*valor):
                    k[4] = 5
                elif(k[3] <= ordenado[0] + 6*valor):
                    k[4] = 6
                elif(k[3] <= ordenado[0] + 7*valor):
                    k[4] = 7
                else:
                    k[4] = 8

        return lista

    #求最大值
    def _maiorLista(self,lista):
        maior = lista[0]
        for k in lista:
            if(maior < k):
                maior = k
        return maior

    #求最小值
    def _menorLista(self,lista):
        menor = lista[0]
        for k in lista:
            if(k < menor):
                menor = k
        return menor

    def _distance(self, x1, y1, x2, y2):
        """计算两点之间的距离"""
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def _energy_consumption_tx(self, energy, distance, packet_size):
        """计算发送能耗"""
        # 发送能耗 = 电路能耗 + 放大能耗
        return energy - (self.energy_tx * packet_size*8 + 
                         self.energy_amp * packet_size*8 * (distance**2))
    
    def _energy_consumption_rx(self, energy, packet_size):
        """计算接收能耗"""
        # 接收能耗 = 电路能耗
        return energy - (self.energy_rx * packet_size*8)
    
    def _energy_consumption_agg(self, energy, packet_size, num_packets):
        """计算数据聚合能耗"""
        return energy - (self.energy_agg * packet_size*8 * num_packets)
    

    
    """
        调整簇头节点的通信范围，使其能够覆盖簇内最远的成员节点
        
        参数:
            CH: 簇头节点列表（每个簇头是完整数据结构）
            
        说明:
            对于每个簇头节点:
            - 遍历其簇内成员列表(nodeCH[8])簇头节点的node[8]存储成员节点信息列表
            - 每个成员节点信息是一个列表:
                [0]: 节点ID
                [1]: x坐标
                [2]: y坐标
                [3]: 到簇头的距离
                [4]: 扇区号
            - 找到最大的成员节点距离
            - 将该距离设置为簇头节点的通信范围(nodeCH[4])
    """
    def _ajuste_alcance_nodeCH(self,CH):
        for nodeCH in CH:
            maior = 0
            # Verifica os elementos do cluster
            for node in nodeCH["cluster_info"]:
                if(maior < node[3]):
                    maior = node[3]
            # Escolhe a maior distância e configura o rádio
            nodeCH["distance"] = maior
    
    #求id出现在listaID的次数
    def _contEncaminhamento(self,id,listaID):
        cont = 0
        for k in listaID:
            if(k == id):
                cont += 1
        return cont
    
    #查找为id的节点
    def _localizaObjetoCH(self,id, node):
        for k in node:
            if (k["id"] == id):
                return k   
    
    #簇头状态检查：检查是否所有节点都当过簇头（用于重置轮次）
    def _verifica_eleitos(self):
        total = 0
        for k in self.nodes:
            if(k["cluster_head"]):
                total +=   1
        if(total == len(self.nodes)):
            return True
        return False
    
    def _select_ch(self,total_round_num):
        """选择簇头"""
        # 重置所有节点
        for node in self.nodes:
            if node["fault"]==False:
                # node["cluster_head"] = False
                node["distance"]=self.max_comm_range
                node["routing_information"]=[]
                node["cluster_info"]=[]

        # if(self._verifica_eleitos()):
        if (total_round_num%round(1.0 / self.cluster_head_percentage)==0):
            for k in self.nodes:
                k["cluster_head"] = False        
        
        # 清空簇结构和非簇结构
        cluster_heads = []
        non_cluster_heads=[]

        #原始的选择簇头的公式
        limiar = self.cluster_head_percentage / (1.0 - self.cluster_head_percentage * (total_round_num % round(1.0 / self.cluster_head_percentage)))

        for node in self.nodes:
            if (node["fault"]==False):
                rand=random.random()
                #考虑能量的选择簇头的方式（改进TODO）
                # limiar=limiar*(node["energy"]/self.initial_energy)
                if(limiar>rand) and (node["cluster_head"]!=True):
                    node["cluster_head"]=True
                    cluster_heads.append(node)
                else:
                    non_cluster_heads.append(node)    
        
        #确保必须有一个簇头
        if(len(cluster_heads)==0):
            for node in self.nodes:
                if (node["fault"]==False):
                    if node["cluster_head"]!=True:
                        node["cluster_head"]=True
                        cluster_heads.append(node)
                        non_cluster_heads.remove(node)
                        break                    
        if(len(cluster_heads)==0):
            for node in self.nodes:
                if (node["fault"]==False):
                    node["cluster_head"]=True
                    cluster_heads.append(node)
                    non_cluster_heads.remove(node)
                    break

        if(len(cluster_heads)!=0):
            # 2. 簇形成阶段
            #簇头发送广播
            pacotesBroadcast = []
            for k in cluster_heads:
                pacotesBroadcast.append( [k["id"],k["x"],k["y"],0.0,k["sector"]] )
                # Registro da BS para envio
                baseStation=[self.base_station["id"],self.base_station["x"],self.base_station["y"],self.base_station["distance"],self.base_station["section"]]
                k["routing_information"].append(baseStation) # 添加基站到路由表
                k["energy"] = self._energy_consumption_tx(k["energy"],k["distance"],self.configuring_package_size)

            # 簇头接收其他簇头广播
            for k in cluster_heads:
                for node in pacotesBroadcast:
                    if(node[3] != k["id"]):
                        k["routing_information"].append(node)
                        k["energy"] = self._energy_consumption_rx(k["energy"],self.configuring_package_size)    #会接收到其他簇头的广播的能量
            
            # 普通节点选择簇头
            if(non_cluster_heads!=[]):
                # 普通节点接收簇头广播
                for k in non_cluster_heads:
                    menorDistancia = k["distance"]
                    nodeMenorDistancia = []
                    # Escolha do CH (o mais próximo)
                    for nodesCH in pacotesBroadcast:
                        k["energy"] = self._energy_consumption_rx(k["energy"],self.configuring_package_size)    #会接收到所有簇头的广播的能量
                        dist = self._distance(k["x"],k["y"],nodesCH[1],nodesCH[2])
                        if(dist < menorDistancia):
                            menorDistancia = dist
                            nodeMenorDistancia = nodesCH
                    # Atualização dos valores
                    k["routing_information"] = [ nodeMenorDistancia ]   #只有一条路由信息（只会保持给最近的簇头）
                    k["distance"] = menorDistancia                            
                # 普通节点发送加入请求
                for k in non_cluster_heads:
                    node = [k["id"],k["x"],k["y"],k["distance"],0]
                    # localiza o CH escolhido na lista de CH e coloca seu node em ListCL do CH
                    for nodeCH in cluster_heads:
                        if(k["routing_information"][0][0] == nodeCH["id"]):
                            nodeCH["cluster_info"].append(node)

                    k["energy"] = self._energy_consumption_tx(k["energy"],k["distance"],self.configuring_package_size)
                    
                # 簇头接收加入请求
                for k in cluster_heads:
                    # Nodes atribuídos na função anterior
                    for l in range( len(k["cluster_info"]) ):
                        k["energy"] = self._energy_consumption_rx(k["energy"],self.configuring_package_size)

                # 簇头发送TDMA调度表
                self._ajuste_alcance_nodeCH(cluster_heads)   # 1. 调整簇头通信范围
                clusters = []   # 2. 创建空列表存储簇信息
                for k in cluster_heads:    # 3. 遍历所有簇头
                    # 4. 对簇内成员进行扇区划分
                    # setorizacao()函数为每个成员节点分配扇区号
                    # 5. 创建簇结构：[簇头ID, 带扇区号的成员列表]
                    clusters.append( [k["id"], self._setorizacao(k["cluster_info"],self.number_of_sectors)] )
                    k["energy"] = self._energy_consumption_tx(k["energy"],k["distance"],self.TDMA_package_size)
                # 普通节点接收TDMA表
                for k in non_cluster_heads:
                    idCH = k["routing_information"][0][0]
                    # Localiza o cluster do CH
                    for clstr in clusters:
                        if(clstr[0] == idCH):
                            k["cluster_info"] = clstr[1]
                    k["energy"] = self._energy_consumption_rx(k["energy"],self.TDMA_package_size)            

                # 簇头设置到基站的距离
                for k in cluster_heads:
                    k["distance"] = self._distance(k["x"],k["y"], self.base_station["x"],self.base_station["y"])

                # 簇内多跳路由
                if(self.intra_cluster_multihop == 1):
                    # 1. 查找节点自身的扇区号
                    for k in non_cluster_heads:
                        # Acho o setor dentro do clusters
                        setor = 0
                        for node in k["cluster_info"]: # 遍历簇成员列表
                            if(k["id"] == node[0]): # 找到当前节点自身的信息
                                setor = node[4] # 获取节点自身的扇区号
                                break
                        # Achar node vizinho mais proximo
                        # 2. 寻找最佳中继节点
                        id = k["routing_information"][0][0] # 初始化中继节点ID为簇头ID
                        menor = k["distance"] # 初始化最小距离为到簇头的距离
                        rou_x=k["x"]
                        rou_y=k["y"]
                        rou_sec=0
                        # 再次遍历簇成员列表
                        for node in k["cluster_info"]:
                            # 计算到当前邻居节点的距离
                            dist = self._distance(k["x"],k["y"], node[1],node[2])
                            # 检查两个条件：
                            #   a) 该邻居比当前候选更近 (dist < menor)
                            #   b) 该邻居在更靠近簇头的扇区 (node[4] < setor)
                            if(dist < menor and node[4] < setor):
                                id = node[0] # 更新中继节点ID
                                menor = dist # 更新最小距离
                                rou_x=node[1]
                                rou_y=node[2]
                                rou_sec=node[4]
                        # 3. 更新路由信息
                        k["routing_information"] = [[id,rou_x,rou_y,menor,rou_sec]] # 设置下一跳路由
                        k["distance"] = menor # 更新通信距离
                # 簇间多跳路由
                if(self.inter_cluster_multihop == 1):
                    for k in cluster_heads:
                        menor = k["distance"]
                        for node in k["routing_information"]:
                            dist = self._distance(k["x"],k["y"], node[1],node[2])
                            if(dist < menor and node[4] < k["sector"]):
                                menor = dist
                                k["distance"] = menor
                                k["routing_information"] = [node]   
        return cluster_heads,non_cluster_heads
            
    def _simulate_round(self, cluster_heads,non_cluster_heads):
        """模拟一轮网络数据传输运行"""
 
        """模拟数据传输"""
        # 创建路由映射表
        mapaEncaminhamento = []
        for k in non_cluster_heads:
            mapaEncaminhamento.append( k["routing_information"][0][0] )

        # === 数据传输阶段 ===（先采样再发送数据）
        for contFram in range(self.frames_per_round):
            confirmaFrame = 0   # 确认帧标志    成功传输到基站的帧计数器
            actual_transmission =0
            # 普通节点发送数据（在TDMA分配的时隙）
            for k in non_cluster_heads:
                #先采样
                k["energy"]=k["energy"]-self.energy_sampling
                if(self.heartbeat_count_per_sampling!=0):    #发送心跳
                    k["energy"]=self._energy_consumption_tx(k["energy"],k["distance"],self.heartbeat_size*self.heartbeat_count_per_sampling)
                # 如果启用簇内多跳，计算数据聚合能耗
                if(self.intra_cluster_multihop == 1):
                    # Gasto de agregação de dados
                    # 计算当前节点作为中继的次数
                    totalContEnc = self._contEncaminhamento(k["id"], mapaEncaminhamento)
                    if(totalContEnc > 0):
                        # 数据聚合能耗
                        k["energy"]=self._energy_consumption_agg(k["energy"],self.packet_size,(totalContEnc + 1))
                if(self.rts_cts_flag==1):
                    #发送RTS包
                    k["energy"] = self._energy_consumption_tx(k["energy"],k["distance"],self.rts_cts_size)
                    #收到CTS包
                    k["energy"] =  self._energy_consumption_rx(k["energy"],self.rts_cts_size)
                #发送数据
                k["energy"] = self._energy_consumption_tx(k["energy"],k["distance"],self.packet_size)
                actual_transmission=actual_transmission+1
                if k["energy"]>0:
                    confirmaFrame=confirmaFrame+1
                else:
                    if("wireless_power" in self.selected_modules):
                        confirmaFrame=confirmaFrame+1   #无线充电不会因为电量而导致数据丢失
                        k["energy"]=k["energy"]+self.initial_energy
                    else:
                        k["fault"]=True
            # 簇头接收数据
            # CH: Recebe Pacote
            for k in cluster_heads:
                for l in range( self._contEncaminhamento(k["id"], mapaEncaminhamento) ):
                    if(self.rts_cts_flag==1):
                        #接收RTS包
                        k["energy"] =  self._energy_consumption_rx(k["energy"],self.rts_cts_size)
                        #发送CTS包
                        k["energy"] = self._energy_consumption_tx(k["energy"],k["distance"],self.rts_cts_size)
                    #接收数据
                    k["energy"] = self._energy_consumption_rx(k["energy"],self.packet_size)
            # NCH: Recebe Pacote
            # 普通节点接收数据（仅簇内多跳）
            if(self.intra_cluster_multihop == 1):
                for k in non_cluster_heads:
                    for l in range( self._contEncaminhamento(k["id"], mapaEncaminhamento) ):
                        if(self.rts_cts_flag==1):
                            #接收RTS包
                            k["energy"] =  self._energy_consumption_rx(k["energy"],self.rts_cts_size)
                            #发送CTS包
                            k["energy"] = self._energy_consumption_tx(k["energy"],k["distance"],self.rts_cts_size)
                        k["energy"] =  self._energy_consumption_rx(k["energy"],self.packet_size)
            # CH: Envia Pacote para a BS
            # 簇头向基站发送数据（可能多跳）
            for k in cluster_heads:
                #先采样
                k["energy"]=k["energy"]-self.energy_sampling
                if(self.heartbeat_count_per_sampling!=0):    #发送心跳
                    k["energy"]=self._energy_consumption_tx(k["energy"],k["distance"],self.heartbeat_size*self.heartbeat_count_per_sampling)
                # Gasto de agregação de dados
                totalContEnc = self._contEncaminhamento(k["id"], mapaEncaminhamento)
                if(totalContEnc > 0):
                    # 数据聚合能耗
                    k["energy"] = self._energy_consumption_agg(k["energy"],self.packet_size,(totalContEnc + 1))
                node = k
                idDestino = node["routing_information"][0][0]
                while(idDestino != 0):  # 0表示基站
                    if(self.rts_cts_flag==1):
                        #发送RTS包
                        node["energy"] = self._energy_consumption_tx(node["energy"],node["distance"],self.rts_cts_size)
                        #收到CTS包
                        node["energy"] =  self._energy_consumption_rx(node["energy"],self.rts_cts_size)
                    node["energy"] = self._energy_consumption_tx(node["energy"],node["distance"],self.packet_size)
                    actual_transmission=actual_transmission+1
                    if node["energy"]>0:
                        confirmaFrame=confirmaFrame+1
                    else:
                        if("wireless_power" in self.selected_modules):
                            confirmaFrame=confirmaFrame+1
                            node["energy"]=node["energy"]+self.initial_energy
                        else:
                            node["fault"]=True
                    node = self._localizaObjetoCH(idDestino,cluster_heads)
                    if(self.rts_cts_flag==1):
                        #接收RTS包
                        k["energy"] =  self._energy_consumption_rx(k["energy"],self.rts_cts_size)
                        #发送CTS包
                        k["energy"] = self._energy_consumption_tx(k["energy"],k["distance"],self.rts_cts_size)
                    # Gasto Recepção do node destino
                    node["energy"] = self._energy_consumption_rx(node["energy"],self.packet_size)
                    idDestino = node["routing_information"][0][0]
                # 最后发送到基站
                if(self.rts_cts_flag==1):
                    #发送RTS包
                    node["energy"] = self._energy_consumption_tx(node["energy"],node["distance"],self.rts_cts_size)
                    #收到CTS包
                    node["energy"] =  self._energy_consumption_rx(node["energy"],self.rts_cts_size)
                node["energy"] = self._energy_consumption_tx(node["energy"],node["distance"],self.packet_size)
                actual_transmission=actual_transmission+1
                if(node["energy"] > 0):
                    # Confirma que houve um envio a BS
                     confirmaFrame += 1
                else:
                    if("wireless_power" in self.selected_modules):
                        confirmaFrame=confirmaFrame+1
                        node["energy"]=node["energy"]+self.initial_energy
                    else:
                        node["fault"]=True
 
        loss_data_count=actual_transmission-confirmaFrame        
        
        return loss_data_count


    def plot_initial_topology(self, save_folder="visualizations"):
        """
        绘制26层建筑结构（25层住宅+1层大堂）的初始拓扑图
        """
        # 创建保存目录
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # 创建图形
        plt.figure(figsize=(12, 16))
        
        # 电表场景的特殊处理
        if self.application == "electricity_meter":
            # 建筑物参数
            building_width = self.area * 0.8
            building_height = self.area * 2.6  # 26层高度
            building_left = (self.area - building_width) / 2
            building_bottom = 0
            floor_height = building_height / 26
            
            # 绘制建筑物轮廓
            building = plt.Rectangle(
                (building_left, building_bottom), 
                building_width, 
                building_height,
                fill=True, 
                color='#F5F5F5',
                edgecolor='#333333',
                linewidth=2
            )
            plt.gca().add_patch(building)
            
            # 绘制楼层分隔线（每层）
            for i in range(1, 26):  # 25条分隔线（1-25层）
                y_pos = building_bottom + i * floor_height
                plt.plot(
                    [building_left, building_left + building_width],
                    [y_pos, y_pos],
                    color='#CCCCCC',
                    linewidth=0.7,
                    linestyle='-'
                )
            
            
            # 绘制电梯（建筑物中央左侧）
            center_x = self.area / 2
            elevator_width = building_width * 0.1
            elevator_height = building_height - floor_height  # 从1层到屋顶
            elevator_left = center_x - building_width * 0.15
            elevator_bottom = building_bottom + floor_height  # 从1层开始
            
            elevator = plt.Rectangle(
                (elevator_left, elevator_bottom),
                elevator_width,
                elevator_height,
                fill=True,
                color='#E8E8E8',
                edgecolor='#999999',
                linewidth=1.5
            )
            plt.gca().add_patch(elevator)
            
            # 添加"ELEVATOR"文字
            plt.text(
                elevator_left + elevator_width/2, 
                elevator_bottom + elevator_height/2,
                'ELEVATOR',
                rotation=90,
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                color='#555555'
            )
            
            # 绘制楼梯（建筑物中央右侧）
            stairs_width = building_width * 0.1
            stairs_height = building_height - floor_height  # 从1层到屋顶
            stairs_left = center_x + building_width * 0.05
            stairs_bottom = building_bottom + floor_height  # 从1层开始
            
            stairs = plt.Rectangle(
                (stairs_left, stairs_bottom),
                stairs_width,
                stairs_height,
                fill=True,
                color='#E8E8E8',
                edgecolor='#999999',
                linewidth=1.5
            )
            plt.gca().add_patch(stairs)
            
            # 添加"STAIRS"文字
            plt.text(
                stairs_left + stairs_width/2, 
                stairs_bottom + stairs_height/2,
                'STAIRS',
                rotation=90,
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                color='#555555'
            )
            
            # 绘制大堂区域（底层）
            lobby_height = floor_height
            lobby = plt.Rectangle(
                (building_left, building_bottom),
                building_width,
                lobby_height,
                fill=True,
                color='#F0F0FF',
                edgecolor='#333333',
                linewidth=1.5,
                alpha=0.7
            )
            plt.gca().add_patch(lobby)
            
            # 绘制大门（大堂中央）
            door_width = building_width * 0.15
            door_height = lobby_height * 0.7
            door_left = center_x - door_width/2
            door_bottom = building_bottom
            
            door = plt.Rectangle(
                (door_left, door_bottom),
                door_width,
                door_height,
                fill=True,
                color='#8B4513',
                linewidth=1.5
            )
            plt.gca().add_patch(door)
            
            # 添加"MAIN ENTRANCE"标签
            plt.text(
                center_x,
                door_bottom + door_height + 5,
                'MAIN ENTRANCE',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='#8B4513'
            )
            # 停车场场景的特殊处理
        elif self.application == "parking_lot":
            # 绘制停车位网格
            slot_width = (self.area / 25) * 0.8  # 车位宽度
            slot_height = ((self.area-20) / 12) * 0.8  # 车位高度
            
            # 绘制所有停车位
            for i in range(1, self.sensor_num+1):
                # 计算车位位置（与节点生成逻辑一致）
                i_mode = i % 8
                i_chu = math.floor((i-1)/8)
                
                if(i_mode==1):
                    y=10+((self.area-20) / 12)*0.5
                elif(i_mode==2):
                    y=10+((self.area-20) / 12)*2.5
                elif(i_mode==3):
                    y=10+((self.area-20) / 12)*3.5
                elif(i_mode==4):
                    y=10+((self.area-20) / 12)*5.5      
                elif(i_mode==5):
                    y=10+((self.area-20) / 12)*6.5
                elif(i_mode==6):
                    y=10+((self.area-20) / 12)*8.5
                elif(i_mode==7):
                    y=10+((self.area-20) / 12)*9.5      
                else:
                    y=10+((self.area-20) / 12)*11.5   
                x=(self.area / 25)*(i_chu) 
                
                # 绘制车位矩形
                slot = plt.Rectangle(
                    (x - slot_width/2, y - slot_height/2),
                    slot_width,
                    slot_height,
                    fill=False,
                    edgecolor='gray',
                    linewidth=0.7,
                    alpha=0.7
                )
                plt.gca().add_patch(slot)                      
            
            # 添加停车场边界
            boundary = plt.Rectangle(
                (0, 0), self.area, self.area,
                fill=False, edgecolor='black', linewidth=2
            )
            plt.gca().add_patch(boundary)
        # 动物房场景的特殊处理
        elif self.application == "animal_room":            
            # 绘制房间分隔线
            # 竖线 (垂直分隔)
            v_lines = [0, 200, 300, 500]
            for x in v_lines:
                plt.plot([x, x], [0, 500], color='blue', linestyle='-', linewidth=1)
            
            # 横线 (水平分隔)
            h_lines = [0, 100, 200, 300, 400, 500]
            for y in h_lines:
                plt.plot([0, 500], [y, y], color='blue', linestyle='-', linewidth=1)
            
            # 填充走廊区域
            plt.fill_between([200, 300], 0, 500, color='lightgray', alpha=0.3)
            
            # 添加走廊标签
            plt.text(200, 300, 'CORRIDOR', 
                    rotation=90, 
                    ha='center', va='center', 
                    fontsize=12, fontweight='bold',
                    color='darkgray')
   
        
        # 绘制基站（紫色正方形）
        plt.plot(self.base_station["x"], self.base_station["y"], 's', 
                color='purple', markersize=18, label='Base Station')
        

        # 绘制所有节点（绿色圆圈）
        for node in self.nodes:
            plt.plot(node["x"], node["y"], 'o', color='green', markersize=8, alpha=0.7)
        
        # 设置图例
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', 
                    markersize=12, label='Base Station'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                    markersize=8, label=f'Nodes: {len(self.nodes)}'),
        ]
        
        # 添加建筑元素图例
        if self.application == "electricity_meter":
            legend_elements.extend([
                plt.Line2D([0], [0], color='#F5F5F5', lw=4, label='Building (26 Floors)'),
                plt.Line2D([0], [0], color='#E8E8E8', lw=4, label='Elevator/Stairs'),
                plt.Line2D([0], [0], color='#F0F0FF', lw=4, label='Lobby'),
                plt.Line2D([0], [0], color='#8B4513', lw=4, label='Main Entrance')
            ])
            # 添加停车场图例
        elif self.application == "parking_lot":
            legend_elements.append(
                plt.Line2D([0], [0], color='gray', lw=1, label='Parking Slots')
            )
        elif self.application=="animal_room":
            # 添加建筑元素图例
            legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='Room Walls'))
            legend_elements.append(Line2D([0], [0], color='lightgray', lw=4, label='Corridor')) 


        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # 设置标题和坐标轴
        plt.title('Initial WSN Topology')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # 设置坐标轴范围
        if self.application == "electricity_meter":
            plt.xlim([building_left - 10, building_left + building_width + 30])
            plt.ylim([-building_height*0.05, building_height*1.05])
        elif self.application == "parking_lot":
            plt.xlim([-5, self.area + 5])
            plt.ylim([-5, self.area + 5])
        elif self.application=="animal_room":
            # 设置坐标轴范围
            plt.xlim([-10, 550])
            plt.ylim([-10, 550])
            
        plt.grid(True, alpha=0.2)
        
        # 保存图片
        save_path = os.path.join(save_folder, 'initial_topology.png')
        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.08)  # 为底部文本留出空间
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        return save_path
 

    def run_simulation(self, visualize=True):
        """运行完整的仿真"""
        # 初始可视化
        if visualize:
            self.plot_initial_topology()

        # 按月运行仿真
        for month in range(1, self.total_months + 1):
            self.current_month = month
                    
            # 按轮次运行
            for round_num in range(1, round(self.rounds_per_month) + 1):
                # 模拟一轮网络运行
                total_round_num=(month-1)*self.rounds_per_month+round_num
                cluster_heads,non_cluster_heads=self._select_ch(total_round_num)            
                self.simulation_fault(total_round_num)
                loss_data_count = self._simulate_round(cluster_heads,non_cluster_heads)                      
                # 每次检查可维护性时进行可视化
                if visualize and total_round_num % self.check_round == (self.check_round-1):  
                    self._visualize_network(month, round_num,cluster_heads,non_cluster_heads)
                #电量的修复
                self.simulation_power_failre(total_round_num)
                #定期进行维护检查并对故障进行维护
                if total_round_num % self.check_round == 0:
                    #故障修复
                    self.fixing_fault(total_round_num)
        node_fault_list = [node["fault_timers"] for node in self.nodes]
        return loss_data_count,node_fault_list,self.check_count

    #可视化图片 
    def _visualize_network(self, month, round_num, cluster_heads, non_cluster_heads, save_folder="visualizations"):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # 创建图形和Axes
        fig = plt.figure(figsize=(12, 16))
        ax = fig.add_subplot(111)

            # 电表场景的特殊处理 - 绘制建筑结构
        if self.application == "electricity_meter":
            # 建筑物参数
            building_width = self.area * 0.8
            building_height = self.area * 2.6  # 26层高度
            building_left = (self.area - building_width) / 2
            building_bottom = 0
            floor_height = building_height / 26
            
            # 绘制建筑物轮廓
            building = plt.Rectangle(
                (building_left, building_bottom), 
                building_width, 
                building_height,
                fill=True, 
                color='#F5F5F5',
                edgecolor='#333333',
                linewidth=2,
                zorder=1  # 确保在底层
            )
            ax.add_patch(building)
            
            # 绘制楼层分隔线（每层）
            for i in range(1, 26):  # 25条分隔线（1-25层）
                y_pos = building_bottom + i * floor_height
                ax.plot(
                    [building_left, building_left + building_width],
                    [y_pos, y_pos],
                    color='#CCCCCC',
                    linewidth=0.7,
                    linestyle='-',
                    zorder=2
                )
            
            # 绘制电梯（建筑物中央左侧）
            center_x = self.area / 2
            elevator_width = building_width * 0.1
            elevator_height = building_height - floor_height  # 从1层到屋顶
            elevator_left = center_x - building_width * 0.15
            elevator_bottom = building_bottom + floor_height  # 从1层开始
            
            elevator = plt.Rectangle(
                (elevator_left, elevator_bottom),
                elevator_width,
                elevator_height,
                fill=True,
                color='#E8E8E8',
                edgecolor='#999999',
                linewidth=1.5,
                zorder=2
            )
            ax.add_patch(elevator)
            
            # 添加"ELEVATOR"文字
            ax.text(
                elevator_left + elevator_width/2, 
                elevator_bottom + elevator_height/2,
                'ELEVATOR',
                rotation=90,
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                color='#555555',
                zorder=3
            )
            
            # 绘制楼梯（建筑物中央右侧）
            stairs_width = building_width * 0.1
            stairs_height = building_height - floor_height  # 从1层到屋顶
            stairs_left = center_x + building_width * 0.05
            stairs_bottom = building_bottom + floor_height  # 从1层开始
            
            stairs = plt.Rectangle(
                (stairs_left, stairs_bottom),
                stairs_width,
                stairs_height,
                fill=True,
                color='#E8E8E8',
                edgecolor='#999999',
                linewidth=1.5,
                zorder=2
            )
            ax.add_patch(stairs)
            
            # 添加"STAIRS"文字
            ax.text(
                stairs_left + stairs_width/2, 
                stairs_bottom + stairs_height/2,
                'STAIRS',
                rotation=90,
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                color='#555555',
                zorder=3
            )
            
            # 绘制大堂区域（底层）
            lobby_height = floor_height
            lobby = plt.Rectangle(
                (building_left, building_bottom),
                building_width,
                lobby_height,
                fill=True,
                color='#F0F0FF',
                edgecolor='#333333',
                linewidth=1.5,
                alpha=0.7,
                zorder=3
            )
            ax.add_patch(lobby)
            
            # 绘制大门（大堂中央）
            door_width = building_width * 0.15
            door_height = lobby_height * 0.7
            door_left = center_x - door_width/2
            door_bottom = building_bottom
            
            door = plt.Rectangle(
                (door_left, door_bottom),
                door_width,
                door_height,
                fill=True,
                color='#8B4513',
                linewidth=1.5,
                zorder=4
            )
            ax.add_patch(door)
            
            # 添加"MAIN ENTRANCE"标签
            ax.text(
                center_x,
                door_bottom + door_height + 5,
                'MAIN ENTRANCE',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='#8B4513',
                zorder=5
            )
        # 停车场场景的特殊处理
        elif self.application == "parking_lot":
            # 绘制停车位网格
            slot_width = (self.area / 25) * 0.8  # 车位宽度
            slot_height = ((self.area-20) / 12) * 0.8  # 车位高度
            
            # 绘制所有停车位
            for i in range(1, self.sensor_num+1):
                # 计算车位位置（与节点生成逻辑一致）
                i_mode = i % 8
                i_chu = math.floor((i-1)/8)
                
                if(i_mode==1):
                    y=10+((self.area-20) / 12)*0.5
                elif(i_mode==2):
                    y=10+((self.area-20) / 12)*2.5
                elif(i_mode==3):
                    y=10+((self.area-20) / 12)*3.5
                elif(i_mode==4):
                    y=10+((self.area-20) / 12)*5.5      
                elif(i_mode==5):
                    y=10+((self.area-20) / 12)*6.5
                elif(i_mode==6):
                    y=10+((self.area-20) / 12)*8.5
                elif(i_mode==7):
                    y=10+((self.area-20) / 12)*9.5      
                else:
                    y=10+((self.area-20) / 12)*11.5   
                x=(self.area / 25)*(i_chu) 
                
                # 绘制车位矩形
                slot = plt.Rectangle(
                    (x - slot_width/2, y - slot_height/2),
                    slot_width,
                    slot_height,
                    fill=False,
                    edgecolor='gray',
                    linewidth=0.7,
                    alpha=0.7,
                    zorder=1  # 确保在底层
                )
                ax.add_patch(slot)
                
                # 添加车位编号
                ax.text(
                    x, y,
                    str(i),
                    ha='center',
                    va='center',
                    fontsize=6,
                    zorder=2
                )
            
            # 设置坐标轴范围
            ax.set_xlim([-5, self.area + 5])
            ax.set_ylim([-5, self.area + 5])
            
            # 添加停车场边界
            boundary = plt.Rectangle(
                (0, 0), self.area, self.area,
                fill=False, edgecolor='black', linewidth=2, zorder=1
            )
            ax.add_patch(boundary)
        # 动物房场景的特殊处理
        elif self.application == "animal_room":
            # 整个区域尺寸
            area_width = 500
            area_height = 500
            
            # 绘制房间分隔线
            # 竖线 (垂直分隔)
            v_lines = [0, 200, 300, 500]
            for x in v_lines:
                ax.plot([x, x], [0, 500], color='blue', linestyle='-', linewidth=1, zorder=1)
            
            # 横线 (水平分隔)
            h_lines = [0, 100, 200, 300, 400, 500]
            for y in h_lines:
                ax.plot([0, 500], [y, y], color='blue', linestyle='-', linewidth=1, zorder=1)
            
            # 填充走廊区域
            ax.fill_between([200, 300], 0, 500, color='lightgray', alpha=0.3, zorder=0)
            
            # 添加走廊标签
            ax.text(200, 300, 'CORRIDOR', 
                    rotation=90, 
                    ha='center', va='center', 
                    fontsize=12, fontweight='bold',
                    color='darkgray', zorder=1)
            
        
        # 绘制基站
        ax.plot(self.base_station["x"], self.base_station["y"], 's', color='purple', markersize=15, label='Base Station')
        
        # 分类节点
        fault_nodes = [node for node in self.nodes if node["fault"]]
        normal_nodes = non_cluster_heads
        ch_nodes = cluster_heads
        
        # 创建能量颜色映射
        norm_energy = plt.Normalize(0, self.initial_energy)
        cmap = LinearSegmentedColormap.from_list('energy_cmap', ['red', 'yellow', 'green'])
        
        # 绘制故障节点
        for node in fault_nodes:
            ax.plot(node["x"], node["y"], 'o', color='black', markersize=8, alpha=0.7)
        
        # 绘制普通节点
        for node in normal_nodes:
            if not node["fault"]:
                color = cmap(norm_energy(node["energy"]))
                ax.plot(node["x"], node["y"], 'o', color=color, markersize=8, alpha=0.7)
        
        # 绘制簇头节点
        for node in ch_nodes:
            if not node["fault"]:
                color = cmap(norm_energy(node["energy"]))
                ax.plot(node["x"], node["y"], '*', color=color, markersize=12, alpha=0.9)
        
        # 绘制数据传输箭头
        for node in normal_nodes + ch_nodes:
            if not node["fault"] and node["routing_information"]:
                next_hop = node["routing_information"][0]
                if next_hop[0] == 0:  # 基站
                    target_x, target_y = self.base_station["x"], self.base_station["y"]
                else:
                    target_node = next((n for n in ch_nodes + normal_nodes if n["id"] == next_hop[0]), None)
                    if target_node:
                        target_x, target_y = target_node["x"], target_node["y"]
                    else:
                        continue
                # 使用ax.arrow替代plt.arrow
                ax.arrow(node["x"], node["y"], 
                        target_x - node["x"], target_y - node["y"],
                        head_width=0.5, head_length=1, 
                        fc='blue', ec='blue', alpha=0.5,
                        length_includes_head=True)
        
        # 设置图例
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=10, label='Base Station'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=12, label='Cluster Head'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Normal Node'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Fault Node'),
            Line2D([0], [0], color='blue', lw=2, label='Data Flow'),
        ]
        
        # 添加颜色条（指定ax参数）
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_energy)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label('Node Energy Level')
        
        # 添加图例
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        # 添加建筑元素图例
        if self.application == "electricity_meter":
            legend_elements.extend([
                Line2D([0], [0], color='#F5F5F5', lw=4, label='Building (26 Floors)'),
                Line2D([0], [0], color='#E8E8E8', lw=4, label='Elevator/Stairs'),
                Line2D([0], [0], color='#F0F0FF', lw=4, label='Lobby'),
                Line2D([0], [0], color='#8B4513', lw=4, label='Main Entrance')
            ])
            # 添加停车场图例
        elif self.application == "parking_lot":
            legend_elements.append(
                Line2D([0], [0], color='gray', lw=1, label='Parking Slots')
            )
        elif self.application=="animal_room":
            # 添加建筑元素图例
            legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='Room Walls'))
            legend_elements.append(Line2D([0], [0], color='lightgray', lw=4, label='Corridor')) 


        # 设置标题和坐标轴
        ax.set_title(f"WSN Topology - Month {month}, Round {round_num}\n"
                    f"Fault Nodes: {len(fault_nodes)}/{len(self.nodes)}")
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        # 设置坐标轴范围
        if self.application == "electricity_meter":
            ax.set_xlim([building_left - 10, building_left + building_width + 30])
            ax.set_ylim([-building_height*0.05, building_height*1.05])
        elif self.application == "parking_lot":
            ax.set_xlim([-5, self.area + 5])
            ax.set_ylim([-5, self.area + 5])
        elif self.application=="animal_room":
            # 设置坐标轴范围
            ax.set_xlim([-10, 550])
            ax.set_ylim([-10, 550])


        ax.grid(True, alpha=0.3)
        
        # 保存图片
        save_path = os.path.join(save_folder, f'topology_month_{month}_round_{round_num}.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path

    #power failure 故障模拟与修复(1.无线充电模块没有电就可以自动充电 2.刚好心跳小于预警电量的时候换电池 3.刚好维护检查的时候发现没电或者小于预警电量的时候换电池)
    def simulation_power_failre(self,current_round_num):
        for node in self.nodes:
            timer = node["fault_timers"]["power failure"]
            if("wireless_power" in self.selected_modules):  #有无线充电模块
                if (node["energy"]<=0):     #当电量不足时，可以自动充电
                    node["energy"]=node["energy"]+self.initial_energy   #出现仿真的时候就已经电量不足，那个时候的电量也要计算上去（仿真时也有充电的场景）
            else:
                if (node["energy"]<self.warning_energy*self.initial_energy):    #预警，还未出现故障就已经换电池了 
                    #刚好维护检查到大于0而且可以换电池了，或者每一轮至少都有超过一个心跳所以每轮都能通过心跳检查大于0可以换电池了
                    if (current_round_num % self.check_round == 0 and node["energy"]>=0) or ("heartbeat" in self.selected_modules and self.heartbeat_count_per_round >=1 and node["energy"]>=0):
                        timer["count"]+=1
                        # FAULT_TYPES["power failure"]["count"]+=1    #这只是计算这个故障的维护代价的数量（没有故障,电量还大于0）
                        node["energy"]=self.initial_energy  #提前换电池
                    elif "heartbeat" in self.selected_modules and current_round_num%self.heartbeat_count_per_round<=1e-10 and node["energy"]>=0:
                        timer["count"]+=1
                        # FAULT_TYPES["power failure"]["count"]+=1    #这只是计算这个故障的维护代价的数量（没有故障,电量还大于0）
                        node["energy"]=self.initial_energy  #提前换电池
                    else:
                        if (node["energy"]<=0):  #出现故障
                            # FAULT_TYPES["power failure"]["count"]+=1
                            timer["count"]+=1
                            node["energy"]=self.initial_energy  #没电再换电池(理论上数据丢失率会比前面高)  
                            node["fault"]=False     #换电池后故障已经修复 
                            timer["last_fault_round"]=current_round_num    #记录当前故障的时间          
                            timer["fault_flag"]=1              

    def simulation_fault(self,current_round_num):
        # scale=self.month_time/60
        for node in self.nodes:
            for fault_type, params in FAULT_TYPES.items():
                timer = node["fault_timers"][fault_type]
                raw_prob = params['probability']
                # 应用模块效果
                prevented = any(fault_type in MODULES[module].get("prevents", []) 
                        for module in self.selected_modules)
                if prevented or raw_prob<=0:
                    continue  # 完全防止或者故障概率为0，成本为0
                # 减少效果
                reduction = 0
                for module in self.selected_modules:
                    if "reduces" in MODULES[module] and fault_type in MODULES[module]["reduces"]:
                        reduction = max(reduction, MODULES[module]["reduces"][fault_type])
                
                if(reduction>0):
                    raw_prob=params['probability']*(1-reduction)
                if(raw_prob!=0):
                    if timer["next_time"] == -1:
                        # 仿真的故障概率转每轮故障率
                        lambda_fail = -math.log(1 - raw_prob) / self.rounds_per_month
                        # lambda_fail=lambda_fail*scale   #由于概率太小导致故障为0，所以添加缩放因子
                        # 指数分布生成下次故障时间
                        timer["next_time"]=current_round_num + round(random.expovariate(lambda_fail))     

                    if(timer["next_time"]==current_round_num):
                        if(self.attempt_repair(node["id"],fault_type)==False):
                            timer["count"]+=1
                            # FAULT_TYPES[fault_type]["count"]+=1
                            node["fault"]=True
                            timer["last_fault_round"]=current_round_num
                            timer["fault_flag"]=1  
                        # 仿真的故障概率转每轮故障率
                        lambda_fail = -math.log(1 - raw_prob) / self.rounds_per_month 
                        # 指数分布生成下次故障时间
                        timer["next_time"]=current_round_num + round(random.expovariate(lambda_fail))                                   

    # 只有在预防性检查间隔的时候才会发现故障修改，或者心跳模块（出现预警电量/丢包个数大于阈值的时候再修改）
    def fixing_fault(self,current_round_num):
        for node in self.nodes:  
            if node["fault"]==True:
                for fault_type, params in FAULT_TYPES.items():
                    timer = node["fault_timers"][fault_type]
                    if timer["fault_flag"]==1:
                        #刚好检查到出现故障 或者 使用了心跳，刚好心跳丢失个数超过阈值
                        if (current_round_num % self.check_round == 0) or ("heartbeat" in self.selected_modules and current_round_num-timer["last_fault_round"]>=self.heartbeat_loss_threshold) :
                            node["fault"]=False
                            timer["fault_flag"]=0
                            timer["fault_round"]+=current_round_num-timer["last_fault_round"]   
                        

    def get_repair_modules(self, fault_type):
        """获取可用于修复指定故障的模块列表"""
        repair_modules = []
        for module_name in self.selected_modules:
            module = MODULES[module_name]
            if 'fixed_success' in module and fault_type in module['fixed_success']:
                repair_modules.append({
                    'name': module_name,
                    'success_rate': module['fixed_success'][fault_type]
                })
        return repair_modules
    
    def attempt_repair(self, node_id, fault_type):
        """尝试分级维护修复故障"""
        repair_modules = self.get_repair_modules(fault_type)
        result=False
        # 如果没有可用的维护模块，修复失败
        if not repair_modules:
            return result

        # 尝试第一个维护手段
        rand = random.random()
        if rand < repair_modules[0]['success_rate']:
            result=self.maintenance_modules_energy(node_id,repair_modules[0]['name'])
            return result
        
        # 如果第一个维护失败，尝试第二个维护手段（如果有）
        if len(repair_modules) > 1:
            rand = random.random()
            if rand < repair_modules[1]['success_rate']:
                result=self.maintenance_modules_energy(node_id,repair_modules[1]['name'])
                return result
        
        # 如果第二个维护失败，尝试第三个维护手段（如果有）（每个故障最多有三个可维护模块被修复）
        if len(repair_modules) > 2:
            rand = random.random()
            if rand < repair_modules[2]['success_rate']:
                result=self.maintenance_modules_energy(node_id,repair_modules[2]['name'])
                return result
        
        # 所有维护手段都失败
        return result

    def maintenance_modules_energy(self,node_id,module):
        result=True

        if module=="remote_restart" or module=="remote_reset" or module=="noise":    #远程
            if(module=="noise"):
                result=self.send_command_to_node(node_id,self.maintenance_noise_size)
            else:
                result=self.send_command_to_node(node_id,self.maintenance_instruction_size)
                
        if module=="short_restart" or module=="short_reset" or module=="activation":  #进程
            target_node = self._localizaObjetoCH(node_id, self.nodes)
            if(self.rts_cts_flag==1):
                #接收RTS包
                target_node["energy"] =  self._energy_consumption_rx(target_node["energy"],self.rts_cts_size)
                #发送CTS包
                target_node["energy"] = self._energy_consumption_tx(target_node["energy"],target_node["distance"],self.rts_cts_size)
            #接收数据
            target_node["energy"] = self._energy_consumption_rx(target_node["energy"],self.packet_size)    
            result=True

        return result

    def send_command_to_node(self, target_node_id, command_size=None):
        """
        基站发送指令到指定节点
        :param target_node_id: 目标节点ID
        :param cluster_heads: 当前簇头节点列表
        :param non_cluster_heads: 当前非簇头节点列表
        :param command_size: 指令数据包大小（字节），如果为None则使用默认包大小
        :return: 是否成功送达 (True/False)
        """
        if command_size is None:
            command_size = self.maintenance_instruction_size  
            
        # 查找目标节点
        target_node = self._localizaObjetoCH(target_node_id, self.nodes)

        # 获取从基站到目标节点的路径
        path = self._get_path_to_node(target_node)
        if not path:
            return False  # 没有找到有效路径
        
        # 沿着路径发送指令
        current_node = self.base_station  # 当前发送节点（基站开始）
        for next_node in path:
            # 基站发送不消耗能量，只考虑节点接收/发送的能耗
            #除最后一个节点外，其余节点都需发送
            if  next_node != target_node and current_node != self.base_station:
                # 当前节点（中继节点）发送数据
                if self.rts_cts_flag == 1:
                    # RTS/CTS握手
                    #发送RTS
                    current_node["energy"] = self._energy_consumption_tx(
                        current_node["energy"], 
                        self._distance(current_node["x"],current_node["y"], next_node["x"], next_node["y"]),
                        self.rts_cts_size
                    )
                    #接收CTS
                    current_node["energy"] = self._energy_consumption_rx(
                        current_node["energy"], 
                        self.rts_cts_size
                    )
                
                # 发送实际指令数据
                current_node["energy"] = self._energy_consumption_tx(
                    current_node["energy"], 
                    self._distance(current_node["x"],current_node["y"], next_node["x"], next_node["y"]),
                    command_size
                )
                
                # 检查中继节点能量
                if current_node["energy"] <= 0:
                    if "wireless_power" in self.selected_modules:
                        current_node["energy"] += self.initial_energy
                    else:
                        current_node["fault"] = True
                        return False
            #基站不接收数据
            if current_node != self.base_station:
                #收到RTS
                if(self.rts_cts_flag==1):
                    next_node["energy"] = self._energy_consumption_tx(
                        next_node["energy"], 
                        self._distance(current_node["x"],current_node["y"], next_node["x"], next_node["y"]),
                        self.rts_cts_size
                        )
                    #接收CTS
                    next_node["energy"] = self._energy_consumption_rx(
                        next_node["energy"], 
                        self.rts_cts_size
                    )
                    # 下一节点接收数据
                    next_node["energy"] = self._energy_consumption_rx(
                            next_node["energy"], 
                            command_size
                        )
                    
            # 检查接收节点能量
            if next_node["energy"] <= 0:
                if "wireless_power" in self.selected_modules:
                    next_node["energy"] += self.initial_energy
                else:
                    next_node["fault"] = True
                    return False
            
            current_node = next_node  # 移动到下一节点

        return True  # 指令成功送达目标节点


    # 辅助函数：获取从基站到目标节点的路径
    def _get_path_to_node(self, target_node):
        """
        构建从基站到目标节点的反向路径
        返回节点列表：[中继节点1, 中继节点2, ..., 目标节点]
        """
        path = [target_node]
        current = target_node
        # 添加安全检查 - 确保路由信息存在且有效
        if not current["routing_information"] or not current["routing_information"][0]:
            # 如果路由信息为空或无效，返回空路径
            return []
        # 沿着路由信息回溯到基站
        
        while current["routing_information"][0][0] != 0:  # 0表示基站
            next_hop_id = current["routing_information"][0][0]
            next_node = self._localizaObjetoCH(next_hop_id, self.nodes)
            
            if next_node is None:
                return []  # 路径中断
            
            path.append(next_node)
            current = next_node
        
        path.reverse()  
        return path
    

