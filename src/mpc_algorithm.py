import networkx as nx
import numpy as np
import random
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

class MPCAlgorithm:
    """
    最小属性切割(MPC)RDF图分区算法
    
    此算法通过最小化切割属性的数量来分区RDF图，同时平衡分区的大小。
    基于原始MPC论文和C++实现重写。
    """
    
    def __init__(self, graph: nx.Graph, num_partitions: int):
        """
        初始化MPC算法
        
        Args:
            graph: 表示RDF数据的NetworkX图
            num_partitions: 要创建的分区数量
        """
        self.graph = graph
        self.num_partitions = num_partitions
        
        # 基本属性
        self.entity_to_id = {}  # 实体到ID的映射
        self.id_to_entity = [""]  # ID到实体的映射（从索引1开始）
        self.predicate_to_id = {}  # 谓词到ID的映射
        self.id_to_predicate = [""]  # ID到谓词的映射（从索引1开始）
        
        # 图结构
        self.entity_count = 0  # 实体计数
        self.predicate_type_count = 0  # 谓词类型计数
        self.triples_count = 0  # 三元组计数
        self.invalid_edge_count = 0  # 无效边计数
        self.edge = [[]]  # 按谓词ID存储的边列表
        self.edge_cnt = {}  # 每种谓词的边计数
        self.entity_triples = [0]  # 每个实体的三元组数量
        self.subject_set_of_predicate = []  # 每个谓词的主语集合
        self.object_set_of_predicate = []  # 每个谓词的宾语集合
        
        # 分区结构
        self.partitions = {}  # 分区ID到实体集合的映射
        self.vertex_partition_map = {}  # 顶点到分区ID的映射
        self.coarsening_point = []  # 谓词的粗化点
        self.invalid = []  # 无效谓词标记
        
        # 算法参数
        self.limit = 0  # 分区大小限制
        self.predicate_set = set()  # 所有谓词的集合
        
    def load_graph_from_edges(self, edges: List[Tuple[Any, Any, Dict]]) -> None:
        """
        从边列表加载图
        
        Args:
            edges: 边列表，每个元素是(主语,宾语,属性字典)的元组
        """
        # 初始化
        self.subject_set_of_predicate.append(set())
        self.object_set_of_predicate.append(set())
        self.edge.append([])
        
        # 处理每条边
        for s, o, attrs in edges:
            self.triples_count += 1
            
            # 获取谓词
            if 'property' in attrs:
                p = attrs['property']
                self.predicate_set.add(p)
                self.edge_cnt[p] = self.edge_cnt.get(p, 0) + 1
            else:
                continue  # 跳过没有property属性的边
            
            # 处理实体ID映射，保留原始URI
            s_id = self.get_or_create_entity_id(s)
            o_id = self.get_or_create_entity_id(o)
            
            # 更新实体的三元组计数
            self.entity_triples[s_id] += 1
            
            # 处理谓词ID映射
            if p not in self.predicate_to_id:
                self.predicate_type_count += 1
                self.predicate_to_id[p] = self.predicate_type_count
                self.id_to_predicate.append(p)
                self.subject_set_of_predicate.append(set())
                self.object_set_of_predicate.append(set())
                self.edge.append([])
            
            pred_id = self.predicate_to_id[p]
            self.edge[pred_id].append((s_id, o_id))
            self.entity_triples[o_id] += 1
            self.subject_set_of_predicate[pred_id].add(s_id)
            self.object_set_of_predicate[pred_id].add(o_id)
        
        # 计算分区大小限制
        self.limit = self.entity_count // self.num_partitions // 2
        
        # 输出图信息
        print(f"实体数量: {self.entity_count}")
        print(f"谓词类型数量: {self.predicate_type_count}")
        print(f"三元组数量: {self.triples_count}")
        print(f"分区大小限制: {self.limit}")
        
    def get_or_create_entity_id(self, entity: Any) -> int:
        """
        获取实体ID或创建新ID
        
        Args:
            entity: 实体对象
            
        Returns:
            实体的内部ID
        """
        if entity not in self.entity_to_id:
            self.entity_count += 1
            self.entity_to_id[entity] = self.entity_count
            self.id_to_entity.append(entity)
            self.entity_triples.append(0)
        
        return self.entity_to_id[entity]
    
    def get_parent(self, son: int, fa: List[int]) -> int:
        """
        获取并查集中节点的根
        使用路径压缩优化
        
        Args:
            son: 子节点ID
            fa: 父节点列表
            
        Returns:
            根节点ID
        """
        k = son
        while k != fa[k]:
            k = fa[k]
            
        # 路径压缩
        i = son
        while i != k:
            j = fa[i]
            fa[i] = k
            i = j
            
        return k
    
    def get_parent_map(self, son: int, fa: Dict[int, int]) -> int:
        """
        获取字典存储的并查集中节点的根
        使用路径压缩优化
        
        Args:
            son: 子节点ID
            fa: 父节点字典
            
        Returns:
            根节点ID
        """
        k = son
        while k != fa[k]:
            k = fa[k]
            
        # 路径压缩
        i = son
        while i != k:
            j = fa[i]
            fa[i] = k
            i = j
            
        return k
    
    def coarsening(self) -> None:
        """
        图粗化过程，为每个谓词构建弱连通分量
        """
        self.invalid = [False] * (self.predicate_type_count + 1)
        self.coarsening_point = [defaultdict(int) for _ in range(self.predicate_type_count + 1)]
        
        for pred_id in range(1, self.predicate_type_count + 1):
            son_cnt = [1] * (self.entity_count + 1)
            rank = [0] * (self.entity_count + 1)
            
            # 初始化该谓词的并查集
            coarsen_map = defaultdict(int)
            for (a, b) in self.edge[pred_id]:
                # 将孤立点加入以自环形式
                if a not in coarsen_map:
                    coarsen_map[a] = a
                if b not in coarsen_map:
                    coarsen_map[b] = b
                
                parent_a = self.get_parent_map(a, coarsen_map)
                parent_b = self.get_parent_map(b, coarsen_map)
                
                if rank[parent_a] < rank[parent_b]:
                    parent_a, parent_b = parent_b, parent_a
                
                if parent_a != parent_b:
                    coarsen_map[parent_b] = parent_a
                    son_cnt[parent_a] += son_cnt[parent_b]
                    rank[parent_a] = max(rank[parent_a], rank[parent_b] + 1)
                    
                    # 如果某个WCC超过了大小限制，标记为无效
                    if son_cnt[parent_a] > self.limit:
                        self.invalid[pred_id] = True
                        self.invalid_edge_count += 1
                        print(f"无效谓词: {pred_id} {self.id_to_predicate[pred_id]}")
                        break
            
            self.coarsening_point[pred_id] = coarsen_map
            
    def _update_property_counts(self) -> Dict[str, int]:
        """
        更新跨越分区边界的属性计数
        
        Returns:
            属性到切割计数的映射
        """
        property_counts = {}
        
        for pred_id in range(1, self.predicate_type_count + 1):
            for u, v in self.edge[pred_id]:
                p_u = self.vertex_partition_map.get(u)
                p_v = self.vertex_partition_map.get(v)
                
                if p_u != p_v:  # 边跨越分区边界
                    prop = self.id_to_predicate[pred_id]
                    property_counts[prop] = property_counts.get(prop, 0) + 1
                    
        return property_counts
        
    def initialize_partitions(self) -> None:
        """
        随机初始化均匀分区
        """
        vertices = list(range(1, self.entity_count + 1))
        random.shuffle(vertices)
        
        # 将顶点均匀分配到各分区
        partition_size = len(vertices) // self.num_partitions
        remainder = len(vertices) % self.num_partitions
        
        start_idx = 0
        for i in range(self.num_partitions):
            size = partition_size + (1 if i < remainder else 0)
            end_idx = start_idx + size
            self.partitions[i] = set(vertices[start_idx:end_idx])
            
            # 将顶点映射到其分区
            for v in vertices[start_idx:end_idx]:
                self.vertex_partition_map[v] = i
                
            start_idx = end_idx
    
    def greed3(self) -> None:
        """
        基于属性重要性排序的贪心策略
        当谓词数量较多时使用
        """
        print("\n使用Greed3算法（基于WCC数量排序）\n")
        
        # 初始化迭代历史
        iteration_history = {
            'iterations': [],
            'property_cuts': [],
            'balance': [],
            'vertex_moves': [],
            'iteration_states': []  # 每次迭代的节点到分区的映射
        }
        
        self.invalid = [False] * (self.predicate_type_count + 1)
        threshold = int(self.entity_count * 0.0001)  # 小规模属性的阈值
        
        # 初始化并查集
        fa = list(range(self.entity_count + 1))
        FA = list(range(self.entity_count + 1))  # 最终结果的并查集
        RANK = [0] * (self.entity_count + 1)
        SONCNT = [1] * (self.entity_count + 1)  # 最终WCC的大小
        
        choice = [0] * (self.predicate_type_count + 1)  # 标记内部属性
        arr = []  # 存储(WCC数量, 谓词ID)对
        
        # 记录初始状态：所有节点未分配
        initial_mapping = {}
        for i in range(1, self.entity_count + 1):
            initial_mapping[i] = -1  # -1表示未分配分区
        
        iteration_history['iteration_states'].append(initial_mapping.copy())
        iteration_history['iterations'].append(0)
        iteration_history['property_cuts'].append(0)
        iteration_history['balance'].append(1.0)
        iteration_history['vertex_moves'].append(0)
        
        # 分析小规模属性 - 迭代1
        total_moves = 0
        small_pred_changes = {}
        
        # 重置临时连通分量ID为负数，避免与分区ID冲突
        wcc_id_counter = -100  # 从-100开始分配连通分量ID
        wcc_to_nodes = {}
        
        for pred_id in range(1, self.predicate_type_count + 1):
            # 如果属性的边数小于阈值，选择作为内部属性
            if len(self.edge[pred_id]) < threshold:
                moves_in_iteration = 0
                
                for a, b in self.edge[pred_id]:
                    parent_a = self.get_parent(a, FA)
                    parent_b = self.get_parent(b, FA)
                    
                    if RANK[parent_a] < RANK[parent_b]:
                        parent_a, parent_b = parent_b, parent_a
                        
                    if parent_a != parent_b:
                        FA[parent_b] = parent_a
                        SONCNT[parent_a] += SONCNT[parent_b]
                        RANK[parent_a] = max(RANK[parent_a], RANK[parent_b] + 1)
                        moves_in_iteration += 1
                        
                choice[pred_id] = 1
                small_pred_changes[pred_id] = moves_in_iteration
                total_moves += moves_in_iteration
        
        # 首次处理小规模属性后记录状态：标记连通分量
        phase1_mapping = {}
        # 为每个根节点分配唯一临时ID
        root_to_temp_id = {}
        
        for i in range(1, self.entity_count + 1):
            root = self.get_parent(i, FA)
            if root not in root_to_temp_id:
                root_to_temp_id[root] = wcc_id_counter
                wcc_id_counter -= 1
                wcc_to_nodes[root_to_temp_id[root]] = []
            
            # 使用临时ID而不是实际分区
            phase1_mapping[i] = root_to_temp_id[root]
            wcc_to_nodes[root_to_temp_id[root]].append(i)
        
        iteration_history['iteration_states'].append(phase1_mapping.copy())
        iteration_history['iterations'].append(1)
        iteration_history['property_cuts'].append(0)  # 尚未计算属性切割
        iteration_history['balance'].append(1.0)  # 尚未计算平衡度
        iteration_history['vertex_moves'].append(total_moves)
        
        # 对于中大规模谓词，计算WCC并排序
        iteration_count = 2
        arr_processing = []
        
        for pred_id in range(1, self.predicate_type_count + 1):
            if len(self.edge[pred_id]) >= threshold:
                # 对于较大的属性，计算其WCC数量
                parent = fa.copy()
                son_cnt = [1] * (self.entity_count + 1)
                invalid_pred = False
                
                for a, b in self.edge[pred_id]:
                    parent_a = self.get_parent(a, parent)
                    parent_b = self.get_parent(b, parent)
                    
                    if parent_a != parent_b:
                        parent[parent_b] = parent_a
                        son_cnt[parent_a] += son_cnt[parent_b]
                        
                        if son_cnt[parent_a] > self.limit:
                            self.invalid[pred_id] = True
                            invalid_pred = True
                            print(f"无效谓词: {pred_id}")
                            break
                            
                if invalid_pred:
                    continue
                    
                # 计算WCC数量
                wcc_cnt = 0
                for p in range(1, self.entity_count + 1):
                    if self.get_parent(p, parent) == p:
                        wcc_cnt += 1
                        
                arr.append((wcc_cnt, pred_id))
                arr_processing.append((wcc_cnt, pred_id))
        
        # 按WCC数量排序（降序）
        arr.sort()  # 升序排列
        
        # 记录分析谓词后的状态
        iteration_history['iteration_states'].append(phase1_mapping.copy())
        iteration_history['iterations'].append(iteration_count)
        iteration_history['property_cuts'].append(0)
        iteration_history['balance'].append(1.0)
        iteration_history['vertex_moves'].append(len(arr_processing))
        iteration_count += 1
        
        # 处理大规模谓词 - 每个谓词是一次迭代
        # 从WCC数量最多的开始处理，逆序遍历
        mid_mapping = phase1_mapping.copy()  # 用于跟踪中间状态
        
        for i in range(len(arr) - 1, -1, -1):
            pred_id = arr[i][1]
            moves_in_iteration = 0
            
            # 创建这次迭代的临时映射
            current_mapping = mid_mapping.copy()
            
            # 记录原始状态以计算变化
            prev_state = {node: root for node, root in current_mapping.items()}
            
            for a, b in self.edge[pred_id]:
                parent_a = self.get_parent(a, FA)
                parent_b = self.get_parent(b, FA)
                
                if RANK[parent_a] < RANK[parent_b]:
                    parent_a, parent_b = parent_b, parent_a
                    
                if parent_a != parent_b:
                    old_root_b = parent_b
                    FA[parent_b] = parent_a
                    SONCNT[parent_a] += SONCNT[parent_b]
                    RANK[parent_a] = max(RANK[parent_a], RANK[parent_b] + 1)
                    moves_in_iteration += 1
                    
                    # 更新映射：将属于old_root_b的所有节点重新映射到parent_a
                    for node, root in current_mapping.items():
                        if self.get_parent(node, FA) == parent_a:
                            # 更新只影响移动后的连通分量ID
                            temp_id = wcc_id_counter
                            current_mapping[node] = temp_id
                    
                    wcc_id_counter -= 1
                    
                    if SONCNT[parent_a] > self.limit:
                        self.invalid[pred_id] = True
                        break
                        
            if self.invalid[pred_id]:
                break
                
            choice[pred_id] = 1
            mid_mapping = current_mapping.copy()  # 更新中间状态
            
            # 记录每次处理谓词后的状态
            iteration_history['iteration_states'].append(current_mapping)
            iteration_history['iterations'].append(iteration_count)
            # 记录当前处理的谓词ID作为指标
            iteration_history['property_cuts'].append(pred_id)
            
            # 计算变化节点数
            changes = sum(1 for node in current_mapping if current_mapping[node] != prev_state.get(node, -1))
            
            # 计算当前连通分量的大小分布以估算平衡度
            current_wcc_sizes = {}
            for node, wcc_id in current_mapping.items():
                if wcc_id not in current_wcc_sizes:
                    current_wcc_sizes[wcc_id] = 0
                current_wcc_sizes[wcc_id] += 1
            
            if current_wcc_sizes:
                sizes = list(current_wcc_sizes.values())
                balance = max(sizes) / (sum(sizes) / len(sizes)) if len(sizes) > 0 else 1.0
            else:
                balance = 1.0
                
            iteration_history['balance'].append(balance)
            iteration_history['vertex_moves'].append(changes)
            iteration_count += 1
        
        # 统计跨边属性
        cross_edge = 0
        for pred_id in range(1, self.predicate_type_count + 1):
            if choice[pred_id] == 0:
                print(f"{pred_id}\t{self.id_to_predicate[pred_id]}")
                cross_edge += 1
                
        print(f"跨边属性数量: {cross_edge}")
        
        # 执行分区：这将把临时连通分量ID映射到最终分区ID
        self.union_block(choice, iteration_history)
        
        # 保存迭代历史
        self.iteration_history = iteration_history
    
    def union_block(self, choice: List[int], iteration_history=None) -> None:
        """
        将内部属性连接的组件分配到分区
        
        Args:
            choice: 标记内部属性的列表
            iteration_history: 用于记录迭代历史
        """
        print("执行分区...")
        
        # 初始化并查集
        parent = list(range(self.entity_count + 1))
        rank = [0] * (self.entity_count + 1)
        
        # 合并由内部属性连接的组件
        for pred_id in range(1, self.predicate_type_count + 1):
            if choice[pred_id] == 1:
                for a, b in self.edge[pred_id]:
                    parent_a = self.get_parent(a, parent)
                    parent_b = self.get_parent(b, parent)
                    
                    if rank[parent_a] < rank[parent_b]:
                        parent_a, parent_b = parent_b, parent_a
                        
                    if parent_a != parent_b:
                        rank[parent_a] = max(rank[parent_a], rank[parent_b] + 1)
                        parent[parent_b] = parent_a
                        self.entity_triples[parent_a] += self.entity_triples[parent_b]
        
        # 收集所有块
        block = []
        block_num = 0
        for p in range(1, self.entity_count + 1):
            if p == self.get_parent(p, parent):
                block.append((self.entity_triples[p], p))
                block_num += 1
                
        print(f"块数量: {block_num}")
        
        # 按三元组大小排序
        block.sort()
        
        # 使用贪心算法将块分配到分区
        # 采用最小堆方法，始终分配给当前大小最小的分区
        partition_sizes = [(0, i) for i in range(self.num_partitions)]
        block_to_partition = [0] * (self.entity_count + 1)
        
        # 从大到小分配块
        for i in range(len(block) - 1, -1, -1):
            # 获取当前最小分区
            partition_sizes.sort()  # 按分区大小排序
            partition_id = partition_sizes[0][1]
            
            # 分配块到分区
            block_id = block[i][1]
            block_size = block[i][0]
            block_to_partition[block_id] = partition_id
            
            # 更新分区大小
            partition_sizes[0] = (partition_sizes[0][0] + block_size, partition_id)
        
        # 更新分区映射
        self.partitions = {i: set() for i in range(self.num_partitions)}
        self.vertex_partition_map = {}
        
        for entity in range(1, self.entity_count + 1):
            entity_parent = self.get_parent(entity, parent)
            partition_id = block_to_partition[entity_parent]
            self.partitions[partition_id].add(entity)
            self.vertex_partition_map[entity] = partition_id
        
        # 记录最终分区状态
        if iteration_history is not None:
            final_mapping = {}
            for entity, partition_id in self.vertex_partition_map.items():
                final_mapping[entity] = partition_id
            
            iteration_count = len(iteration_history['iterations'])
            property_counts = self._update_property_counts()
            total_cuts = sum(property_counts.values())
            
            # 计算最终平衡度
            partition_sizes = [len(part) for part in self.partitions.values()]
            if min(partition_sizes) > 0:
                balance_ratio = max(partition_sizes) / min(partition_sizes)
            else:
                balance_ratio = float('inf')
            
            iteration_history['iteration_states'].append(final_mapping)
            iteration_history['iterations'].append(iteration_count)
            iteration_history['property_cuts'].append(total_cuts)
            iteration_history['balance'].append(balance_ratio)
            iteration_history['vertex_moves'].append(len(self.vertex_partition_map))
    
    def optimize(self) -> Dict:
        """
        运行MPC优化算法
        
        Returns:
            包含优化结果的字典
        """
        # 执行图粗化
        self.coarsening()
        
        # 初始化迭代历史
        self.iteration_history = {
            'iterations': [],
            'property_cuts': [],
            'balance': [],
            'vertex_moves': [],
            'iteration_states': []
        }
        
        # 根据谓词数量选择不同的算法
        if self.predicate_type_count <= 20:
            # 对于小规模数据，使用枚举法（未实现）
            print("谓词数量较少，应该使用枚举法，但当前未实现")
            self.greed3()
        elif self.predicate_type_count <= 40 and self.entity_count <= 20000000:
            # 使用相似度和WCC组合的贪心策略（未实现）
            print("谓词规模适中，应该使用组合贪心法，但当前未实现")
            self.greed3()
        elif self.predicate_type_count <= 120:
            # 仅使用WCC贪心策略（未实现）
            print("谓词规模较大，应该使用WCC贪心法，但当前未实现")
            self.greed3()
        else:
            # 对于大规模数据，使用基于属性排序的贪心策略
            self.greed3()
        
        # 计算结果指标
        property_counts = self._update_property_counts()
        
        return {
            'partitions': self.partitions,
            'vertex_mapping': self.vertex_partition_map,
            'property_cuts': property_counts,
            'history': self.iteration_history
        }
    
    def get_partition_metrics(self) -> Dict:
        """
        获取当前分区的指标
        
        Returns:
            包含分区指标的字典
        """
        property_counts = self._update_property_counts()
        total_cut_properties = sum(property_counts.values())
        unique_cut_properties = len(property_counts)
        
        # 确保在映射回原始实体后仍能保持正确的分区大小统计
        partition_sizes = [len(part) for part in self.partitions.values()]
        if min(partition_sizes) > 0:
            balance_ratio = max(partition_sizes) / min(partition_sizes)
        else:
            balance_ratio = float('inf')
        
        return {
            'total_cut_properties': total_cut_properties,
            'unique_cut_properties': unique_cut_properties,
            'partition_sizes': partition_sizes,
            'balance_ratio': balance_ratio
        } 