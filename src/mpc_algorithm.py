import networkx as nx
import numpy as np
import random
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

class MPCAlgorithm:
    """
    最小属性切割(MPC)RDF图分区算法
    
    此算法通过最小化切割属性的数量来分区RDF图，同时平衡分区的大小。
    基于最小属性切割论文实现。
    """
    
    def __init__(self, graph: nx.Graph, num_partitions: int, epsilon: float = 0.1):
        """
        初始化MPC算法
        
        Args:
            graph: 表示RDF数据的NetworkX图
            num_partitions: 要创建的分区数量
            epsilon: 平衡因子，控制分区大小的不平衡度
        """
        self.graph = graph
        self.num_partitions = num_partitions
        self.epsilon = epsilon
        
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
        self.Lin = []  # 内部属性集合
        
        # 日志和统计
        self.log_buffer = []  # 存储所有日志消息
        self.step_logs = {}   # 按步骤存储日志
        self.current_step = "初始化"
    
    def log(self, message: str) -> None:
        """记录日志消息"""
        print(message)
        self.log_buffer.append(message)
        
        # 将消息添加到当前步骤的日志中
        if self.current_step not in self.step_logs:
            self.step_logs[self.current_step] = []
        self.step_logs[self.current_step].append(message)
    
    def start_step(self, step_name: str) -> None:
        """开始一个新的步骤，添加分隔符并设置当前步骤名称"""
        separator = "\n" + "="*57 + "\n"
        self.log(separator)
        self.log(step_name)
        self.log(separator)
        self.current_step = step_name
        
    def load_graph_from_edges(self, edges: List[Tuple[Any, Any, Dict]]) -> None:
        """
        从边列表加载图（步骤1：读取原始RDF数据图G，并将边属性保存到集合L中）
        
        Args:
            edges: 边列表，每个元素是(主语,宾语,属性字典)的元组
        """
        self.start_step("步骤1: 读取原始RDF数据图G，将边属性保存到集合L中")
        
        # 初始化
        self.subject_set_of_predicate.append(set())
        self.object_set_of_predicate.append(set())
        self.edge.append([])
        
        # 记录边的详细信息
        edge_details = []
        
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
            
            # 收集边详情
            edge_details.append({
                "source_id": s_id,
                "source": s,
                "target_id": o_id,
                "target": o,
                "predicate": p
            })
            
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
        
        # 根据epsilon计算分区大小限制
        total_vertices = self.entity_count
        self.limit = int((1 + self.epsilon) * total_vertices / self.num_partitions)
        
        # 输出图信息
        self.log(f"实体数量: {self.entity_count}")
        self.log(f"谓词类型数量: {self.predicate_type_count}")
        self.log(f"三元组数量: {self.triples_count}")
        self.log(f"分区大小限制 (1+ε)×|V|/k: {self.limit}")
        
        # 按谓词分组记录所有边
        self.log("\n所有谓词及其关系:")
        predicates_groups = {}
        for edge in edge_details:
            pred = edge["predicate"]
            if pred not in predicates_groups:
                predicates_groups[pred] = []
            predicates_groups[pred].append(edge)
        
        # 详细记录所有边
        for pred, pred_edges in predicates_groups.items():
            self.log(f"\n谓词: {pred} (总共 {len(pred_edges)} 条边)")
            for i, edge in enumerate(pred_edges):
                source = edge["source"]
                target = edge["target"]
                self.log(f"  边 {i+1}: {source} -> {target}")
        
        # 全部边的列表形式记录
        self.log("\n所有边列表 (每行一条三元组):")
        for edge in edge_details:
            self.log(f"{edge['source']} --[{edge['predicate']}]--> {edge['target']}")
    
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
    
    def calculate_wcc_and_cost(self) -> List[Tuple[int, int, float]]:
        """
        步骤2: 遍历属性集合L，计算每个属性p对应的弱连通组件分量WCC(G(p))以及对应的代价Cost({p})
        
        Returns:
            包含(WCC数量, 谓词ID, 代价)的列表
        """
        self.start_step("步骤2: 计算每个属性p的弱连通组件数量WCC(G(p))及其代价")
        
        result = []
        
        for pred_id in range(1, self.predicate_type_count + 1):
            pred_name = self.id_to_predicate[pred_id]
            edge_count = len(self.edge[pred_id])
            
            # 对每个谓词计算WCC
            self.log(f"\n分析谓词 {pred_id}: {pred_name} (边数: {edge_count})")
            
            # 初始化并查集
            parent = list(range(self.entity_count + 1))
            son_cnt = [1] * (self.entity_count + 1)
            
            # 计算谓词p的弱连通分量
            invalid_pred = False
            
            # 收集所有可能涉及的节点
            involved_nodes = set()
            for a, b in self.edge[pred_id]:
                involved_nodes.add(a)
                involved_nodes.add(b)
                
            # 处理每条边
            for idx, (a, b) in enumerate(self.edge[pred_id]):
                parent_a = self.get_parent(a, parent)
                parent_b = self.get_parent(b, parent)
                
                entity_a = self.id_to_entity[a] if a < len(self.id_to_entity) else f"未知实体{a}"
                entity_b = self.id_to_entity[b] if b < len(self.id_to_entity) else f"未知实体{b}"
                
                if parent_a != parent_b:
                    parent[parent_b] = parent_a
                    son_cnt[parent_a] += son_cnt[parent_b]
                    
                    self.log(f"  合并: {a}({entity_a}) -> {b}({entity_b})")
                    self.log(f"  连通分量 {parent_a} 更新大小: {son_cnt[parent_a]}")
                    
                    # 检查大小限制
                    if son_cnt[parent_a] > self.limit:
                        invalid_pred = True
                        self.log(f"  谓词 {pred_id} 超出大小限制 ({son_cnt[parent_a]} > {self.limit})，标记为无效")
                        break
            
            if invalid_pred:
                # 标记为无效，不添加到结果中
                continue
                
            # 计算WCC数量 - 只考虑涉及到的节点
            wcc_cnt = 0
            wcc_roots = set()
            for node in involved_nodes:
                root = self.get_parent(node, parent)
                if root not in wcc_roots:
                    wcc_roots.add(root)
                    wcc_cnt += 1
            
            # 计算Cost(p) = 边数量 / WCC数量的比值
            # 这个比值越低意味着这个谓词将图分割得越好，代价越低
            if wcc_cnt > 0:  # 避免除以零
                cost = edge_count / wcc_cnt
            else:
                cost = float('inf')
                
            self.log(f"  谓词 {pred_id} 的WCC数量: {wcc_cnt}")
            self.log(f"  谓词 {pred_id} 的代价 (边数/WCC数): {cost:.2f}")
            
            result.append((wcc_cnt, pred_id, cost))
        
        # 按照代价排序（从小到大）
        result.sort(key=lambda x: x[2])
        
        self.log("\n属性代价排序结果:")
        for wcc_cnt, pred_id, cost in result:
            self.log(f"  谓词 {pred_id} ({self.id_to_predicate[pred_id]}): WCC={wcc_cnt}, 代价={cost:.2f}")
            
        return result
    
    def select_internal_properties(self, sorted_predicates: List[Tuple[int, int, float]]) -> List[int]:
        """
        步骤3: 尽可能从属性集合L中选择更多的内部属性Lin
        
        Args:
            sorted_predicates: 按照代价排序的谓词列表(WCC数量, 谓词ID, 代价)
            
        Returns:
            选中的内部属性ID列表
        """
        self.start_step("步骤3: 选择内部属性Lin，构建粗化图")
        
        internal_predicates = []
        
        # 初始化合并映射和大小
        merged_blocks = list(range(self.entity_count + 1))
        block_sizes = [1] * (self.entity_count + 1)
        
        # 按照代价从小到大处理谓词（贪心策略）
        for wcc_cnt, pred_id, cost in sorted_predicates:
            pred_name = self.id_to_predicate[pred_id]
            self.log(f"\n考虑谓词 {pred_id}: {pred_name} (WCC={wcc_cnt}, 代价={cost:.2f})")
            
            # 检查将此谓词添加为内部属性是否会超过大小限制
            can_add = True
            temp_merged = merged_blocks.copy()
            temp_sizes = block_sizes.copy()
            
            # 模拟合并
            for a, b in self.edge[pred_id]:
                parent_a = self.get_parent(a, temp_merged)
                parent_b = self.get_parent(b, temp_merged)
                
                if parent_a != parent_b:
                    # 合并两个块
                    temp_merged[parent_b] = parent_a
                    temp_sizes[parent_a] += temp_sizes[parent_b]
                    
                    # 检查大小限制
                    if temp_sizes[parent_a] > self.limit:
                        can_add = False
                        self.log(f"  添加谓词 {pred_id} 将导致块大小 {temp_sizes[parent_a]} 超过限制 {self.limit}，跳过")
                        break
            
            if can_add:
                # 真正添加为内部属性
                internal_predicates.append(pred_id)
                
                # 更新合并状态
                for a, b in self.edge[pred_id]:
                    parent_a = self.get_parent(a, merged_blocks)
                    parent_b = self.get_parent(b, merged_blocks)
                    
                    if parent_a != parent_b:
                        # 合并两个块
                        merged_blocks[parent_b] = parent_a
                        block_sizes[parent_a] += block_sizes[parent_b]
                        self.log(f"  合并块: {parent_b} -> {parent_a}, 新大小: {block_sizes[parent_a]}")
                
                self.log(f"  将谓词 {pred_id} 添加为内部属性")
            
        # 找出所有超点（代表弱连通分量的根节点）
        supernodes = {}
        for i in range(1, self.entity_count + 1):
            root = self.get_parent(i, merged_blocks)
            if root not in supernodes:
                supernodes[root] = set()
            supernodes[root].add(i)
        
        self.log(f"\n共选择了 {len(internal_predicates)} 个内部属性，形成了 {len(supernodes)} 个超点")
        
        # 展示一些超点信息
        self.log("\n超点示例:")
        for i, (root, nodes) in enumerate(list(supernodes.items())[:5]):
            self.log(f"  超点 {i+1} (根: {root}, 大小: {len(nodes)})")
            node_examples = list(nodes)[:5]
            node_names = [f"{n}({self.id_to_entity[n]})" for n in node_examples]
            self.log(f"    包含节点: {', '.join(node_names)}" + 
                  (f"... 等共{len(nodes)}个" if len(nodes) > 5 else ""))
        
        # 存储超点信息和内部属性，供后续步骤使用
        self.supernodes = supernodes
        self.merged_blocks = merged_blocks
        self.block_sizes = block_sizes
        self.Lin = internal_predicates
        
        return internal_predicates
    
    def partition_supernodes(self):
        """
        步骤4: 对粗化图中的超点使用自顶向下分区算法
        """
        self.start_step("步骤4: 对粗化图中的超点使用自顶向下分区算法")
        
        # 收集所有超点（连通分量的根节点）
        supernodes = []
        for root, size in enumerate(self.block_sizes):
            if root > 0 and self.get_parent(root, self.merged_blocks) == root:
                # 这是一个超点的根
                supernodes.append((size, root))
        
        # 按大小排序（从大到小）
        supernodes.sort(reverse=True)
        
        # 初始化分区
        partition_sizes = [0] * self.num_partitions
        supernode_to_partition = {}
        
        self.log("按大小顺序分配超点:")
        for size, root in supernodes:
            # 找出最小的分区
            target_partition = partition_sizes.index(min(partition_sizes))
            
            # 分配超点到分区
            supernode_to_partition[root] = target_partition
            partition_sizes[target_partition] += size
            
            # 显示分配情况
            self.log(f"  分配超点 {root} (大小: {size}) 到分区 {target_partition}")
            self.log(f"  分区 {target_partition} 新大小: {partition_sizes[target_partition]}")
        
        # 检查分区平衡性
        max_size = max(partition_sizes)
        min_size = min(partition_sizes)
        if min_size > 0:
            balance_ratio = max_size / min_size
            self.log(f"\n分区平衡度: {balance_ratio:.2f}")
            
            # 检查每个分区是否满足限制条件
            max_allowed = int((1 + self.epsilon) * self.entity_count / self.num_partitions)
            for i, size in enumerate(partition_sizes):
                self.log(f"  分区 {i}: 大小 {size}/{max_allowed}" + 
                      (f" (超出限制 {size - max_allowed})" if size > max_allowed else ""))
        else:
            self.log("\n警告: 存在空分区")
        
        # 存储超点到分区的映射
        self.supernode_to_partition = supernode_to_partition
        
        return supernode_to_partition
    
    def map_to_original_graph(self):
        """
        步骤5: 将超点集合映射回原始数据图得到最终分区
        """
        self.start_step("步骤5: 将超点集合映射回原始数据图得到最终分区")
        
        # 初始化分区
        self.partitions = {i: set() for i in range(self.num_partitions)}
        self.vertex_partition_map = {}
        
        # 将每个节点映射到其分区
        for entity in range(1, self.entity_count + 1):
            # 找到实体所属的超点
            supernode_root = self.get_parent(entity, self.merged_blocks)
            
            # 获取超点的分区
            partition_id = self.supernode_to_partition.get(supernode_root, 0)  # 默认分配到分区0
            
            # 更新分区映射
            self.partitions[partition_id].add(entity)
            self.vertex_partition_map[entity] = partition_id
            
            # 获取实体名称用于显示
            entity_name = self.id_to_entity[entity] if entity < len(self.id_to_entity) else f"未知实体{entity}"
            
            # 记录日志（只显示前几个）
            if len(self.partitions[partition_id]) <= 5:
                self.log(f"  节点 {entity}({entity_name}) 映射到分区 {partition_id}")
        
        # 显示分区大小
        self.log("\n最终分区大小:")
        for p_id, entities in self.partitions.items():
            self.log(f"  分区 {p_id}: {len(entities)} 个节点")
        
        # 统计属性切割
        cuts = self._update_property_counts()
        cut_count = sum(cuts.values())
        unique_cut_props = len(cuts)
        
        self.log(f"\n属性切割结果:")
        self.log(f"  总切割次数: {cut_count}")
        self.log(f"  唯一切割属性数: {unique_cut_props}")
        
        # 显示每个被切割的属性
        self.log("\n被切割的属性:")
        for pred, count in sorted(cuts.items(), key=lambda x: x[1], reverse=True)[:10]:
            self.log(f"  {pred}: {count} 次切割")
            
        return {
            'partitions': self.partitions,
            'vertex_mapping': self.vertex_partition_map,
            'property_cuts': cuts
        }
    
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
    
    def optimize(self) -> Dict:
        """
        运行MPC优化算法
        
        Returns:
            包含优化结果的字典
        """
        self.start_step("开始执行MPC算法")
        
        # 步骤2: 计算每个属性的WCC和代价
        sorted_predicates = self.calculate_wcc_and_cost()
        
        # 步骤3: 选择内部属性，构建粗化图
        internal_predicates = self.select_internal_properties(sorted_predicates)
        
        # 步骤4: 对超点进行分区
        self.partition_supernodes()
        
        # 步骤5: 映射回原始图
        result = self.map_to_original_graph()
        
        # 计算结果指标
        property_counts = self._update_property_counts()
        
        # 记录最终结果
        final_stats = f"\n算法执行完成，最终结果:\n"
        final_stats += f"- 总分区数: {self.num_partitions}\n"
        final_stats += f"- 总切割属性数: {sum(property_counts.values())}\n"
        final_stats += f"- 唯一切割属性数: {len(property_counts)}\n"
        
        # 计算平衡度
        sizes = [len(part) for part in self.partitions.values()]
        if min(sizes) > 0:
            balance = max(sizes) / min(sizes)
            final_stats += f"- 分区大小平衡度: {balance:.2f}\n"
        
        self.log(final_stats)
        
        # 返回结果和日志
        result['logs'] = self.log_buffer
        result['step_logs'] = self.step_logs
        
        return result
    
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
            'balance_ratio': balance_ratio,
            'logs': self.log_buffer,
            'step_logs': self.step_logs
        } 