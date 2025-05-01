import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
import re
from collections import defaultdict, deque
import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.plugins.sparql import prepareQuery
from io import StringIO

class QueryEdge:
    """表示SPARQL查询中的一条边"""
    
    def __init__(self, source: str, target: str, predicate: str, is_variable_predicate: bool = False):
        """
        初始化查询边
        
        Args:
            source: 源节点（主语）
            target: 目标节点（宾语）
            predicate: 谓词
            is_variable_predicate: 谓词是否为变量
        """
        self.source = source
        self.target = target
        self.predicate = predicate
        self.is_variable_predicate = is_variable_predicate
    
    def __str__(self):
        return f"{self.source} --[{self.predicate}]--> {self.target}"
    
    def is_cross_partition(self, partition_map: Dict[Any, int]) -> bool:
        """
        判断边是否跨越分区
        
        Args:
            partition_map: 节点到分区ID的映射
            
        Returns:
            如果边跨越分区则返回True，否则返回False
        """
        # 如果源节点或目标节点是变量，则不能确定是否跨分区
        if self.source.startswith('?') or self.target.startswith('?'):
            return True
            
        # 如果谓词是变量，也视为跨分区边
        if self.is_variable_predicate:
            return True
            
        # 如果源节点和目标节点不在partition_map中，则视为可能跨分区
        if self.source not in partition_map or self.target not in partition_map:
            return True
            
        # 如果源节点和目标节点在不同分区，则边跨分区
        return partition_map[self.source] != partition_map[self.target]

class SubQuery:
    """表示一个子查询"""
    
    def __init__(self, id: int = 0):
        """
        初始化子查询
        
        Args:
            id: 子查询ID
        """
        self.id = id
        self.edges = []  # 子查询包含的边
        self.variables = set()  # 子查询包含的变量
        self.constants = set()  # 子查询包含的常量
    
    def add_edge(self, edge: QueryEdge):
        """
        添加边到子查询
        
        Args:
            edge: 要添加的边
        """
        self.edges.append(edge)
        
        # 更新变量和常量集合
        for node in [edge.source, edge.target]:
            if node.startswith('?'):
                self.variables.add(node)
            else:
                self.constants.add(node)
                
        # 如果谓词是变量，也添加到变量集合
        if edge.is_variable_predicate:
            self.variables.add(edge.predicate)
        elif edge.predicate.startswith('?'):
            self.variables.add(edge.predicate)
        else:
            self.constants.add(edge.predicate)
    
    def get_vertex_count(self) -> int:
        """
        获取子查询中的顶点数量
        
        Returns:
            顶点数量
        """
        return len(self.variables) + len(self.constants)
    
    def __str__(self):
        edges_str = '\n  '.join(str(edge) for edge in self.edges)
        return f"SubQuery {self.id} (顶点数: {self.get_vertex_count()}):\n  {edges_str}"

class SPARQLProcessor:
    """SPARQL查询处理器，用于分解和优化查询"""
    
    def __init__(self, partition_map: Dict[Any, int] = None):
        """
        初始化SPARQL处理器
        
        Args:
            partition_map: 节点到分区ID的映射，来自MPC算法结果
        """
        self.partition_map = partition_map or {}
        self.subqueries = []  # 分解后的子查询列表
        
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
        
    def parse_query(self, sparql_query: str) -> List[QueryEdge]:
        """
        解析SPARQL查询，提取三元组模式
        
        Args:
            sparql_query: SPARQL查询字符串
            
        Returns:
            查询边列表
        """
        self.start_step("SPARQL查询解析")
        
        self.log(f"原始查询:\n{sparql_query}")
        
        # 预处理查询：移除注释和不必要的空白
        lines = []
        for line in sparql_query.split('\n'):
            # 移除注释
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if line:  # 只保留非空行
                lines.append(line)
        
        # 重建查询文本
        clean_query = ' '.join(lines)
        self.log(f"清理后的查询:\n{clean_query}")
        
        # 处理前缀
        prefixes = {}
        prefix_pattern = r'PREFIX\s+(\w+):\s+<([^>]+)>'
        for match in re.finditer(prefix_pattern, clean_query, re.IGNORECASE):
            prefix = match.group(1)
            uri = match.group(2)
            prefixes[prefix] = uri
        
        self.log(f"识别到的前缀: {prefixes}")
        
        # 提取WHERE子句
        where_clause = ""
        if "WHERE" in clean_query:
            where_index = clean_query.upper().index("WHERE")
            where_clause = clean_query[where_index + 5:]
            # 移除首尾的花括号
            where_clause = where_clause.strip()
            if where_clause.startswith('{'):
                where_clause = where_clause[1:]
            if where_clause.endswith('}'):
                where_clause = where_clause[:-1]
            where_clause = where_clause.strip()
            
        self.log(f"WHERE子句:\n{where_clause}")
        
        # 更简单的方法来提取三元组
        edges = []
        # 按点号分割语句
        statements = where_clause.split('.')
        for statement in statements:
            statement = statement.strip()
            if not statement or statement.startswith('FILTER'):
                continue  # 跳过空语句和FILTER语句
                
            # 分割为主语、谓词、宾语
            parts = statement.strip().split()
            if len(parts) >= 3:
                source = self._resolve_uri_term(parts[0], prefixes)
                predicate = self._resolve_uri_term(parts[1], prefixes)
                target = self._resolve_uri_term(parts[2], prefixes)
                
                # 确定谓词是否为变量
                is_variable_predicate = predicate.startswith('?')
                
                edge = QueryEdge(source, target, predicate, is_variable_predicate)
                edges.append(edge)
                self.log(f"解析出边: {edge}")
        
        return edges
        
    def _resolve_uri_term(self, term: str, prefixes: Dict[str, str]) -> str:
        """
        解析URI术语，处理前缀
        
        Args:
            term: URI术语
            prefixes: 前缀映射
            
        Returns:
            解析后的URI
        """
        # 如果是变量，直接返回
        if term.startswith('?'):
            return term
            
        # 如果是完整URI（带尖括号），去掉尖括号
        if term.startswith('<') and term.endswith('>'):
            return term[1:-1]
            
        # 如果是前缀URI（形如prefix:name）
        if ':' in term:
            prefix, name = term.split(':', 1)
            if prefix in prefixes:
                return f"{prefixes[prefix]}{name}"
                
        # 其他情况，返回原始术语
        return term
    
    def decompose_query(self, sparql_query: str) -> List[SubQuery]:
        """
        将SPARQL查询分解为子查询
        
        Args:
            sparql_query: SPARQL查询字符串
            
        Returns:
            子查询列表
        """
        self.start_step("步骤1: 初始化空集合Q用于存储分解后的子查询")
        
        # 步骤1: 初始化一个空集合Q，用于存储分解后的子查询
        Q = []
        
        # 解析查询为边列表
        edges = self.parse_query(sparql_query)
        self.log(f"解析出 {len(edges)} 条边:")
        for i, edge in enumerate(edges):
            self.log(f"  边 {i+1}: {edge}")
        
        self.start_step("步骤2: 划分SPARQL查询中边属性为变量或跨越属性的边")
        
        # 步骤2: 划分SPARQL查询中边属性为变量或跨越属性的边，得到一组弱连通分量
        wcc_edges = []  # 用于计算WCC的边
        cross_partition_edges = []  # 跨分区或变量属性的边
        
        for edge in edges:
            if edge.is_variable_predicate or edge.is_cross_partition(self.partition_map):
                cross_partition_edges.append(edge)
                self.log(f"  跨越分区或变量属性的边: {edge}")
            else:
                wcc_edges.append(edge)
                self.log(f"  内部属性的边: {edge}")
        
        # 构建图来计算WCC
        G = nx.Graph()
        for edge in wcc_edges:
            G.add_edge(edge.source, edge.target, predicate=edge.predicate)
        
        # 获取弱连通分量
        wccs = list(nx.connected_components(G))
        
        self.log(f"得到 {len(wccs)} 个弱连通分量:")
        for i, wcc in enumerate(wccs):
            self.log(f"  WCC {i+1}: {', '.join(wcc)}")
        
        # 为每个WCC创建一个子查询
        wcc_subqueries = []
        for i, wcc in enumerate(wccs):
            subquery = SubQuery(i + 1)
            # 添加属于此WCC的边
            for edge in wcc_edges:
                if edge.source in wcc or edge.target in wcc:
                    subquery.add_edge(edge)
            wcc_subqueries.append(subquery)
        
        self.start_step("步骤3-5: 处理跨分区边或变量属性边")
        
        # 步骤3: 遍历SPARQL查询中变量属性或跨越属性的边
        if cross_partition_edges:
            # 如果有跨分区边，处理它们
            for edge in cross_partition_edges:
                self.log(f"\n处理边: {edge}")
                
                # 找到包含源节点和目标节点的子查询
                source_subqueries = []
                target_subqueries = []
                
                for sq in wcc_subqueries:
                    if edge.source in sq.variables or edge.source in sq.constants:
                        source_subqueries.append(sq)
                    if edge.target in sq.variables or edge.target in sq.constants:
                        target_subqueries.append(sq)
                
                # 步骤4: 如果源和目标在同一个子查询中
                common_subqueries = [sq for sq in source_subqueries if sq in target_subqueries]
                if common_subqueries:
                    sq = common_subqueries[0]
                    self.log(f"  源节点和目标节点在同一子查询中: {sq.id}")
                    sq.add_edge(edge)
                    self.log(f"  将边添加到子查询 {sq.id}")
                else:
                    # 步骤5: 比较顶点数量，决定将边添加到哪个查询
                    if source_subqueries and target_subqueries:
                        source_sq = source_subqueries[0]
                        target_sq = target_subqueries[0]
                        self.log(f"  源节点在子查询 {source_sq.id} 中，目标节点在子查询 {target_sq.id} 中")
                        
                        # 比较顶点数量
                        source_count = source_sq.get_vertex_count()
                        target_count = target_sq.get_vertex_count()
                        
                        if source_count <= target_count:
                            self.log(f"  源子查询顶点数 ({source_count}) <= 目标子查询顶点数 ({target_count})")
                            self.log(f"  将边添加到子查询 {target_sq.id}")
                            target_sq.add_edge(edge)
                        else:
                            self.log(f"  源子查询顶点数 ({source_count}) > 目标子查询顶点数 ({target_count})")
                            self.log(f"  将边添加到子查询 {source_sq.id}")
                            source_sq.add_edge(edge)
                    elif source_subqueries:
                        # 只有源节点在现有子查询中
                        sq = source_subqueries[0]
                        self.log(f"  只有源节点在子查询 {sq.id} 中")
                        self.log(f"  将边添加到子查询 {sq.id}")
                        sq.add_edge(edge)
                    elif target_subqueries:
                        # 只有目标节点在现有子查询中
                        sq = target_subqueries[0]
                        self.log(f"  只有目标节点在子查询 {sq.id} 中")
                        self.log(f"  将边添加到子查询 {sq.id}")
                        sq.add_edge(edge)
                    else:
                        # 两个节点都不在任何现有子查询中，创建新的子查询
                        new_sq = SubQuery(len(wcc_subqueries) + 1)
                        new_sq.add_edge(edge)
                        wcc_subqueries.append(new_sq)
                        self.log(f"  创建新子查询 {new_sq.id} 并添加边")
        
        self.start_step("步骤6: 将顶点数量大于1的子查询加入结果集")
        
        # 步骤6: 将顶点数量大于1的子查询加入结果集Q
        for sq in wcc_subqueries:
            vertex_count = sq.get_vertex_count()
            if vertex_count > 1:
                Q.append(sq)
                self.log(f"  将子查询 {sq.id} (顶点数: {vertex_count}) 加入结果集")
            else:
                self.log(f"  子查询 {sq.id} (顶点数: {vertex_count}) 只有一个查询点，不加入结果集")
        
        # 更新实例的子查询列表
        self.subqueries = Q
        
        self.log(f"\n最终分解得到 {len(Q)} 个子查询:")
        for sq in Q:
            self.log(f"\n{sq}")
        
        return Q
    
    def optimize_execution(self) -> Dict:
        """
        优化子查询的执行计划
        
        Returns:
            优化后的执行计划
        """
        self.start_step("优化子查询执行计划")
        
        # 如果没有分区映射，则无法优化
        if not self.partition_map:
            self.log("未提供分区映射，无法优化执行计划")
            return {"subqueries": self.subqueries}
        
        # 为每个子查询确定最优的执行分区
        execution_plan = {}
        for sq in self.subqueries:
            # 分析子查询中的常量节点所在分区
            partition_counts = defaultdict(int)
            for node in sq.constants:
                if node in self.partition_map:
                    partition_counts[self.partition_map[node]] += 1
            
            # 如果有节点在分区中，选择包含最多节点的分区
            if partition_counts:
                optimal_partition = max(partition_counts.items(), key=lambda x: x[1])[0]
                execution_plan[sq.id] = optimal_partition
                self.log(f"子查询 {sq.id} 将在分区 {optimal_partition} 执行")
            else:
                # 如果没有常量节点在分区中，随机选择一个分区
                execution_plan[sq.id] = 0
                self.log(f"子查询 {sq.id} 将在分区 0 执行（默认）")
        
        return {
            "subqueries": self.subqueries,
            "execution_plan": execution_plan
        }
    
    def generate_sparql_for_subquery(self, subquery: SubQuery) -> str:
        """
        为子查询生成SPARQL查询字符串
        
        Args:
            subquery: 子查询对象
            
        Returns:
            SPARQL查询字符串
        """
        # 添加通用前缀
        prefixes = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX ex: <http://example.org/>
"""
        
        # 构建SELECT子句
        select_vars = ' '.join(sorted(subquery.variables))
        
        # 构建WHERE子句
        where_clauses = []
        for edge in subquery.edges:
            source = edge.source if edge.source.startswith('?') else self._format_uri(edge.source)
            predicate = edge.predicate if edge.predicate.startswith('?') else self._format_uri(edge.predicate)
            target = edge.target if edge.target.startswith('?') else self._format_uri(edge.target)
            
            triple = f"{source} {predicate} {target} ."
            where_clauses.append(triple)
        
        where_body = '\n  '.join(where_clauses)
        
        # 完整的SPARQL查询
        sparql = f"""
{prefixes}
SELECT {select_vars}
WHERE {{
  {where_body}
}}
        """
        
        return sparql.strip()
    
    def _format_uri(self, uri: str) -> str:
        """
        格式化URI，添加合适的前缀或尖括号
        
        Args:
            uri: URI字符串
            
        Returns:
            格式化后的URI
        """
        # 处理特殊情况
        if uri == "a":
            return "a"
            
        # 处理已经包含前缀的URI
        if ":" in uri and not uri.startswith("http"):
            return uri
            
        # 处理全URI
        if uri.startswith("http"):
            # 检查是否可以用前缀替代
            if uri.startswith("http://xmlns.com/foaf/0.1/"):
                return f"foaf:{uri[len('http://xmlns.com/foaf/0.1/'):]}"
            elif uri.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#"):
                return f"rdf:{uri[len('http://www.w3.org/1999/02/22-rdf-syntax-ns#'):]}"
            elif uri.startswith("http://www.w3.org/2000/01/rdf-schema#"):
                return f"rdfs:{uri[len('http://www.w3.org/2000/01/rdf-schema#'):]}"
            elif uri.startswith("http://dbpedia.org/ontology/"):
                return f"dbo:{uri[len('http://dbpedia.org/ontology/'):]}"
            elif uri.startswith("http://example.org/"):
                return f"ex:{uri[len('http://example.org/'):]}"
            else:
                # 包装在尖括号中
                return f"<{uri}>"
        
        # 其他情况
        return uri
    
    def generate_all_subqueries(self) -> Dict[int, str]:
        """
        为所有子查询生成SPARQL查询字符串
        
        Returns:
            子查询ID到SPARQL字符串的映射
        """
        self.start_step("生成子查询SPARQL")
        
        result = {}
        for sq in self.subqueries:
            sparql = self.generate_sparql_for_subquery(sq)
            result[sq.id] = sparql
            self.log(f"\n子查询 {sq.id} 生成的SPARQL:\n{sparql}")
        
        return result
    
    def execute_subqueries(self, rdf_data: str = None, rdf_graph: Graph = None, format: str = "turtle") -> Dict[int, List[Dict]]:
        """
        执行所有子查询并返回结果
        
        Args:
            rdf_data: RDF数据字符串
            rdf_graph: 已加载的RDF图对象
            format: RDF数据格式
            
        Returns:
            子查询ID到结果的映射
        """
        self.start_step("执行子查询")
        
        # 确保我们有一个RDF图
        g = rdf_graph
        if g is None and rdf_data:
            g = Graph()
            g.parse(StringIO(rdf_data), format=format)
        
        if g is None:
            self.log("未提供RDF数据或图，无法执行查询")
            return {}
            
        # 执行每个子查询
        results = {}
        for sq in self.subqueries:
            sq_id = sq.id
            sparql = self.generate_sparql_for_subquery(sq)
            
            self.log(f"\n执行子查询 {sq_id}:")
            self.log(sparql)
            
            try:
                # 准备查询
                from rdflib.plugins.sparql import prepareQuery
                query = prepareQuery(sparql)
                
                # 执行查询
                query_results = g.query(query)
                
                # 将结果转换为字典列表
                rows = []
                for row in query_results:
                    row_dict = {}
                    for i, var in enumerate(query_results.vars):
                        var_name = str(var)
                        value = row[i] if i < len(row) else None
                        
                        # 转换RDF节点为字符串表示
                        if isinstance(value, URIRef):
                            value = str(value)
                        elif isinstance(value, Literal):
                            value = str(value)
                        elif isinstance(value, BNode):
                            value = f"_:{value}"
                            
                        row_dict[var_name] = value
                    rows.append(row_dict)
                
                results[sq_id] = rows
                self.log(f"找到 {len(rows)} 条结果")
                
                # 显示部分结果示例
                if rows:
                    self.log("结果示例:")
                    for i, row in enumerate(rows[:5]):  # 最多显示5行
                        row_str = ", ".join([f"{k}={v}" for k, v in row.items()])
                        self.log(f"  {i+1}. {row_str}")
                    if len(rows) > 5:
                        self.log(f"  ... 等共{len(rows)}行")
                
            except Exception as e:
                self.log(f"执行子查询 {sq_id} 时出错: {str(e)}")
                # 打印更详细的错误信息
                import traceback
                self.log(traceback.format_exc())
                results[sq_id] = []
                
        return results
    
    def execute_original_query(self, sparql_query: str, rdf_data: str = None, rdf_graph: Graph = None, format: str = "turtle") -> List[Dict]:
        """
        执行原始完整查询
        
        Args:
            sparql_query: 原始SPARQL查询
            rdf_data: RDF数据字符串
            rdf_graph: 已加载的RDF图对象
            format: RDF数据格式
            
        Returns:
            查询结果列表
        """
        self.start_step("执行原始完整查询")
        
        # 确保我们有一个RDF图
        g = rdf_graph
        if g is None and rdf_data:
            g = Graph()
            g.parse(StringIO(rdf_data), format=format)
        
        if g is None:
            self.log("未提供RDF数据或图，无法执行查询")
            return []
            
        self.log(f"执行原始查询:")
        self.log(sparql_query)
        
        try:
            # 准备查询
            from rdflib.plugins.sparql import prepareQuery
            query = prepareQuery(sparql_query)
            
            # 执行查询
            query_results = g.query(query)
            
            # 将结果转换为字典列表
            rows = []
            for row in query_results:
                row_dict = {}
                for i, var in enumerate(query_results.vars):
                    var_name = str(var)
                    value = row[i] if i < len(row) else None
                    
                    # 转换RDF节点为字符串表示
                    if isinstance(value, URIRef):
                        value = str(value)
                    elif isinstance(value, Literal):
                        value = str(value)
                    elif isinstance(value, BNode):
                        value = f"_:{value}"
                        
                    row_dict[var_name] = value
                rows.append(row_dict)
            
            self.log(f"找到 {len(rows)} 条结果")
            
            # 显示部分结果示例
            if rows:
                self.log("结果示例:")
                for i, row in enumerate(rows[:5]):  # 最多显示5行
                    row_str = ", ".join([f"{k}={v}" for k, v in row.items()])
                    self.log(f"  {i+1}. {row_str}")
                if len(rows) > 5:
                    self.log(f"  ... 等共{len(rows)}行")
            
            return rows
                
        except Exception as e:
            self.log(f"执行原始查询时出错: {str(e)}")
            # 打印更详细的错误信息
            import traceback
            self.log(traceback.format_exc())
            return [] 