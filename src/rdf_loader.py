import networkx as nx
import rdflib
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, FOAF, DC
import os
from typing import Dict, List, Any, Tuple

class RDFLoader:
    """
    RDF数据加载和处理类
    """
    
    def __init__(self):
        """初始化RDF加载器"""
        self.rdf_graph = None
        self.nx_graph = None
        self.subject_map = {}  # Maps URIs to node IDs
        self.object_map = {}
        self.property_map = {}
        self.reverse_mapping = {}  # Maps node IDs back to URIs
        
    def load_from_file(self, file_path: str, format: str = "turtle") -> None:
        """
        从文件加载RDF数据
        
        Args:
            file_path: RDF文件路径
            format: RDF格式 (turtle, xml, n3, nt, jsonld 等)
        """
        self.rdf_graph = Graph()
        self.rdf_graph.parse(file_path, format=format)
        self._convert_to_networkx()
        
    def load_from_string(self, data: str, format: str = "turtle") -> None:
        """
        从字符串加载RDF数据
        
        Args:
            data: RDF数据字符串
            format: RDF格式 (turtle, xml, n3, nt, jsonld 等)
        """
        self.rdf_graph = Graph()
        self.rdf_graph.parse(data=data, format=format)
        self._convert_to_networkx()
        
    def _convert_to_networkx(self) -> None:
        """将RDF图转换为NetworkX图"""
        if not self.rdf_graph:
            raise ValueError("RDF图未加载")
            
        self.nx_graph = nx.Graph()
        
        # 添加所有出现在三元组中的实体作为节点
        for s, p, o in self.rdf_graph:
            # 如果主语不在图中，添加它
            if s not in self.nx_graph:
                self.nx_graph.add_node(str(s))
                self._add_node_attributes(s)
                
            # 如果宾语是URI（而不是文字），且不在图中，添加它
            if isinstance(o, URIRef) and o not in self.nx_graph:
                self.nx_graph.add_node(str(o))
                self._add_node_attributes(o)
                
            # 如果主语和宾语都是URI，添加边
            if isinstance(o, URIRef):
                # 使用谓词作为边的属性
                if not self.nx_graph.has_edge(str(s), str(o)):
                    self.nx_graph.add_edge(str(s), str(o), property=str(p))
                    
    def _add_node_attributes(self, node: URIRef) -> None:
        """
        为节点添加属性信息
        
        Args:
            node: 要添加属性的节点URI
        """
        node_str = str(node)
        
        # 查找与此节点相关的所有三元组，其中节点是主语
        for _, p, o in self.rdf_graph.triples((node, None, None)):
            # 添加基本类型信息
            if p == RDF.type:
                self.nx_graph.nodes[node_str]['type'] = str(o)
                
            # 添加标签信息
            elif p == RDFS.label:
                self.nx_graph.nodes[node_str]['label'] = str(o)
            
            # 添加名称信息
            elif p == FOAF.name:
                self.nx_graph.nodes[node_str]['name'] = str(o)
                
            # 添加描述信息
            elif p == DC.description:
                self.nx_graph.nodes[node_str]['description'] = str(o)
                
            # 添加年龄信息
            elif p == FOAF.age:
                self.nx_graph.nodes[node_str]['age'] = str(o)
                
            # 添加其他属性，使用谓词的最后部分作为属性名
            else:
                prop_name = str(p).split('/')[-1]
                self.nx_graph.nodes[node_str][prop_name] = str(o)
                
    def get_graph(self) -> nx.Graph:
        """
        获取NetworkX图
        
        Returns:
            NetworkX图实例
        """
        if not self.nx_graph:
            raise ValueError("图未创建，请先加载RDF数据")
        return self.nx_graph
        
    def get_rdf_graph(self) -> Graph:
        """
        获取RDFLib的Graph对象，用于SPARQL查询
        
        Returns:
            RDFLib Graph实例
        """
        if not self.rdf_graph:
            raise ValueError("RDF图未加载，请先加载RDF数据")
        return self.rdf_graph
        
    def get_property_statistics(self) -> Dict:
        """
        Get statistics about properties in the graph
        
        Returns:
            Dictionary with property statistics
        """
        property_counts = {}
        
        for _, _, data in self.nx_graph.edges(data=True):
            prop = data.get('property', 'unknown')
            property_counts[prop] = property_counts.get(prop, 0) + 1
            
        return {
            'total_properties': len(self.property_map),
            'property_counts': property_counts
        }
        
    def get_node_info(self, node_id: int) -> Dict:
        """
        Get information about a node
        
        Args:
            node_id: ID of the node
            
        Returns:
            Dictionary with node information
        """
        if node_id not in self.reverse_mapping:
            return {}
            
        uri = self.reverse_mapping[node_id]
        neighbors = list(self.nx_graph.neighbors(node_id))
        
        return {
            'id': node_id,
            'uri': str(uri),
            'degree': len(neighbors),
            'neighbors': neighbors
        }
    
    def export_partitions_to_rdf(self, partitions: Dict[int, set], base_filename: str, format: str = "turtle") -> None:
        """
        将分区导出为RDF文件
        
        Args:
            partitions: 分区映射（分区ID到节点集合）
            base_filename: 基本文件名（将附加分区ID）
            format: 输出RDF格式
        """
        if not self.rdf_graph:
            raise ValueError("RDF图未加载")
            
        for p_id, nodes in partitions.items():
            # 为每个分区创建一个新的RDF图
            partition_graph = rdflib.Graph()
            
            # 复制命名空间
            for prefix, namespace in self.rdf_graph.namespaces():
                partition_graph.bind(prefix, namespace)
                
            # 找到与此分区中的节点相关的所有三元组
            for node in nodes:
                node_uri = URIRef(node)
                
                # 添加以该节点为主语的三元组
                for s, p, o in self.rdf_graph.triples((node_uri, None, None)):
                    partition_graph.add((s, p, o))
                    
                # 添加以该节点为宾语的三元组
                for s, p, o in self.rdf_graph.triples((None, None, node_uri)):
                    partition_graph.add((s, p, o))
            
            # 构建输出文件名
            output_filename = f"{base_filename}_{p_id}.{format}"
            
            # 序列化到文件
            partition_graph.serialize(destination=output_filename, format=format) 