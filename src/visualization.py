import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class MPCVisualizer:
    """
    Visualization tools for MPC algorithm results
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize visualizer
        
        Args:
            graph: NetworkX graph to visualize
        """
        self.graph = graph
        self.layout = None
        self.partition_colors = {
            0: "#1f77b4",  # blue
            1: "#ff7f0e",  # orange
            2: "#2ca02c",  # green
            3: "#d62728",  # red
            4: "#9467bd",  # purple
            5: "#8c564b",  # brown
            6: "#e377c2",  # pink
            7: "#7f7f7f",  # gray
            8: "#bcbd22",  # olive
            9: "#17becf",  # cyan
            -1: "#cccccc",  # light gray (未分配)
        }
        
        # 临时分区ID使用的颜色序列 (用于迭代过程中的临时连通分量)
        self.temp_colors = [
            "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", 
            "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", 
            "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"
        ]
        
    def compute_layout(self, algorithm: str = "spring") -> None:
        """
        Compute graph layout for visualization
        
        Args:
            algorithm: Layout algorithm ('spring', 'kamada_kawai', 'spectral', etc.)
        """
        if algorithm == "spring":
            self.layout = nx.spring_layout(self.graph, seed=42)
        elif algorithm == "kamada_kawai":
            self.layout = nx.kamada_kawai_layout(self.graph)
        elif algorithm == "spectral":
            self.layout = nx.spectral_layout(self.graph)
        else:
            self.layout = nx.spring_layout(self.graph, seed=42)
    
    def get_node_label(self, node):
        """获取节点的标签或名称"""
        # 先检查常见的标签属性
        for attr in ["label", "name", "rdfs:label", "foaf:name"]:
            if attr in self.graph.nodes[node]:
                return self.graph.nodes[node][attr]
        
        # 如果没有常见标签，返回节点本身的简短表示
        if isinstance(node, str) and "/" in node:
            return node.split("/")[-1]
        return str(node)
    
    def is_meaningful_node(self, node):
        """
        判断节点是否有意义（有有效标签或属性，且类型已知）
        
        Args:
            node: 要检查的节点
            
        Returns:
            布尔值，表示节点是否有意义
        """
        # 检查节点是否有类型
        node_type = self.get_node_type(node)
        if node_type == "未知":
            return False
        
        # 如果节点有任何标签属性，认为是有意义的
        for attr in ["label", "name", "rdfs:label", "foaf:name"]:
            if attr in self.graph.nodes[node]:
                return True
        
        # 如果节点有任何属性，也认为是有意义的
        if len(self.graph.nodes[node]) > 0:
            return True
            
        # 如果节点有多个连接，也认为是有意义的
        if self.graph.degree(node) > 1:
            return True
            
        # 没有标签、属性且连接少，认为无意义
        return False
        
    def filter_meaningless_nodes(self):
        """
        从图中移除无意义的节点
        
        Returns:
            经过过滤的新图
        """
        filtered_graph = self.graph.copy()
        nodes_to_remove = []
        
        for node in filtered_graph.nodes():
            if not self.is_meaningful_node(node):
                nodes_to_remove.append(node)
                
        filtered_graph.remove_nodes_from(nodes_to_remove)
        
        print(f"过滤前节点数: {len(self.graph.nodes())}")
        print(f"过滤后节点数: {len(filtered_graph.nodes())}")
        print(f"移除了 {len(nodes_to_remove)} 个无意义节点")
        
        return filtered_graph
    
    def plot_graph(self, partition_map: Dict = None, title: str = "RDF图可视化", filter_nodes: bool = False, show_edge_labels: bool = True) -> go.Figure:
        """
        Create a Plotly figure of the graph
        
        Args:
            partition_map: Dictionary mapping vertex to partition ID
            title: Title for the plot
            filter_nodes: 是否过滤无意义的节点
            show_edge_labels: 是否显示边的标签
            
        Returns:
            Plotly figure object
        """
        if self.layout is None:
            self.compute_layout()
        
        # 如果需要，过滤无意义节点
        display_graph = self.filter_meaningless_nodes() if filter_nodes else self.graph
        
        # 重新计算布局（如果过滤了节点）
        if filter_nodes:
            temp_vis = MPCVisualizer(display_graph)
            temp_vis.compute_layout()
            temp_layout = temp_vis.layout
        else:
            temp_layout = self.layout
        
        # 调试信息
        if partition_map:
            print(f"传入的partition_map长度: {len(partition_map)}")
            print(f"partition_map前5个元素: {list(partition_map.items())[:5]}")
            print(f"display_graph.nodes前5个元素: {list(display_graph.nodes())[:5]}")
            
            # 检查节点与分区映射的匹配情况
            match_count = 0
            for node in display_graph.nodes():
                if node in partition_map:
                    match_count += 1
            print(f"节点在partition_map中的匹配率: {match_count}/{len(display_graph.nodes())}")
        
        # 为边分类，按谓词分组
        edge_by_predicate = {}
        for u, v, data in display_graph.edges(data=True):
            pred = data.get("property", "unknown")
            if pred not in edge_by_predicate:
                edge_by_predicate[pred] = []
            edge_by_predicate[pred].append((u, v, data))
            
        # 为每种谓词创建不同颜色的边
        edge_traces = []
        
        # 预定义一些颜色供边使用
        edge_colors = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", 
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
        ]
        
        # 创建一个图例以显示边的类型
        edge_legend_traces = []
        
        for i, (pred, edges) in enumerate(edge_by_predicate.items()):
            # 使用颜色循环
            color = edge_colors[i % len(edge_colors)]
            
            # 创建这种类型边的数据点
            edge_x = []
            edge_y = []
            edge_text = []
            
            # 为每条边创建文本和坐标
            for u, v, data in edges:
                x0, y0 = temp_layout[u]
                x1, y1 = temp_layout[v]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # 准备边的悬停文本
                source_label = self.get_node_label(u)
                target_label = self.get_node_label(v)
                edge_text.append(f"关系: {pred}<br>从: {source_label}<br>到: {target_label}")
                edge_text.append(f"关系: {pred}<br>从: {source_label}<br>到: {target_label}")
                edge_text.append(None)  # 为None点添加空文本
            
            # 创建这种类型的边的轨迹
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.5, color=color),
                hoverinfo="text",
                text=edge_text,
                mode="lines",
                name=self.get_short_predicate_name(pred)
            )
            
            edge_traces.append(edge_trace)
            
            # 添加箭头来表示边的方向
            for u, v, data in edges:
                x0, y0 = temp_layout[u]
                x1, y1 = temp_layout[v]
                
                # 计算边的方向向量
                dx = x1 - x0
                dy = y1 - y0
                
                # 边的长度
                length = (dx**2 + dy**2)**0.5
                
                if length == 0:  # 避免除以零
                    continue
                
                # 单位向量
                udx = dx / length
                udy = dy / length
                
                # 将箭头放在边的80%处，避免与节点重合
                arrow_pos = 0.8
                arrow_x = x0 + arrow_pos * dx
                arrow_y = y0 + arrow_pos * dy
                
                # 箭头大小基于边长度
                arrow_size = min(0.02 * length, 0.3)
                
                # 创建箭头
                arrow_trace = go.Scatter(
                    x=[arrow_x, arrow_x - arrow_size * (udx * 0.7 + udy * 0.7), arrow_x - arrow_size * (udx * 0.7 - udy * 0.7), arrow_x],
                    y=[arrow_y, arrow_y - arrow_size * (udy * 0.7 - udx * 0.7), arrow_y - arrow_size * (udy * 0.7 + udx * 0.7), arrow_y],
                    mode='lines',
                    line=dict(color=color, width=1.5),
                    fill='toself',
                    fillcolor=color,
                    hoverinfo='none',
                    showlegend=False
                )
                
                edge_traces.append(arrow_trace)
            
            # 添加一个隐藏的轨迹用于图例
            legend_trace = go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(width=2, color=color),
                name=self.get_short_predicate_name(pred)
            )
            edge_legend_traces.append(legend_trace)
        
        # 创建边标签，如果需要
        edge_label_traces = []
        if show_edge_labels:
            # 为每种谓词类型创建标签
            for pred, edges in edge_by_predicate.items():
                # 为避免标签过多，只为每种类型选择部分边
                sample_edges = edges[:min(len(edges), 5)]  # 每种类型最多5个标签
                
                for u, v, data in sample_edges:
                    x0, y0 = temp_layout[u]
                    x1, y1 = temp_layout[v]
                    
                    # 计算边的中点位置
                    mid_x = (x0 + x1) / 2
                    mid_y = (y0 + y1) / 2
                    
                    # 创建标签轨迹
                    label_trace = go.Scatter(
                        x=[mid_x],
                        y=[mid_y],
                        mode="text",
                        text=[self.get_short_predicate_name(pred)],
                        textposition="middle center",
                        textfont=dict(size=9, color="#555"),
                        hoverinfo="none",
                        showlegend=False
                    )
                    edge_label_traces.append(label_trace)
        
        # Create node traces
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node in display_graph.nodes():
            x, y = temp_layout[node]
            node_x.append(x)
            node_y.append(y)
            
            # Assign color based on partition
            if partition_map and node in partition_map:
                partition = partition_map[node]
                if partition >= 0:
                    # 正常分区ID
                    color = self.partition_colors.get(partition, "#000000")
                else:
                    # 负数ID (临时连通分量或未分配)
                    if partition == -1:
                        color = "#CCCCCC"  # 未分配
                    else:
                        # 使用临时ID的颜色
                        color_idx = abs(partition) % len(self.temp_colors)
                        color = self.temp_colors[color_idx]
            else:
                color = "#1f77b4"  # default blue
                
            node_colors.append(color)
            
            # Node size based on degree
            size = 10 + self.graph.degree(node)
            node_sizes.append(size)
            
            # 获取节点详细信息
            node_label = self.get_node_label(node)
            node_type = self.get_node_type(node)
            
            # Node text for hover
            text = f"节点: {node_label}<br>类型: {node_type}<br>连接数: {self.graph.degree(node)}"
            if partition_map and node in partition_map:
                if partition_map[node] >= 0:
                    text += f"<br>分区: {partition_map[node]}"
                else:
                    if partition_map[node] == -1:
                        text += "<br>分区: 未分配"
                    else:
                        text += f"<br>临时连通分量: {partition_map[node]}"
                
            # 添加节点的属性信息
            node_props = self.get_node_properties(node)
            if node_props:
                text += "<br><br>属性:"
                for key, value in node_props.items():
                    text += f"<br>- {key}: {value}"
                
            node_text.append(text)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=1, color="#000000")
            ),
            name="节点"
        )
        
        # 创建节点标签轨迹
        node_label_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="text",
            text=[self.get_node_label(node) for node in display_graph.nodes()],
            textposition="top center",
            textfont=dict(size=10),
            hoverinfo="none",
            showlegend=False
        )
        
        # 所有轨迹的组合，先画边，再画节点和标签
        all_traces = edge_traces + edge_label_traces + [node_trace, node_label_trace]
        
        # Create figure
        fig = go.Figure(
            data=all_traces,
            layout=go.Layout(
                title=title,
                showlegend=True,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        )
        
        return fig
    
    def get_node_type(self, node):
        """获取节点的类型"""
        for attr in ["type", "rdf:type", "a"]:
            if attr in self.graph.nodes[node]:
                type_val = self.graph.nodes[node][attr]
                if isinstance(type_val, str) and "/" in type_val:
                    return type_val.split("/")[-1]
                return str(type_val)
        return "未知"
    
    def get_node_properties(self, node):
        """获取节点的所有属性（排除类型和标签）"""
        result = {}
        for key, value in self.graph.nodes[node].items():
            if key not in ["type", "rdf:type", "a", "label", "rdfs:label", "name", "foaf:name"]:
                result[key] = value
        return result
    
    def plot_partition_sizes(self, partitions: Dict) -> go.Figure:
        """
        Create a bar chart of partition sizes
        
        Args:
            partitions: Dictionary mapping partition IDs to sets of vertices
            
        Returns:
            Plotly figure object
        """
        partition_ids = []
        sizes = []
        colors = []
        
        for p_id, nodes in partitions.items():
            partition_ids.append(f"分区 {p_id}")
            sizes.append(len(nodes))
            colors.append(self.partition_colors.get(p_id, "#000000"))
        
        # 使用Figure对象而不是px
        fig = go.Figure()
        
        # 添加分区大小柱状图
        fig.add_trace(go.Bar(
            x=partition_ids,
            y=sizes,
            marker_color=colors,
            text=sizes,
            textposition='auto',
            name='节点数量'
        ))
        
        # 更新布局
        fig.update_layout(
            title="分区大小",
            xaxis_title="分区",
            yaxis_title="节点数量",
            yaxis=dict(
                rangemode='nonnegative',
                zeroline=True
            ),
            bargap=0.3
        )
        
        return fig
    
    def plot_property_cuts(self, property_counts: Dict, top_n: int = 10) -> go.Figure:
        """
        Create a bar chart of most frequently cut properties
        
        Args:
            property_counts: Dictionary mapping property names to cut counts
            top_n: Number of top properties to show
            
        Returns:
            Plotly figure object
        """
        # Sort properties by cut count
        sorted_props = sorted(property_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N
        top_props = sorted_props[:top_n]
        
        # Prepare data
        props = [p[0].split("/")[-1] for p in top_props]  # Extract last part of URI for readability
        counts = [p[1] for p in top_props]
        
        # 使用Figure对象
        fig = go.Figure()
        
        # 添加属性切割柱状图
        fig.add_trace(go.Bar(
            x=props,
            y=counts,
            marker_color='rgba(220, 100, 0, 0.8)',
            text=counts,
            textposition='auto',
            name='切割次数'
        ))
        
        # 更新布局
        fig.update_layout(
            title=f"前 {top_n} 个切割属性",
            xaxis_title="属性",
            yaxis_title="切割次数",
            yaxis=dict(
                rangemode='nonnegative',
                zeroline=True
            ),
            bargap=0.3
        )
        
        return fig
    
    def plot_optimization_history(self, history: Dict) -> go.Figure:
        """
        Plot the optimization history
        
        Args:
            history: Dictionary with optimization history data
            
        Returns:
            Plotly figure object
        """
        iterations = history.get("iterations", [])
        property_cuts = history.get("property_cuts", [])
        balance = history.get("balance", [])
        vertex_moves = history.get("vertex_moves", [])
        
        fig = go.Figure()
        
        # Property cuts line
        fig.add_trace(go.Scatter(
            x=iterations,
            y=property_cuts,
            mode="lines+markers",
            name="属性切割数"
        ))
        
        # Balance ratio line (using secondary y-axis)
        fig.add_trace(go.Scatter(
            x=iterations,
            y=balance,
            mode="lines+markers",
            name="平衡比率",
            yaxis="y2"
        ))
        
        # 如果存在节点移动数据，添加节点移动跟踪
        if vertex_moves and len(vertex_moves) == len(iterations):
            fig.add_trace(go.Scatter(
                x=iterations,
                y=vertex_moves,
                mode="lines+markers",
                name="节点移动数",
                line=dict(dash="dot")
            ))
        
        # Update layout with titles and dual y-axes
        fig.update_layout(
            title="优化历史",
            xaxis_title="迭代次数",
            yaxis_title="指标值",
            yaxis2=dict(
                title="平衡比率",
                overlaying="y",
                side="right"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_iteration_animation(self, history_mappings):
        """
        注意：此方法已被禁用
        
        动画功能已被删除，请使用控制台输出查看迭代详情
        """
        print("动画功能已被禁用，请查看控制台输出的详细迭代过程")
        return None
    
    def get_short_predicate_name(self, predicate):
        """获取谓词的简短名称，用于显示"""
        if isinstance(predicate, str) and '/' in predicate:
            # 移除命名空间前缀，只保留最后部分
            short_name = predicate.split('/')[-1]
            # 如果有#号，取#后面的部分
            if '#' in short_name:
                short_name = short_name.split('#')[-1]
            return short_name
        return str(predicate) 