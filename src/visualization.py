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
    
    def plot_graph(self, partition_map: Dict = None, title: str = "RDF图可视化") -> go.Figure:
        """
        Create a Plotly figure of the graph
        
        Args:
            partition_map: Dictionary mapping vertex to partition ID
            title: Title for the plot
            
        Returns:
            Plotly figure object
        """
        if self.layout is None:
            self.compute_layout()
        
        # 调试信息
        if partition_map:
            print(f"传入的partition_map长度: {len(partition_map)}")
            print(f"partition_map前5个元素: {list(partition_map.items())[:5]}")
            print(f"self.graph.nodes前5个元素: {list(self.graph.nodes())[:5]}")
            
            # 检查节点与分区映射的匹配情况
            match_count = 0
            for node in self.graph.nodes():
                if node in partition_map:
                    match_count += 1
            print(f"节点在partition_map中的匹配率: {match_count}/{len(self.graph.nodes())}")
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_properties = []
        edge_text = []
        
        for u, v, data in self.graph.edges(data=True):
            x0, y0 = self.layout[u]
            x1, y1 = self.layout[v]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            prop = data.get("property", "unknown")
            edge_properties.append(prop)
            
            # 准备边的悬停文本
            source_label = self.get_node_label(u)
            target_label = self.get_node_label(v)
            edge_text.append(f"关系: {prop}<br>从: {source_label}<br>到: {target_label}")
            edge_text.append(f"关系: {prop}<br>从: {source_label}<br>到: {target_label}")
            edge_text.append(None)  # 为None点添加空文本
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.0, color="#888"),
            hoverinfo="text",
            text=edge_text,
            mode="lines")
            
        # Create node traces
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node in self.graph.nodes():
            x, y = self.layout[node]
            node_x.append(x)
            node_y.append(y)
            
            # Assign color based on partition
            if partition_map and node in partition_map:
                partition = partition_map[node]
                color = self.partition_colors.get(partition, "#000000")
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
                text += f"<br>分区: {partition_map[node]}"
                
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
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
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
        创建分区迭代过程的动画
        
        Args:
            history_mappings: 迭代历史中的顶点映射列表，每个元素是一次迭代的分区映射
            
        Returns:
            Plotly动画图表
        """
        if not history_mappings or self.layout is None:
            return None
            
        frames = []
        num_iterations = len(history_mappings)
        
        # 为每一帧创建更详细的解释文本
        explanations = []
        explanations.append("初始状态：算法开始前的图，节点还未被分配到任何分区")
        
        # 分析每一步的变化情况
        for i in range(1, num_iterations):
            prev_mapping = history_mappings[i-1]
            curr_mapping = history_mappings[i]
            
            # 找出哪些节点改变了分区
            changed_nodes = {}
            for node in curr_mapping:
                if node in prev_mapping:
                    if curr_mapping[node] != prev_mapping[node]:
                        changed_nodes[node] = (prev_mapping[node], curr_mapping[node])
            
            # 统计分区变化
            partition_counts = {}
            for node, (old_part, new_part) in changed_nodes.items():
                if new_part not in partition_counts:
                    partition_counts[new_part] = 0
                partition_counts[new_part] += 1
            
            # 获取主要变化节点（前3个）
            example_nodes = list(changed_nodes.keys())[:3]
            node_labels = [self.get_node_label(node) for node in example_nodes]
            
            # 根据迭代序号生成不同的解释文本
            if i == 1:
                if len(changed_nodes) > 0:
                    explanation = f"第{i}次迭代：算法开始构建初始分区。"
                    if example_nodes:
                        explanation += f" 例如，节点 {', '.join(node_labels)} 被分配到分区。"
                else:
                    explanation = f"第{i}次迭代：算法开始分析谓词的弱连通分量(WCC)。"
            elif i == num_iterations - 1:
                explanation = f"第{i}次迭代：算法完成最终分区分配，共移动 {len(changed_nodes)} 个节点。"
                if example_nodes:
                    explanation += f" 最后的节点分配包括 {', '.join(node_labels)}。"
            else:
                # 根据迭代序号和变化生成更详细的解释
                if len(changed_nodes) > 0:
                    if i <= 3:
                        explanation = f"第{i}次迭代：处理小规模谓词，移动 {len(changed_nodes)} 个节点到相应分区。"
                        if example_nodes:
                            explanation += f" 例如，节点 {', '.join(node_labels)} 被重新分配。"
                    elif i <= 6:
                        explanation = f"第{i}次迭代：处理中等规模谓词，移动 {len(changed_nodes)} 个节点以形成更大的连通分量。"
                        if example_nodes:
                            explanation += f" 节点 {', '.join(node_labels)} 等被合并到同一分区。"
                    else:
                        explanation = f"第{i}次迭代：处理大规模谓词 #{i-6}，移动 {len(changed_nodes)} 个节点。"
                        for part_id, count in partition_counts.items():
                            if part_id >= 0:  # 只显示有效分区ID
                                explanation += f" 分区{part_id}增加{count}个节点。"
                else:
                    explanation = f"第{i}次迭代：分析谓词#{i}的连通性，准备下一步分区。"
            
            explanations.append(explanation)
        
        # 创建第一帧的图
        first_frame = self.plot_graph(
            partition_map=history_mappings[0],
            title="分区迭代过程"
        )
        
        # 创建基础图形
        # 使用与第一帧相同的数据格式创建基础图形
        edge_x = []
        edge_y = []
        
        for u, v in self.graph.edges():
            x0, y0 = self.layout[u]
            x1, y1 = self.layout[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.0, color="#888"),
            hoverinfo="none",
            mode="lines"
        )
        
        # 创建初始节点
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        # 创建一个临时ID到颜色的映射
        temp_id_to_color = {}
        
        for node in self.graph.nodes():
            x, y = self.layout[node]
            node_x.append(x)
            node_y.append(y)
            
            # 第一帧的分区
            partition_map = history_mappings[0]
            if node in partition_map:
                partition = partition_map[node]
                if partition >= 0:
                    # 正常分区ID
                    color = self.partition_colors.get(partition, "#000000")
                else:
                    # 负数ID (临时连通分量或未分配)
                    if partition == -1:
                        color = "#CCCCCC"  # 未分配
                    else:
                        # 为临时ID分配一致的颜色
                        if partition not in temp_id_to_color:
                            color_idx = len(temp_id_to_color) % len(self.temp_colors)
                            temp_id_to_color[partition] = self.temp_colors[color_idx]
                        color = temp_id_to_color[partition]
            else:
                color = "#1f77b4"  # default blue
                
            node_colors.append(color)
            
            # Node size based on degree
            size = 10 + self.graph.degree(node)
            node_sizes.append(size)
            
            # 节点文本
            node_label = self.get_node_label(node)
            node_type = self.get_node_type(node)
            text = f"节点: {node_label}<br>类型: {node_type}<br>连接数: {self.graph.degree(node)}"
            if node in partition_map:
                if partition_map[node] >= 0:
                    text += f"<br>分区: {partition_map[node]}"
                else:
                    if partition_map[node] == -1:
                        text += "<br>分区: 未分配"
                    else:
                        text += f"<br>临时连通分量: {partition_map[node]}"
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
            )
        )
        
        # 创建基础图
        base_fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="分区迭代过程",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=100, l=20, r=20, t=100),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                annotations=[
                    dict(
                        text=explanations[0],
                        x=0.5,
                        y=1.05,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14, color="black"),
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4
                    ),
                    dict(
                        text="<b>MPC算法原理:</b><br>1. 最小化跨分区属性切割<br>2. 保持分区大小平衡<br>3. 迭代优化直至稳定",
                        x=0.01,
                        y=0.01,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=12),
                        bgcolor="rgba(230, 230, 255, 0.9)",
                        bordercolor="blue",
                        borderwidth=1,
                        borderpad=4,
                        align="left"
                    )
                ]
            )
        )
        
        # 为每个迭代创建帧
        for i, mapping in enumerate(history_mappings):
            node_colors = []
            node_text = []
            
            # 为这一帧创建临时ID到颜色的映射
            frame_temp_id_to_color = {}
            
            # 找出当前帧相对于上一帧的变化节点
            highlight_nodes = set()
            if i > 0 and i < num_iterations - 1:  # 只在中间帧高亮节点，最终帧不高亮
                prev_mapping = history_mappings[i-1]
                for node in mapping:
                    if node in prev_mapping:
                        if mapping[node] != prev_mapping[node]:
                            highlight_nodes.add(node)
            
            for node in self.graph.nodes():
                # 更新颜色
                if node in mapping:
                    partition = mapping[node]
                    # 如果是刚刚变化的节点，使用高亮颜色
                    if node in highlight_nodes:
                        color = "red"  # 高亮变化的节点
                    elif partition >= 0:
                        # 正式分区ID
                        color = self.partition_colors.get(partition, "#000000")
                    else:
                        # 负数ID (临时连通分量或未分配)
                        if partition == -1:
                            color = "#CCCCCC"  # 未分配
                        else:
                            # 为临时ID分配一致的颜色
                            if partition not in frame_temp_id_to_color:
                                color_idx = abs(partition) % len(self.temp_colors)
                                frame_temp_id_to_color[partition] = self.temp_colors[color_idx]
                            color = frame_temp_id_to_color[partition]
                else:
                    color = "#1f77b4"  # default blue
                    
                node_colors.append(color)
                
                # 更新文本
                node_label = self.get_node_label(node)
                node_type = self.get_node_type(node)
                text = f"节点: {node_label}<br>类型: {node_type}<br>连接数: {self.graph.degree(node)}"
                if node in mapping:
                    if mapping[node] >= 0:
                        text += f"<br>分区: {mapping[node]}"
                    else:
                        if mapping[node] == -1:
                            text += "<br>分区: 未分配"
                        else:
                            text += f"<br>临时连通分量: {mapping[node]}"
                node_text.append(text)
            
            # 创建新的节点跟踪对象
            updated_node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                hoverinfo="text",
                text=node_text,
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=1, color="#000000")
                )
            )
            
            # 创建迭代帧
            frame = go.Frame(
                data=[edge_trace, updated_node_trace],
                name=f"frame_{i}",
                layout=dict(
                    title=f"迭代 {i}/{num_iterations-1}",
                    annotations=[
                        dict(
                            text=explanations[i],
                            x=0.5,
                            y=1.05,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=14, color="black"),
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4
                        )
                    ]
                )
            )
            frames.append(frame)
        
        # 添加所有帧
        base_fig.frames = frames
        
        # 添加动画控制
        base_fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "播放",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "暂停",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                    }
                ],
                "direction": "left",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [[f"frame_{i}"], {"mode": "immediate"}]
                    }
                    for i in range(num_iterations)
                ],
                "x": 0.1,
                "y": 0,
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "迭代：",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "len": 0.9,
                "pad": {"b": 10, "t": 50}
            }]
        )
        
        return base_fig 