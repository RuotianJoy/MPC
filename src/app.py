import os
import json
import networkx as nx
import plotly
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from mpc_algorithm import MPCAlgorithm
from rdf_loader import RDFLoader
from visualization import MPCVisualizer
from sparql_processor import SPARQLProcessor  # 导入SPARQL处理器

app = Flask(__name__)
app.secret_key = "mpc_algorithm_secret_key"

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'ttl', 'rdf', 'owl', 'nt', 'n3', 'jsonld'}
SAMPLE_DATA_PATH = os.path.join('sample_data', 'sample.ttl')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 创建静态目录
STATIC_SAMPLE_DATA = os.path.join('static', 'sample_data')
if not os.path.exists(STATIC_SAMPLE_DATA):
    os.makedirs(STATIC_SAMPLE_DATA)

# 复制示例查询文件到静态目录
sample_queries = [
    'sample_query.sparql',              # 出生在印度的演员及其电影
    'complex_query.sparql',             # 演员与其共同演员
    'residence_query.sparql',           # 居住在马哈拉施特拉邦的人
    'movie_query.sparql',               # 电影及其制作人
    'actor_relation_query.sparql',      # 演员之间的关系
    'location_statistics_query.sparql', # 地点统计
    'movie_analytics_query.sparql',     # 电影分析
    'complex_path_query.sparql'         # 复杂路径查询
]

for query_file in sample_queries:
    src_path = os.path.join('sample_data', query_file)
    dst_path = os.path.join(STATIC_SAMPLE_DATA, query_file)
    if os.path.exists(src_path) and not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)

# Global state
current_data = {
    'rdf_loader': None,
    'graph': None, 
    'mpc_algorithm': None,
    'results': None,
    'visualizer': None
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/sample_data/<path:filename>')
def serve_sample_data(filename):
    """提供示例数据文件的静态内容"""
    return send_from_directory(STATIC_SAMPLE_DATA, filename)

@app.route('/load_sample', methods=['GET'])
def load_sample():
    """Load the sample data file"""
    try:
        # Load the sample RDF data
        loader = RDFLoader()
        
        loader.load_from_file(SAMPLE_DATA_PATH, format="turtle")
        original_graph = loader.get_graph()
        
        # 初始化可视化器并过滤无意义和类型未知的节点
        vis = MPCVisualizer(original_graph)
        vis.compute_layout()
        
        # 过滤类型未知和无意义的节点
        filtered_graph = vis.filter_meaningless_nodes()
        
        # 使用过滤后的图创建新的可视化器
        filtered_vis = MPCVisualizer(filtered_graph)
        filtered_vis.compute_layout()
        
        # 存储全局状态（使用过滤后的图）
        current_data['rdf_loader'] = loader
        current_data['graph'] = filtered_graph  # 保存过滤后的图
        current_data['original_graph'] = original_graph  # 保存原始图以备需要
        current_data['visualizer'] = filtered_vis
        
        # 生成初始图可视化
        graph_fig = filtered_vis.plot_graph(title="已过滤RDF图（移除了无意义和类型未知的节点）")
        graph_json = json.dumps(graph_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        flash(f'成功加载示例数据，原始图有 {len(original_graph.nodes())} 个节点，过滤后有 {len(filtered_graph.nodes())} 个节点')
        return render_template('graph.html', 
                               graph_json=graph_json,
                               graph_stats={
                                   'original_nodes': len(original_graph.nodes()),
                                   'filtered_nodes': len(filtered_graph.nodes()), 
                                   'edges': len(filtered_graph.edges())
                               })
        
    except Exception as e:
        flash(f'加载示例数据时出错: {str(e)}')
        return redirect('/')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('未提供文件')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('未选择文件')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load the RDF data
        loader = RDFLoader()
        
        # Determine format from file extension
        format = filename.rsplit('.', 1)[1].lower()
        
        try:
            loader.load_from_file(filepath, format=format)
            original_graph = loader.get_graph()
            
            # 初始化可视化器并过滤无意义节点
            vis = MPCVisualizer(original_graph)
            vis.compute_layout()
            
            # 过滤类型未知和无意义的节点
            filtered_graph = vis.filter_meaningless_nodes()
            
            # 使用过滤后的图创建新的可视化器
            filtered_vis = MPCVisualizer(filtered_graph)
            filtered_vis.compute_layout()
            
            # 存储全局状态（使用过滤后的图）
            current_data['rdf_loader'] = loader
            current_data['graph'] = filtered_graph  # 保存过滤后的图
            current_data['original_graph'] = original_graph  # 保存原始图以备需要
            current_data['visualizer'] = filtered_vis
            
            # 生成初始图可视化
            graph_fig = filtered_vis.plot_graph(title="已过滤RDF图（移除了无意义和类型未知的节点）")
            graph_json = json.dumps(graph_fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            flash(f'成功加载 {filename}，原始图有 {len(original_graph.nodes())} 个节点，过滤后有 {len(filtered_graph.nodes())} 个节点')
            return render_template('graph.html', 
                                  graph_json=graph_json,
                                  graph_stats={
                                      'original_nodes': len(original_graph.nodes()),
                                      'filtered_nodes': len(filtered_graph.nodes()), 
                                      'edges': len(filtered_graph.edges())
                                  })
            
        except Exception as e:
            flash(f'加载文件时出错: {str(e)}')
            return redirect(request.url)
    
    flash('无效的文件类型')
    return redirect(request.url)

@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    if current_data['graph'] is None:
        flash('未加载图')
        return redirect('/')
    
    try:
        num_partitions = int(request.form.get('num_partitions', 2))
        filter_meaningless = request.form.get('filter_meaningless', 'false') == 'true'
        epsilon = float(request.form.get('epsilon', '0.1'))  # 获取平衡因子epsilon
        
        # 初始化MPC算法（使用新的epsilon参数）
        mpc = MPCAlgorithm(current_data['graph'], num_partitions, epsilon)
        
        # 准备边列表
        edges = []
        for s, t, data in current_data['graph'].edges(data=True):
            edges.append((s, t, data))
        
        print("\n=========================================================")
        print(f"开始执行MPC算法，节点数: {len(current_data['graph'].nodes())}, 边数: {len(edges)}, 分区数: {num_partitions}, 平衡因子: {epsilon}")
        print("=========================================================")
        
        # 加载图
        mpc.load_graph_from_edges(edges)
        
        # 运行优化
        results = mpc.optimize()
        
        # 存储全局状态
        current_data['mpc_algorithm'] = mpc
        current_data['results'] = results
        
        # 将内部ID映射回原始节点标识符
        vertex_mapping = {}
        for internal_id, partition_id in results['vertex_mapping'].items():
            if internal_id <= len(mpc.id_to_entity) - 1:  # 确保ID在有效范围内
                original_entity = mpc.id_to_entity[internal_id]
                vertex_mapping[original_entity] = partition_id
        
        # 打印调试信息
        print("\n=========================================================")
        print("MPC算法执行完成，最终分区映射:")
        print("=========================================================")
        print(f"原始图节点数: {len(current_data['graph'].nodes())}")
        print(f"映射后顶点数: {len(vertex_mapping)}")
        print(f"图节点示例: {list(current_data['graph'].nodes())[:5]}")
        print(f"映射示例: {list(vertex_mapping.items())[:5]}")
        
        # 更新结果中的映射
        results['vertex_mapping'] = vertex_mapping
        
        # 将分区ID到实体集合的映射也转换为原始实体
        partitions = {}
        for partition_id, internal_ids in results['partitions'].items():
            partitions[partition_id] = set()
            for internal_id in internal_ids:
                if internal_id <= len(mpc.id_to_entity) - 1:  # 确保ID在有效范围内
                    original_entity = mpc.id_to_entity[internal_id]
                    partitions[partition_id].add(original_entity)
        
        # 更新结果中的分区
        results['partitions'] = partitions
        
        # 打印分区详情
        print("\n=========================================================")
        print("最终分区详情:")
        print("=========================================================")
        for partition_id, entities in partitions.items():
            print(f"分区 {partition_id} (共{len(entities)}个节点):")
            for i, entity in enumerate(list(entities)[:20]):
                print(f"  {i+1}. {entity}")
            if len(entities) > 20:
                print(f"  ... 等共{len(entities)}个节点")
        
        # 生成可视化
        vis = current_data['visualizer']
        
        # 绘制分区图
        graph_fig = vis.plot_graph(partition_map=results['vertex_mapping'], 
                                  title="分区RDF图")
        graph_json = json.dumps(graph_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 获取指标
        metrics = mpc.get_partition_metrics()
        print("\n=========================================================")
        print("分区指标:")
        print("=========================================================")
        print(f"总切割属性次数: {metrics['total_cut_properties']}")
        print(f"唯一切割属性数: {metrics['unique_cut_properties']}")
        print(f"分区大小: {metrics['partition_sizes']}")
        print(f"平衡比率: {metrics['balance_ratio']:.2f}")
        
        # 获取算法日志
        logs = results.get('logs', [])
        step_logs = results.get('step_logs', {})
        
        flash('成功运行MPC算法')
        return render_template('results.html',
                              graph_json=graph_json,
                              metrics=metrics,
                              num_partitions=num_partitions,
                              epsilon=epsilon,
                              filter_meaningless=filter_meaningless,
                              logs=logs,
                              step_logs=step_logs)
        
    except Exception as e:
        flash(f'运行算法时出错: {str(e)}')
        return redirect('/')

@app.route('/export', methods=['POST'])
def export_partitions():
    if current_data['results'] is None or current_data['rdf_loader'] is None:
        flash('没有可导出的结果')
        return redirect('/')
    
    try:
        export_dir = request.form.get('export_dir', 'exports')
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        base_filename = request.form.get('filename', 'partition')
        format = request.form.get('format', 'turtle')
        
        # Export partitions to RDF files
        current_data['rdf_loader'].export_partitions_to_rdf(
            current_data['results']['partitions'],
            os.path.join(export_dir, base_filename),
            format
        )
        
        flash(f'成功导出分区到 {export_dir}')
        return redirect(url_for('results'))
        
    except Exception as e:
        flash(f'导出分区时出错: {str(e)}')
        return redirect('/')

@app.route('/process_sparql', methods=['POST'])
def process_sparql():
    """
    处理SPARQL查询，将其分解为子查询
    """
    if current_data['results'] is None or 'vertex_mapping' not in current_data['results']:
        flash('请先运行MPC算法')
        return redirect('/results')
    
    try:
        # 获取SPARQL查询
        sparql_query = request.form.get('sparql_query', '')
        
        if not sparql_query:
            flash('请输入SPARQL查询')
            return redirect('/results')
        
        # 获取分区映射
        vertex_mapping = current_data['results']['vertex_mapping']
        
        # 初始化SPARQL处理器
        processor = SPARQLProcessor(vertex_mapping)
        
        # 获取RDF加载器和图数据
        rdf_graph = None
        if current_data['rdf_loader']:
            # 使用RDFLib的Graph对象而不是NetworkX图
            rdf_graph = current_data['rdf_loader'].get_rdf_graph()
            
        # 执行原始查询
        original_results = processor.execute_original_query(sparql_query, rdf_graph=rdf_graph)
        
        # 分解查询
        processor.decompose_query(sparql_query)
        
        # 优化执行计划
        execution_plan = processor.optimize_execution()
        
        # 为子查询生成SPARQL
        subqueries_sparql = processor.generate_all_subqueries()
            
        # 执行子查询
        subquery_results = processor.execute_subqueries(rdf_graph=rdf_graph)
        
        # 存储处理结果
        current_data['sparql_processor'] = processor
        current_data['sparql_results'] = {
            'original_query': sparql_query,
            'original_results': original_results,
            'subqueries': processor.subqueries,
            'execution_plan': execution_plan,
            'subqueries_sparql': subqueries_sparql,
            'subquery_results': subquery_results,
            'logs': processor.log_buffer,
            'step_logs': processor.step_logs
        }
        
        # 重定向到结果页面
        flash('成功处理SPARQL查询')
        return redirect('/results')
        
    except Exception as e:
        flash(f'处理SPARQL查询时出错: {str(e)}')
        return redirect('/results')

@app.route('/results')
def results():
    """Display the results of the MPC algorithm."""
    if current_data['results'] is None:
        flash('请先运行算法')
        return redirect('/')
    
    # 获取MPC算法结果
    results = current_data['results']
    visualizer = current_data['visualizer']
    
    # 如果存在图，则绘制分区图
    graph_fig = visualizer.plot_graph(partition_map=results['vertex_mapping'], 
                                    title="分区RDF图")
    graph_json = json.dumps(graph_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 获取指标
    metrics = current_data['mpc_algorithm'].get_partition_metrics()
    
    # 获取算法日志
    logs = results.get('logs', [])
    step_logs = results.get('step_logs', {})
    
    # 检查是否有SPARQL处理结果
    sparql_results = current_data.get('sparql_results', None)
    
    return render_template('results.html',
                         graph_json=graph_json,
                         metrics=metrics,
                         logs=logs,
                         step_logs=step_logs,
                         num_partitions=current_data['mpc_algorithm'].num_partitions,
                         epsilon=current_data['mpc_algorithm'].epsilon,
                         sparql_results=sparql_results)

@app.route('/visualization', methods=['GET'])
def visualize_results():
    """根据请求参数可视化图"""
    if current_data['graph'] is None or current_data['visualizer'] is None:
        flash('未加载图或运行算法')
        return redirect('/')
    
    try:
        filter_nodes = request.args.get('filter', 'false') == 'true'
        show_edge_labels = request.args.get('edge_labels', 'true') == 'true'
        
        # 准备分区映射
        if current_data['results'] and 'vertex_mapping' in current_data['results']:
            partition_map = current_data['results']['vertex_mapping']
        else:
            partition_map = None
        
        # 重新绘制图，应用过滤
        vis = current_data['visualizer']
        graph_fig = vis.plot_graph(
            partition_map=partition_map, 
            filter_nodes=filter_nodes,
            show_edge_labels=show_edge_labels
        )
        graph_json = json.dumps(graph_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('graph_view.html', 
                               graph_json=graph_json,
                               filter_applied=filter_nodes,
                               show_edge_labels=show_edge_labels)
    
    except Exception as e:
        flash(f'可视化时出错: {str(e)}')
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True) 