import os
import json
import networkx as nx
import plotly
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

from mpc_algorithm import MPCAlgorithm
from rdf_loader import RDFLoader
from visualization import MPCVisualizer

app = Flask(__name__)
app.secret_key = "mpc_algorithm_secret_key"

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'ttl', 'rdf', 'owl', 'nt', 'n3', 'jsonld'}
SAMPLE_DATA_PATH = os.path.join('sample_data', 'sample.ttl')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

@app.route('/load_sample', methods=['GET'])
def load_sample():
    """Load the sample data file"""
    try:
        # Load the sample RDF data
        loader = RDFLoader()
        
        loader.load_from_file(SAMPLE_DATA_PATH, format="turtle")
        graph = loader.get_graph()
        
        # Initialize visualizer
        vis = MPCVisualizer(graph)
        vis.compute_layout()
        
        # Store in global state
        current_data['rdf_loader'] = loader
        current_data['graph'] = graph
        current_data['visualizer'] = vis
        
        # Generate initial graph visualization
        graph_fig = vis.plot_graph()
        graph_json = json.dumps(graph_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        flash(f'成功加载示例数据，共 {len(graph.nodes())} 个节点和 {len(graph.edges())} 条边')
        return render_template('graph.html', 
                               graph_json=graph_json,
                               graph_stats={
                                   'nodes': len(graph.nodes()), 
                                   'edges': len(graph.edges())
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
            graph = loader.get_graph()
            
            # Initialize visualizer
            vis = MPCVisualizer(graph)
            vis.compute_layout()
            
            # Store in global state
            current_data['rdf_loader'] = loader
            current_data['graph'] = graph
            current_data['visualizer'] = vis
            
            # Generate initial graph visualization
            graph_fig = vis.plot_graph()
            graph_json = json.dumps(graph_fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            flash(f'成功加载 {filename}，共 {len(graph.nodes())} 个节点和 {len(graph.edges())} 条边')
            return render_template('graph.html', 
                                  graph_json=graph_json,
                                  graph_stats={
                                      'nodes': len(graph.nodes()), 
                                      'edges': len(graph.edges())
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
        
        # 初始化MPC算法（新版本不再使用alpha参数）
        mpc = MPCAlgorithm(current_data['graph'], num_partitions)
        
        # 准备边列表
        edges = []
        for s, t, data in current_data['graph'].edges(data=True):
            edges.append((s, t, data))
        
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
        
        # 处理迭代历史中的状态映射
        if 'history' in results and 'iteration_states' in results['history']:
            iteration_mappings = []
            for state_map in results['history']['iteration_states']:
                # 将内部ID映射回原始节点标识符
                converted_map = {}
                for internal_id, part_id in state_map.items():
                    if internal_id <= len(mpc.id_to_entity) - 1:  # 确保ID在有效范围内
                        original_entity = mpc.id_to_entity[internal_id]
                        converted_map[original_entity] = part_id
                iteration_mappings.append(converted_map)
            
            # 更新历史记录
            results['history']['iteration_states'] = iteration_mappings
        
        # 生成可视化
        vis = current_data['visualizer']
        
        # 绘制分区图
        graph_fig = vis.plot_graph(partition_map=results['vertex_mapping'], 
                                  title="分区RDF图")
        graph_json = json.dumps(graph_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 绘制分区大小
        size_fig = vis.plot_partition_sizes(results['partitions'])
        size_json = json.dumps(size_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 绘制属性切割
        if results['property_cuts']:
            cuts_fig = vis.plot_property_cuts(results['property_cuts'])
            cuts_json = json.dumps(cuts_fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            cuts_json = None
        
        # 绘制优化历史
        if 'history' in results and 'iterations' in results['history'] and len(results['history']['iterations']) > 0:
            history_fig = vis.plot_optimization_history(results['history'])
            history_json = json.dumps(history_fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            history_json = None
        
        # 生成迭代过程动画
        if 'history' in results and 'iteration_states' in results['history'] and len(results['history']['iteration_states']) > 1:
            iteration_animation = vis.create_iteration_animation(results['history']['iteration_states'])
            if iteration_animation:
                iteration_json = json.dumps(iteration_animation, cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False)
            else:
                iteration_json = None
        else:
            iteration_json = None
        
        # 获取指标
        metrics = mpc.get_partition_metrics()
        
        flash('成功运行MPC算法')
        return render_template('results.html',
                              graph_json=graph_json,
                              size_json=size_json,
                              cuts_json=cuts_json,
                              history_json=history_json,
                              iteration_json=iteration_json,
                              metrics=metrics,
                              num_partitions=num_partitions,
                              alpha='自动')
        
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

@app.route('/results')
def results():
    if current_data['results'] is None:
        flash('没有可用结果')
        return redirect('/')
    
    # Generate visualizations
    vis = current_data['visualizer']
    results = current_data['results']
    mpc = current_data['mpc_algorithm']
    
    # Plot graph with partitions
    graph_fig = vis.plot_graph(partition_map=results['vertex_mapping'], 
                              title="分区RDF图")
    graph_json = json.dumps(graph_fig, cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False)
    
    # Debug - print graph data
    print("Graph JSON data length:", len(graph_json))
    
    # Plot partition sizes
    size_fig = vis.plot_partition_sizes(results['partitions'])
    size_json = json.dumps(size_fig, cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False)
    
    # Plot property cuts
    if results['property_cuts']:
        cuts_fig = vis.plot_property_cuts(results['property_cuts'])
        cuts_json = json.dumps(cuts_fig, cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False)
    else:
        cuts_json = None
    
    # Plot optimization history
    if 'history' in results and 'iterations' in results['history'] and len(results['history']['iterations']) > 0:
        history_fig = vis.plot_optimization_history(results['history'])
        history_json = json.dumps(history_fig, cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False)
    else:
        history_json = None
    
    # 生成迭代过程动画
    if 'history' in results and 'iteration_states' in results['history'] and len(results['history']['iteration_states']) > 1:
        iteration_animation = vis.create_iteration_animation(results['history']['iteration_states'])
        if iteration_animation:
            iteration_json = json.dumps(iteration_animation, cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False)
        else:
            iteration_json = None
    else:
        iteration_json = None
    
    # Get metrics
    metrics = mpc.get_partition_metrics()
    
    return render_template('results.html',
                          graph_json=graph_json,
                          size_json=size_json,
                          cuts_json=cuts_json,
                          history_json=history_json,
                          iteration_json=iteration_json,
                          metrics=metrics,
                          num_partitions=mpc.num_partitions,
                          alpha='自动')

if __name__ == '__main__':
    app.run(debug=True) 