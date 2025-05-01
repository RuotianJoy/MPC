# MPC-RDF 图分区与SPARQL查询处理-MPC论文复现

这是一个基于MPC(Minimum Property-Cut)算法的RDF图分区与SPARQL查询处理系统，不仅实现了高效的RDF图分区，还提供了完整的SPARQL查询解析、分解和执行功能。系统通过可视化界面直观展示图分区结果与查询处理过程，适用于分布式RDF数据管理和SPARQL查询优化研究。

## 目录

- [功能特点](#功能特点)
- [系统架构](#系统架构)
- [安装与运行](#安装与运行)
- [使用指南](#使用指南)
- [SPARQL查询示例](#SPARQL查询示例)
- [算法原理](#算法原理)
- [开发者文档](#开发者文档)

## 功能特点

### RDF图分区功能
- **MPC分区算法**：最小化跨分区谓词（属性）的数量，优化分布式查询处理
- **分区平衡控制**：通过epsilon参数调整分区平衡度
- **图过滤与处理**：自动过滤无意义节点，提高分区质量
- **分区指标统计**：计算切割属性数量、分区平衡比率等关键指标
- **分区结果导出**：支持将分区结果导出为多种RDF格式

### SPARQL查询处理功能
- **查询解析与验证**：支持标准SPARQL 1.1语法，正确处理前缀、变量和URI
- **查询分解算法**：基于图分区结果自动将SPARQL查询分解为子查询
- **查询执行与结果展示**：支持执行原始查询和子查询，以表格形式展示结果
- **查询优化**：为子查询生成最优执行计划，确定最佳执行分区
- **详细日志**：记录查询处理的每个步骤，方便理解和调试

### 可视化界面
- **交互式图可视化**：使用Plotly实现的交互式图显示，支持缩放、拖拽和悬停查看
- **分区结果展示**：以不同颜色直观展示分区结果
- **查询过程跟踪**：详细展示查询分解和执行的每个步骤
- **结果比较**：并排展示原始查询和子查询的结果，易于比较和分析

## 系统架构

系统由以下几个主要组件构成：

1. **RDFLoader**：负责加载和处理RDF数据，支持多种格式（Turtle, N-Triples, RDF/XML等）
2. **MPCAlgorithm**：实现MPC图分区算法，将RDF图分为多个平衡的分区
3. **SPARQLProcessor**：处理SPARQL查询，包括解析、分解、优化和执行
4. **MPCVisualizer**：负责可视化图和分区结果
5. **Web应用**：基于Flask的Web界面，集成所有功能

## 安装与运行

### 系统要求
- Python 3.7+
- 依赖包见`requirements.txt`

### 安装步骤

1. 克隆代码库
   ```bash
   git clone https://github.com/your-username/mpc-rdf.git
   cd mpc-rdf
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 运行Web应用
   ```bash
   python -m src.app
   ```

4. 在浏览器中访问
   ```
   http://localhost:5000
   ```

## 使用指南

### 加载RDF数据

1. 在首页选择"加载示例数据"或上传自己的RDF文件（支持.ttl, .nt, .rdf等格式）
2. 系统会自动处理数据并显示图可视化结果
3. 图中会过滤无意义的节点，提高可读性

### 运行图分区

1. 在图显示页面，设置分区数量（默认为2）和平衡因子epsilon（默认为0.1）
2. 点击"运行MPC算法"按钮
3. 系统会执行分区算法并显示结果，不同颜色代表不同分区
4. 查看分区指标，包括切割属性数量和分区平衡度

### 处理SPARQL查询

1. 在分区结果页面，切换到"SPARQL查询处理"选项卡
2. 输入SPARQL查询，或从下拉菜单选择预设的示例查询
3. 点击"处理查询"按钮
4. 系统会解析查询，显示原始查询结果和分解后的子查询结果
5. 查看详细的处理日志，了解查询分解和执行的过程

### 导出分区结果

1. 在分区结果页面，设置输出格式和目录
2. 点击"导出分区"按钮
3. 系统会将每个分区导出为单独的RDF文件

## SPARQL查询示例

系统预置了多种示例查询，从简单到复杂，展示不同的SPARQL功能：

### 基础查询

1. **出生在印度的演员及其电影**
   ```sparql
   PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX foaf: <http://xmlns.com/foaf/0.1/>
   PREFIX dbo: <http://dbpedia.org/ontology/>
   PREFIX ex: <http://example.org/>

   SELECT ?person ?name ?movie ?movieName
   WHERE {
     ?person a foaf:Person .
     ?person rdfs:label ?name .
     ?person dbo:birthPlace ex:India .
     ?movie dbo:starring ?person .
     ?movie rdfs:label ?movieName .
   }
   ```

2. **电影及其制作人**
   ```sparql
   PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX foaf: <http://xmlns.com/foaf/0.1/>
   PREFIX dbo: <http://dbpedia.org/ontology/>

   SELECT ?movie ?movieName ?producer ?producerName
   WHERE {
     ?movie a dbo:Film .
     ?movie rdfs:label ?movieName .
     ?movie dbo:producer ?producer .
     ?producer a foaf:Person .
     ?producer rdfs:label ?producerName .
   }
   ```

### 高级查询

1. **演员关系网络**
   ```sparql
   PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX foaf: <http://xmlns.com/foaf/0.1/>
   PREFIX dbo: <http://dbpedia.org/ontology/>
   PREFIX ex: <http://example.org/>

   SELECT ?actor1 ?actor1Name ?actor2 ?actor2Name ?relationship ?movieName
   WHERE {
     ?actor1 a foaf:Person .
     ?actor2 a foaf:Person .
     ?actor1 rdfs:label ?actor1Name .
     ?actor2 rdfs:label ?actor2Name .
     FILTER(?actor1 != ?actor2)
     
     {
       ?movie dbo:starring ?actor1 .
       ?movie dbo:starring ?actor2 .
       ?movie rdfs:label ?movieName .
       BIND("共同出演电影" AS ?relationship)
     } 
     UNION 
     {
       ?actor1 dbo:spouse ?actor2 .
       BIND("配偶" AS ?relationship)
       BIND("" AS ?movieName)
     }
   }
   ```

2. **地点统计分析**
   ```sparql
   PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX foaf: <http://xmlns.com/foaf/0.1/>
   PREFIX dbo: <http://dbpedia.org/ontology/>

   SELECT ?location ?locationName (COUNT(DISTINCT ?person) AS ?personCount)
          (GROUP_CONCAT(DISTINCT ?name; separator=", ") AS ?personNames)
   WHERE {
     ?location a ?locationType .
     ?location rdfs:label ?locationName .
     
     {
       ?person dbo:birthPlace ?location .
       ?person rdfs:label ?name .
     }
     UNION
     {
       ?person dbo:residence ?location .
       ?person rdfs:label ?name .
     }
     
     ?person a foaf:Person .
     FILTER(?locationType IN (dbo:Place, dbo:Country))
   }
   GROUP BY ?location ?locationName
   ORDER BY DESC(?personCount) ?locationName
   ```

3. **电影详细分析**
   ```sparql
   PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX foaf: <http://xmlns.com/foaf/0.1/>
   PREFIX dbo: <http://dbpedia.org/ontology/>

   SELECT ?movie ?movieName ?actorCount
          ?producerName ?producerResidence
          (GROUP_CONCAT(DISTINCT ?actorName; separator=", ") AS ?actors)
   WHERE {
     ?movie a dbo:Film .
     ?movie rdfs:label ?movieName .
     
     {
       SELECT ?movie (COUNT(DISTINCT ?actor) AS ?actorCount)
       WHERE {
         ?movie dbo:starring ?actor .
         ?actor a foaf:Person .
       }
       GROUP BY ?movie
     }
     
     OPTIONAL {
       ?movie dbo:starring ?actor .
       ?actor rdfs:label ?actorName .
     }
     
     OPTIONAL {
       ?movie dbo:producer ?producer .
       ?producer rdfs:label ?producerName .
       
       OPTIONAL {
         ?producer dbo:residence ?residence .
         ?residence rdfs:label ?producerResidence .
       }
     }
   }
   GROUP BY ?movie ?movieName ?actorCount ?producerName ?producerResidence
   ```

## 算法原理

### MPC图分区算法

MPC (Minimum Property-Cut) 算法的核心思想是最小化跨分区的谓词（属性）数量，主要步骤如下：

1. **初始化**：读取RDF数据图G，将边属性保存到集合L中
2. **计算WCC**：遍历属性集合L，计算每个属性p对应的弱连通组件分量WCC(G(p))以及对应的代价Cost({p})
3. **选择内部属性**：尽可能从属性集合L中选择更多的内部属性Lin
4. **构建粗化图**：使用内部属性构建粗化图，将相关节点合并为超点
5. **分区超点**：对粗化图中的超点使用自顶向下分区算法
6. **映射回原图**：将超点集合映射回原始数据图，得到最终分区结果

详细的算法流程和数学证明请参考[原始论文](https://ieeexplore.ieee.org/document/9835293)。

### SPARQL查询分解算法

系统实现的SPARQL查询分解算法基于以下步骤：

1. **初始化**：初始化一个空集合Q，用于存储分解后的子查询
2. **划分边属性**：将SPARQL查询中的边划分为变量属性或跨越属性的边，得到一组弱连通分量
3. **处理跨分区边**：遍历SPARQL查询中变量属性或跨越属性的边
4. **子查询合并决策**：根据顶点数量比较，决定将边添加到哪个子查询
5. **构建子查询**：将顶点数量大于1的子查询加入结果集Q
6. **生成执行计划**：为每个子查询生成最优的执行计划

## 开发者文档

### 项目结构

```
src/
├── app.py                # Flask应用主入口
├── mpc_algorithm.py      # MPC算法实现
├── rdf_loader.py         # RDF数据加载与处理
├── sparql_processor.py   # SPARQL查询处理器
├── visualization.py      # 可视化组件
├── static/               # 静态资源
├── templates/            # HTML模板
└── sample_data/          # 示例数据和查询
```

### 扩展指南

1. **添加新的分区算法**：
   - 创建一个新的类，实现与MPCAlgorithm相同的接口
   - 在app.py中添加算法选择选项

2. **添加新的查询功能**：
   - 在SPARQLProcessor类中添加新的方法
   - 更新templates/results.html以显示新功能的结果

3. **添加新的可视化**：
   - 在MPCVisualizer类中添加新的可视化方法
   - 更新相应的HTML模板以显示新的可视化结果
