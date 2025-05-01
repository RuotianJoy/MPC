# MPC-RDF 图分区算法

MPC (Minimum Property-Cut) 是一种专为RDF图设计的分区算法，通过最小化跨分区谓词（属性）的数量来实现高效的分布式查询处理。与传统的图分区算法不同，MPC算法专注于减少跨分区边的种类（谓词/属性），而不仅仅是边的数量。

## 目录

- [原理介绍](#原理介绍)
- [算法流程](#算法流程)
- [安装和运行](#安装和运行)
- [示例说明](#示例说明)
- [可视化界面](#可视化界面)
- [算法实现细节](#算法实现细节)
- [性能分析](#性能分析)

## 原理介绍

### RDF图分区的挑战

RDF（资源描述框架）数据通常表示为主语-谓词-宾语的三元组集合，形成一个具有丰富语义信息的图结构。在大规模RDF数据管理中，将数据分区存储在多个节点上可以提高查询处理效率。然而，传统的图分区算法通常只关注减少跨分区边的数量，而忽略了RDF图中谓词（属性）的重要性。

### MPC算法核心思想

MPC算法的核心思想是：

1. **最小化跨分区谓词数量**：不同于传统分区算法关注的边数量，MPC专注于减少跨分区的谓词类型数量，因为在SPARQL查询中，谓词通常是重要的连接点。

2. **保持分区大小平衡**：确保每个分区的节点数量相对均衡，避免出现过大或过小的分区。

3. **利用弱连通分量(WCC)分析**：通过分析每个谓词形成的弱连通分量，识别应该放在同一分区的节点组。

## 算法流程

MPC算法的执行流程分为以下几个关键步骤：

1. **图分析与预处理**
   - 加载RDF图数据
   - 为每个实体和谓词分配唯一ID
   - 统计每个谓词的出现频率和连通性

2. **谓词排序与分类**
   - 根据弱连通分量(WCC)的数量对谓词进行排序
   - 将谓词分为小规模、中规模和大规模三类进行处理

3. **图粗化(Coarsening)**
   - 对每个谓词进行弱连通分量分析
   - 标记超过限制大小的连通分量为无效

4. **贪心合并处理**
   - 首先处理小规模谓词，将其连接的节点合并为连通分量
   - 然后按WCC数量排序处理剩余谓词，不断合并节点形成更大的连通分量
   - 记录每一步迭代的分区状态，用于可视化和分析

5. **分区分配**
   - 将合并后的连通分量分配到最终分区
   - 使用贪心算法确保分区大小平衡
   - 计算最终的分区指标，包括切割谓词数量和分区平衡度

## 安装和运行

### 系统要求

- Python 3.7+
- 依赖包见 `requirements.txt`

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

## 示例说明

我们以一个简单的社交网络RDF数据为例，演示MPC算法的工作过程。

### 示例数据 (sample.ttl)

下面是一个简化的社交网络RDF数据：

```turtle
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix dc: <http://purl.org/dc/elements/1.1/> .

# 人物
ex:Person1 a foaf:Person ;
    rdfs:label "张三" ;
    foaf:age "28" ;
    foaf:knows ex:Person2, ex:Person3 ;
    ex:livesIn ex:City1 ;
    ex:worksAt ex:Company1 .

ex:Person2 a foaf:Person ;
    rdfs:label "李四" ;
    foaf:age "32" ;
    foaf:knows ex:Person1, ex:Person3 ;
    ex:livesIn ex:City1 ;
    ex:worksAt ex:Company2 .

ex:Person3 a foaf:Person ;
    rdfs:label "王五" ;
    foaf:age "45" ;
    foaf:knows ex:Person1, ex:Person2 ;
    ex:livesIn ex:City2 ;
    ex:worksAt ex:Company1 .

# 城市和公司
ex:City1 a ex:City ;
    rdfs:label "北京" .

ex:City2 a ex:City ;
    rdfs:label "上海" .

ex:Company1 a ex:Company ;
    rdfs:label "科技有限公司" ;
    ex:locatedIn ex:City1 .

ex:Company2 a ex:Company ;
    rdfs:label "金融服务公司" ;
    ex:locatedIn ex:City2 .
```

### 算法执行过程说明

1. **初始状态**: 所有节点未分配到任何分区（标记为-1）
   
2. **第1次迭代**: 处理小规模谓词 `rdf:type` 和 `rdfs:label`，将相同类型的实体分到临时连通分量中

3. **第2次迭代**: 分析谓词的弱连通分量(WCC)，为后续处理做准备
   
4. **第3-4次迭代**: 处理 `foaf:knows` 谓词，合并社交关系紧密的人物节点
   
5. **第5-6次迭代**: 处理 `ex:livesIn` 和 `ex:worksAt` 谓词，考虑地理位置和工作关系
   
6. **最终分区**: 将临时连通分量映射到最终分区
   - 分区0: 主要包含张三、李四及北京相关实体
   - 分区1: 主要包含王五及上海相关实体

### 最终结果分析

- **分区大小**: 分区0包含5个节点，分区1包含4个节点，分区大小较为平衡
- **切割谓词**: 只有 `foaf:knows` 和 `ex:worksAt` 成为跨分区谓词
- **查询性能**: 针对地理位置或公司的查询大多可以在单个分区内完成，提高了查询效率

## 可视化界面

MPC-RDF系统提供了直观的Web可视化界面：

1. **图可视化**: 通过交互式图显示RDF数据的节点和边，不同颜色代表不同分区

2. **迭代过程动画**: 展示算法每一步的执行过程，高亮显示变化的节点

3. **分区统计**: 显示分区大小和平衡度的统计图表

4. **切割谓词分析**: 展示哪些谓词被切割以及切割频率

![分区可视化示例](https://example.org/partition_visualization.png)

## 算法实现细节

### Greed3策略

当前实现主要采用Greed3算法策略，基于以下步骤：

1. 首先处理小规模谓词（边数少于阈值的谓词）
2. 对中大型谓词按WCC数量排序
3. 从WCC数量最多的谓词开始处理（逆序），这样可以先处理更容易分割的谓词

### 并查集优化

算法使用了优化的并查集数据结构来跟踪连通分量：

- **路径压缩**: 加速查找操作
- **按秩合并**: 保持并查集树的平衡
- **增量处理**: 实时跟踪连通分量大小，确保不超过限制

## 性能分析

### 时间复杂度

- 图加载与预处理: O(|E|)，其中|E|是边数
- 谓词排序: O(P log P)，其中P是谓词数量
- 图粗化与合并: O(|E| * α(|V|))，其中α是阿克曼函数的反函数，近似常数
- 总体复杂度: O(|E| * α(|V|) + P log P)

### 空间复杂度

- 图存储: O(|V| + |E|)
- 分区映射: O(|V|)
- 连通分量跟踪: O(|V|)
- 总体空间复杂度: O(|V| + |E|)

### 适用场景

MPC算法特别适合以下场景：

- 具有丰富语义信息的知识图谱
- 查询模式中谓词种类是重要考虑因素的应用
- 需要在分布式环境中高效处理SPARQL查询的系统