import sys
from rdflib import Graph

if len(sys.argv) != 3:
    print("Usage: python convert_ttl_to_nt.py input.ttl output.nt")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# 创建一个空图
g = Graph()

# 从TTL文件加载RDF数据
g.parse(input_file, format="turtle")

# 以NT格式保存
g.serialize(destination=output_file, format="nt")

print(f"转换完成: {input_file} -> {output_file}") 