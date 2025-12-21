# 知识图谱与大模型应用

本项目旨在通过大模型技术，自动化生成针对特定金融领域的知识图谱。
需要从金融文本中抽取实体和关系，基于大模型实现知识图谱的构建，并且将知识图谱存储在Neo4j数据库中。

数据集为src\dataset\company_data.csv
src\dataset\investment_events.csv
src\dataset\investment_structure.csv
这些文件我屏蔽了。如果需要查看，使用命令或直接参考src\dataset\company_data.md
src\dataset\investment_events.md
src\dataset\investment_structure.md

**注意**
代码必须使用csv文件，而非md演示文件。csv文件都已存在，你可以使用终端命令确认。但使用工具查看时这些csv文件是不可见的。
这是为了减少ide索引的开销，实际文件是存在的
公司数据集不需要处理，该数据集与主工作流无关

**NEO4J**
NEO4J位于docker，将7473、7474端口映射到
数据库名称为neo4j
用户名为neo4j
密码为1234567kk