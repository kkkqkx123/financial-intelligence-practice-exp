基于 financial_ie_scheme.py 新建csv处理目录、实体抽取目录，分别包含基类和具体实现，分别处理src\dataset\investment_structure.csv

src\dataset\investment_events.csv

src\dataset\company_data.csv(由于是数据集，内容对你隐藏了)

对应的说明文档为src\dataset\company_data.md

src\dataset\investment_events.md

src\dataset\investment_structure.md

输出到src\extraction_results\entities目录。

每个数据集的处理结果命名为`数据集名称_entities.csv`，列名为entity name,entity_type,entity_name