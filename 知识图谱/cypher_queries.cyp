/*
 * 企业知识图谱Cypher查询脚本
 * 包含各种常用的图数据查询和分析
 */

/* ========== 基础查询 ========== */

-- 1. 查看平安银行的持股股东
MATCH p=(holder:股东)-[r:参股]->(enterprise:企业)
WHERE enterprise.企业名称 = "平安银行"
RETURN 
    holder.股东名称,
    r.持有数量,
    r.持有比例,
    r.公告日期,
    r.报告期
ORDER BY r.持有比例 DESC;

-- 2. 查看与平安银行相关系数绝对值>0.8的企业实体
MATCH p=(e1:企业)-[r]->(e2:企业)
WHERE (e1.企业名称 = "平安银行" OR e2.企业名称 = "平安银行")
  AND abs(r.相关系数) > 0.8
RETURN 
    e1.企业名称,
    e2.企业名称,
    type(r) as 关系类型,
    r.相关系数
ORDER BY abs(r.相关系数) DESC;

-- 3. 查看所有与平安银行共处一个行业的企业
MATCH p=(e1:企业)-[:行业属于]->(industry:行业)<-[:行业属于]-(e2:企业)
WHERE e1.企业名称 = "平安银行" AND e1 <> e2
RETURN 
    e2.企业名称,
    e2.股票代码,
    industry.行业名称
ORDER BY e2.企业名称;

/* ========== 统计分析 ========== */

-- 4. 各行业企业数量统计
MATCH (e:企业)-[:行业属于]->(industry:行业)
RETURN 
    industry.行业名称,
    count(e) as 企业数量
ORDER BY count(e) DESC
LIMIT 20;

-- 5. 概念热度排行（拥有最多企业的概念）
MATCH (e:企业)-[:概念属于]->(concept:概念)
RETURN 
    concept.概念名称,
    count(e) as 关联企业数
ORDER BY count(e) DESC
LIMIT 20;

-- 6. 主要股东统计（持股企业数量排行）
MATCH (holder:股东)-[r:参股]->(enterprise:企业)
WHERE r.持有比例 >= 5.0
RETURN 
    holder.股东名称,
    count(enterprise) as 持股企业数,
    avg(r.持有比例) as 平均持股比例
ORDER BY count(enterprise) DESC
LIMIT 20;

/* ========== 网络分析 ========== */

-- 7. 企业相关性网络分析（以平安银行为中心）
MATCH path = (center:企业)-[:正相关|负相关*1..2]-(related:企业)
WHERE center.企业名称 = "平安银行"
  AND ALL(rel in relationships(path) WHERE abs(rel.相关系数) >= 0.6)
WITH center, related, 
     min(length(path)) as min_length,
     collect(path) as paths
RETURN 
    center.企业名称 as 中心企业,
    related.企业名称 as 关联企业,
    min_length as 关联层数,
    size(paths) as 路径数量
ORDER BY min_length, related.企业名称;

-- 8. 行业聚类分析（查看银行行业的完整生态）
MATCH (bank:企业)-[:行业属于]->(industry:行业)
WHERE industry.行业名称 = "银行"
OPTIONAL MATCH (bank)-[r:正相关|负相关]-(related:企业)
WHERE abs(r.相关系数) >= 0.7
RETURN 
    bank.企业名称,
    bank.股票代码,
    industry.行业名称,
    collect(related.企业名称) as 高关联企业,
    count(related) as 高关联企业数
ORDER BY count(related) DESC;

/* ========== 风险分析 ========== */

-- 9. 高度相关的企业对（相关系数>0.9）
MATCH (e1:企业)-[r:正相关]->(e2:企业)
WHERE r.相关系数 > 0.9 AND id(e1) < id(e2)
RETURN 
    e1.企业名称 as 企业1,
    e2.企业名称 as 企业2,
    r.相关系数
ORDER BY r.相关系数 DESC
LIMIT 50;

-- 10. 负相关的企业对（相关系数<-0.7）
MATCH (e1:企业)-[r:负相关]->(e2:企业)
WHERE r.相关系数 < -0.7 AND id(e1) < id(e2)
RETURN 
    e1.企业名称 as 企业1,
    e2.企业名称 as 企业2,
    r.相关系数
ORDER BY r.相关系数 ASC
LIMIT 50;

/* ========== 概念分析 ========== */

-- 11. 新能源概念企业完整分析
MATCH (e:企业)-[:概念属于]->(concept:概念)
WHERE concept.概念名称 = "新能源"
OPTIONAL MATCH (e)-[:行业属于]->(industry:行业)
OPTIONAL MATCH (e)-[r:正相关|负相关]-(related:企业)
WHERE abs(r.相关系数) >= 0.6
RETURN 
    e.企业名称,
    e.股票代码,
    industry.行业名称,
    concept.概念名称,
    collect(related.企业名称) as 相关企业,
    count(related) as 相关企业数
ORDER BY count(related) DESC;

-- 12. 跨概念企业分析（同时属于多个热门概念的企业）
MATCH (e:企业)-[:概念属于]->(concept:概念)
WHERE concept.概念名称 IN ["新能源", "锂电池", "光伏", "芯片", "人工智能"]
WITH e, collect(concept.概念名称) as 概念列表
WHERE size(概念列表) >= 2
RETURN 
    e.企业名称,
    e.股票代码,
    概念列表,
    size(概念列表) as 概念数量
ORDER BY size(概念列表) DESC;

/* ========== 股东网络分析 ========== */

-- 13. 共同股东分析（拥有相同主要股东的企业）
MATCH (holder:股东)-[r1:参股]->(e1:企业)
WHERE r1.持有比例 >= 3.0
MATCH (holder)-[r2:参股]->(e2:企业)
WHERE r2.持有比例 >= 3.0 AND e1 <> e2
RETURN 
    holder.股东名称,
    e1.企业名称 as 企业1,
    e2.企业名称 as 企业2,
    r1.持有比例 as 企业1持股比例,
    r2.持有比例 as 企业2持股比例
ORDER BY holder.股东名称, e1.企业名称;

-- 14. 投资集中度分析（查看股东投资行业分布）
MATCH (holder:股东)-[r:参股]->(enterprise:企业)
WHERE r.持有比例 >= 2.0
MATCH (enterprise)-[:行业属于]->(industry:行业)
RETURN 
    holder.股东名称,
    industry.行业名称,
    count(enterprise) as 投资企业数,
    avg(r.持有比例) as 平均持股比例
ORDER BY holder.股东名称, count(enterprise) DESC;

/* ========== 路径分析 ========== */

-- 15. 企业间最短路径分析（示例：平安银行到招商银行）
MATCH path = shortestPath((e1:企业)-[*]-(e2:企业))
WHERE e1.企业名称 = "平安银行" AND e2.企业名称 = "招商银行"
RETURN 
    e1.企业名称 as 起点,
    e2.企业名称 as 终点,
    length(path) as 路径长度,
    [n in nodes(path) | labels(n)[0] + ":" + coalesce(n.企业名称, n.股东名称, n.行业名称, n.概念名称)] as 路径节点,
    [r in relationships(path) | type(r)] as 关系类型;

/* ========== 聚类分析 ========== */

-- 16. 高相关性企业聚类（基于相关性构建企业群组）
MATCH (e:企业)
WHERE e.企业名称 CONTAINS "银行" OR e.企业名称 IN ["中国平安", "招商银行", "工商银行", "建设银行"]
MATCH (e)-[r:正相关|负相关]-(related:企业)
WHERE abs(r.相关系数) >= 0.75
WITH e, collect(related.企业名称) as 高关联企业, count(related) as 关联度
WHERE 关联度 >= 3
RETURN 
    e.企业名称,
    高关联企业,
    关联度
ORDER BY 关联度 DESC;