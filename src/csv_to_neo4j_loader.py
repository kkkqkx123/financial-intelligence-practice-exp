"""
直接从CSV文件加载投资和融资方数据到Neo4j的脚本
"""

import pandas as pd
import numpy as np
from py2neo import Graph
import os
import logging
from typing import List, Dict, Any
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVToNeo4jLoader:
    def __init__(self, uri: str, user: str, password: str):
        """初始化Neo4j连接"""
        self.graph = Graph(uri, auth=(user, password))
        
    def close(self):
        """关闭连接"""
        # py2neo的Graph对象不需要显式关闭
        pass
    
    def load_investment_events(self, csv_path: str):
        """加载投资事件数据"""
        logger.info(f"开始加载投资事件数据: {csv_path}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"成功读取 {len(df)} 条投资事件记录")
            
            # 数据清洗和预处理
            df = self._clean_investment_data(df)
            
            # 创建投资方和融资方节点
            self._create_investor_nodes(df)
            self._create_company_nodes(df)
            
            # 创建投资关系
            self._create_investment_relationships(df)
            
            logger.info("投资事件数据加载完成")
            
        except Exception as e:
            logger.error(f"加载投资事件数据失败: {e}")
            raise
    
    def load_investment_structure(self, csv_path: str):
        """加载投资结构数据"""
        logger.info(f"开始加载投资结构数据: {csv_path}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"成功读取 {len(df)} 条投资结构记录")
            
            # 数据清洗和预处理
            df = self._clean_structure_data(df)
            
            # 创建投资机构详细节点
            self._create_investor_detail_nodes(df)
            
            # 创建行业和轮次关系
            self._create_industry_round_relationships(df)
            
            logger.info("投资结构数据加载完成")
            
        except Exception as e:
            logger.error(f"加载投资结构数据失败: {e}")
            raise
    
    def load_company_data(self, csv_path: str):
        """加载公司详细数据"""
        logger.info(f"开始加载公司数据: {csv_path}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"成功读取 {len(df)} 条公司记录")
            
            # 数据清洗和预处理
            df = self._clean_company_data(df)
            
            # 创建公司详细节点
            self._create_company_detail_nodes(df)
            
            logger.info("公司数据加载完成")
            
        except Exception as e:
            logger.error(f"加载公司数据失败: {e}")
            raise
    
    def _clean_investment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗投资事件数据"""
        # 处理空值
        df = df.replace({np.nan: None})
        
        # 标准化金额格式
        df['金额'] = df['金额'].apply(self._standardize_amount)
        
        # 标准化轮次格式
        df['轮次'] = df['轮次'].apply(self._standardize_round)
        
        return df
    
    def _clean_structure_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗投资结构数据"""
        df = df.replace({np.nan: None})
        
        # 处理行业和轮次字段
        df['行业'] = df['行业'].apply(self._parse_industry)
        df['轮次'] = df['轮次'].apply(self._parse_rounds)
        
        return df
    
    def _clean_company_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗公司数据"""
        df = df.replace({np.nan: None})
        
        # 标准化注册资本格式
        df['注册资本'] = df['注册资本'].apply(self._standardize_capital)
        
        return df
    
    def _standardize_amount(self, amount: str) -> str:
        """标准化金额格式"""
        if pd.isna(amount) or amount == '未披露':
            return None
        
        # 提取数字部分
        match = re.search(r'(\d+(?:\.\d+)?)(?:亿|万)?', str(amount))
        if match:
            num = float(match.group(1))
            if '亿' in str(amount):
                num *= 100000000  # 转换为元
            elif '万' in str(amount):
                num *= 10000  # 转换为元
            return str(int(num))
        
        return None
    
    def _standardize_round(self, round_str: str) -> str:
        """标准化轮次格式"""
        if pd.isna(round_str):
            return None
        
        # 标准化轮次名称
        round_mapping = {
            '种子轮': '种子轮',
            '天使轮': '天使轮', 
            'A轮': 'A轮',
            'B轮': 'B轮',
            'C轮': 'C轮',
            'D轮': 'D轮',
            'E轮': 'E轮',
            '战略融资': '战略融资',
            'Pre-A轮': 'Pre-A轮',
            'Pre-B轮': 'Pre-B轮'
        }
        
        return round_mapping.get(round_str, round_str)
    
    def _standardize_capital(self, capital: str) -> str:
        """标准化注册资本格式"""
        if pd.isna(capital):
            return None
        
        # 提取数字部分
        match = re.search(r'(\d+(?:\.\d+)?)(?:万|亿)?', str(capital))
        if match:
            num = float(match.group(1))
            if '亿' in str(capital):
                num *= 100000000
            elif '万' in str(capital):
                num *= 10000
            return str(int(num))
        
        return None
    
    def _parse_industry(self, industry: str) -> List[str]:
        """解析行业字段"""
        if pd.isna(industry):
            return []
        
        # 提取行业名称
        industries = []
        pattern = r'([^0-9]+?)(\d+家)'
        matches = re.findall(pattern, industry)
        
        for industry_name, _ in matches:
            industries.append(industry_name.strip())
        
        return industries
    
    def _parse_rounds(self, rounds: str) -> List[str]:
        """解析轮次字段"""
        if pd.isna(rounds):
            return []
        
        # 提取轮次名称
        round_list = []
        pattern = r'([A-Za-z\-]+轮)(\d+家)'
        matches = re.findall(pattern, rounds)
        
        for round_name, _ in matches:
            round_list.append(round_name)
        
        return round_list
    
    def _create_investor_nodes(self, df: pd.DataFrame):
        """创建投资方节点"""
        # 提取所有投资方
        all_investors = set()
        for investors in df['投资方']:
            if pd.notna(investors):
                investor_list = str(investors).split()
                all_investors.update(investor_list)
        
        logger.info(f"发现 {len(all_investors)} 个投资方")
        
        # 创建投资方节点
        for investor in all_investors:
            self._create_investor_node(investor)
    
    def _create_investor_node(self, investor_name: str):
        """创建单个投资方节点"""
        query = """
        MERGE (i:投资方 {投资方名称: $name})
        SET i.name = $name,
            i.数据来源 = 'CSV导入',
            i.创建时间 = datetime()
        RETURN i
        """
        self.graph.run(query, name=investor_name)
    
    def _create_company_nodes(self, df: pd.DataFrame):
        """创建融资方（公司）节点"""
        # 提取所有融资方
        all_companies = set()
        for company in df['融资方']:
            if pd.notna(company):
                all_companies.add(str(company))
        
        logger.info(f"发现 {len(all_companies)} 个融资方")
        
        # 创建融资方节点
        for company in all_companies:
            self._create_company_node(company)
    
    def _create_company_node(self, company_name: str):
        """创建单个公司节点"""
        query = """
        MERGE (c:公司 {公司名称: $name})
        SET c.name = $name,
            c.数据来源 = 'CSV导入',
            c.创建时间 = datetime()
        RETURN c
        """
        self.graph.run(query, name=company_name)
    
    def _create_investment_relationships(self, df: pd.DataFrame):
        """创建投资关系"""
        for index, row in df.iterrows():
            if pd.notna(row['投资方']) and pd.notna(row['融资方']):
                investors = str(row['投资方']).split()
                company = row['融资方']
                
                for investor in investors:
                    self._create_investment_relationship(investor, company, row)
    
    def _create_investment_relationship(self, investor: str, company: str, row: pd.Series):
        """创建单个投资关系"""
        query = """
        MATCH (i:投资方 {投资方名称: $investor})
        MATCH (c:公司 {公司名称: $company})
        MERGE (i)-[r:投资]->(c)
        SET r.金额 = $amount,
            r.轮次 = $round,
            r.融资时间 = $investment_date,
            r.事件资讯 = $event_info,
            r.数据来源 = 'CSV导入',
            r.创建时间 = datetime()
        RETURN r
        """
        self.graph.run(query, 
               investor=investor, 
               company=company,
               amount=row['金额'],
               round=row['轮次'],
               investment_date=row['融资时间'],
               event_info=row['事件资讯'])
    
    def _create_investor_detail_nodes(self, df: pd.DataFrame):
        """创建投资机构详细节点"""
        for index, row in df.iterrows():
            if pd.notna(row['机构名称']):
                self._create_investor_detail_node(row)
    
    def _create_investor_detail_node(self, row: pd.Series):
        """创建单个投资机构详细节点"""
        query = """
        MERGE (i:投资方 {投资方名称: $name})
        SET i.投资方简介 = $description,
            i.管理资金规模 = $scale,
            i.数据来源 = 'CSV导入',
            i.创建时间 = datetime()
        RETURN i
        """
        self.graph.run(query, 
               name=row['机构名称'],
               description=row['介绍'],
               scale=row['规模'])
    
    def _create_industry_round_relationships(self, df: pd.DataFrame):
        """创建行业和轮次关系"""
        for index, row in df.iterrows():
            if pd.notna(row['机构名称']):
                # 创建行业关系
                for industry in row['行业']:
                    self._create_industry_relationship(row['机构名称'], industry)
                
                # 创建轮次偏好关系
                for round_name in row['轮次']:
                    self._create_round_preference_relationship(row['机构名称'], round_name)
    
    def _create_industry_relationship(self, investor: str, industry: str):
        """创建行业投资关系"""
        query = """
        MATCH (i:投资方 {投资方名称: $investor})
        MERGE (ind:行业 {行业名称: $industry})
        SET ind.name = $industry,
            ind.数据来源 = 'CSV导入',
            ind.创建时间 = datetime()
        MERGE (i)-[r:投资行业]->(ind)
        SET r.数据来源 = 'CSV导入',
            r.创建时间 = datetime()
        RETURN r
        """
        self.graph.run(query, investor=investor, industry=industry)
    
    def _create_round_preference_relationship(self, investor: str, round_name: str):
        """创建轮次偏好关系"""
        query = """
        MATCH (i:投资方 {投资方名称: $investor})
        MERGE (rnd:轮次 {轮次名称: $round})
        SET rnd.name = $round,
            rnd.数据来源 = 'CSV导入',
            rnd.创建时间 = datetime()
        MERGE (i)-[r:偏好轮次]->(rnd)
        SET r.数据来源 = 'CSV导入',
            r.创建时间 = datetime()
        RETURN r
        """
        self.graph.run(query, investor=investor, round=round_name)
    
    def _create_company_detail_nodes(self, df: pd.DataFrame):
        """创建公司详细节点"""
        for index, row in df.iterrows():
            if pd.notna(row['公司名称']):
                self._create_company_detail_node(row)
    
    def _create_company_detail_node(self, row: pd.Series):
        """创建单个公司详细节点"""
        query = """
        MERGE (c:公司 {公司名称: $name})
        SET c.注册公司 = $registered_company,
            c.公司简介 = $description,
            c.注册地址 = $address,
            c.注册号 = $registration_number,
            c.成立时间 = $establishment_date,
            c.法人代表 = $legal_person,
            c.注册资本 = $registered_capital,
            c.统一社会信用代码 = $credit_code,
            c.官网 = $website,
            c.数据来源 = 'CSV导入',
            c.创建时间 = datetime()
        RETURN c
        """
        self.graph.run(query,
               name=row['公司名称'],
               registered_company=row['注册公司'],
               description=row['介绍'],
               address=row['地址'],
               registration_number=row['注册号'],
               establishment_date=row['成立时间'],
               legal_person=row['法人'],
               registered_capital=row['注册资本'],
               credit_code=row['统一社会信用代码'],
               website=row['官网'])

def main():
    """主函数"""
    # Neo4j连接配置
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "1234567kk"
    
    # CSV文件路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INVESTMENT_EVENTS_CSV = os.path.join(BASE_DIR, "dataset", "investment_events.csv")
    INVESTMENT_STRUCTURE_CSV = os.path.join(BASE_DIR, "dataset", "investment_structure.csv")
    COMPANY_DATA_CSV = os.path.join(BASE_DIR, "dataset", "company_data.csv")
    
    # 创建加载器
    loader = CSVToNeo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # 加载投资事件数据
        loader.load_investment_events(INVESTMENT_EVENTS_CSV)
        
        # 加载投资结构数据
        loader.load_investment_structure(INVESTMENT_STRUCTURE_CSV)
        
        # 加载公司数据（可选）
        # loader.load_company_data(COMPANY_DATA_CSV)
        
        logger.info("所有数据加载完成！")
        
    except Exception as e:
        logger.error(f"数据加载过程中发生错误: {e}")
    finally:
        loader.close()

if __name__ == "__main__":
    main()