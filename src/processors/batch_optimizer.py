"""
批处理优化器
集中处理需要LLM增强的地方，减少调用次数
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from datetime import datetime

from .llm_client import get_llm_client, get_enhancement_tracker
from .config import CONFIDENCE_THRESHOLDS, BATCH_PROCESSING_CONFIG

logger = logging.getLogger(__name__)


class BatchOptimizer:
    """批处理优化器"""
    
    def __init__(self):
        self.llm_client = get_llm_client()
        self.enhancement_tracker = get_enhancement_tracker()
        self.config = BATCH_PROCESSING_CONFIG
        self.optimization_stats: Dict[str, Any] = {
            'entities_processed': 0,
            'relationships_processed': 0,
            'llm_calls_made': 0,
            'batch_sizes': defaultdict(int),
            'processing_time': 0.0
        }
    
    def optimize_entity_descriptions(self, entities: Dict[str, Dict], entity_type: str) -> Dict[str, Dict]:
        """优化实体描述"""
        logger.info(f"开始优化{entity_type}实体描述，共 {len(entities)} 个实体")
        
        enhanced_entities = {}
        enhancement_requests = []
        
        for entity_id, entity_data in entities.items():
            # 检查是否需要增强
            if self._needs_description_enhancement(entity_data):
                request_data = {
                    'entity_name': entity_data.get('name', ''),
                    'context': {
                        'industry': entity_data.get('industry', ''),
                        'establish_date': entity_data.get('establish_date', ''),
                        'description': entity_data.get('description', ''),
                        'entity_type': entity_type
                    }
                }
                
                # 添加到跟踪器
                request_id = self.enhancement_tracker.add_enhancement_request(
                    'enhance_description',
                    request_data,
                    priority=self._get_enhancement_priority(entity_data)
                )
                
                enhancement_requests.append({
                    'entity_id': entity_id,
                    'entity_data': entity_data,
                    'request_id': request_id
                })
            else:
                # 不需要增强，直接保留
                enhanced_entities[entity_id] = entity_data
        
        logger.info(f"需要LLM增强的实体描述: {len(enhancement_requests)} 个")
        
        # 批量处理
        if enhancement_requests:
            batch_results = self._process_description_enhancements(enhancement_requests)
            
            # 应用增强结果
            for result in batch_results:
                entity_id = result['entity_id']
                enhanced_description = result['enhanced_description']
                
                # 更新实体数据
                enhanced_entity = entities[entity_id].copy()
                enhanced_entity['description'] = enhanced_description
                enhanced_entity['metadata']['llm_enhanced'] = True
                enhanced_entity['metadata']['enhancement_type'] = 'description'
                enhanced_entity['metadata']['enhancement_confidence'] = result.get('confidence', 0.8)
                
                enhanced_entities[entity_id] = enhanced_entity
        
        self.optimization_stats['entities_processed'] += len(enhancement_requests)
        logger.info(f"实体描述优化完成: {len(enhancement_requests)} 个增强")
        
        return enhanced_entities
    
    def optimize_entity_conflicts(self, conflicts: List[Dict]) -> Dict[str, Dict]:
        """优化实体冲突解决"""
        logger.info(f"开始优化实体冲突解决，共 {len(conflicts)} 个冲突")
        
        resolved_conflicts = {}
        conflict_groups = self._group_conflicts_by_similarity(conflicts)
        
        for group_id, conflict_group in conflict_groups.items():
            if len(conflict_group) > 1:
                # 创建冲突解决请求
                request_data = {
                    'conflicting_entities': conflict_group
                }
                
                request_id = self.enhancement_tracker.add_enhancement_request(
                    'resolve_conflict',
                    request_data,
                    priority='high'  # 冲突解决优先级高
                )
                
                # 批量处理冲突
                resolution = self._process_conflict_resolution(conflict_group)
                
                # 应用解决结果
                for conflict in conflict_group:
                    entity_id = conflict['id']
                    resolved_conflicts[entity_id] = resolution
        
        self.optimization_stats['entities_processed'] += len(conflicts)
        logger.info(f"实体冲突优化完成: {len(conflicts)} 个冲突处理")
        
        return resolved_conflicts
    
    def optimize_relationship_extraction(self, text_sources: List[Dict]) -> List[Dict]:
        """优化关系提取"""
        logger.info(f"开始优化关系提取，共 {len(text_sources)} 个文本源")
        
        extracted_relationships = []
        
        # 按文本长度和重要性分组
        text_groups = self._group_texts_for_processing(text_sources)
        
        for group in text_groups:
            # 合并相关文本以提高效率
            combined_text = self._combine_related_texts(group)
            
            if combined_text:
                # 创建关系提取请求
                request_data = {
                    'text': combined_text
                }
                
                request_id = self.enhancement_tracker.add_enhancement_request(
                    'extract_relationships',
                    request_data,
                    priority='medium'
                )
        
        # 批量处理所有关系提取请求
        batch_results = self._process_relationship_extraction()
        
        # 整理结果
        for result in batch_results:
            if result.get('relationships'):
                extracted_relationships.extend(result['relationships'])
        
        self.optimization_stats['relationships_processed'] += len(extracted_relationships)
        logger.info(f"关系提取优化完成: {len(extracted_relationships)} 个关系")
        
        return extracted_relationships
    
    def optimize_industry_classification(self, companies: Dict[str, Dict]) -> Dict[str, List[str]]:
        """优化行业分类"""
        logger.info(f"开始优化行业分类，共 {len(companies)} 个公司")
        
        industry_classifications = {}
        classification_requests = []
        
        for company_id, company_data in companies.items():
            # 检查是否需要分类
            if self._needs_industry_classification(company_data):
                request_data = {
                    'company_info': company_data
                }
                
                request_id = self.enhancement_tracker.add_enhancement_request(
                    'classify_industry',
                    request_data,
                    priority=self._get_classification_priority(company_data)
                )
                
                classification_requests.append({
                    'company_id': company_id,
                    'company_data': company_data,
                    'request_id': request_id
                })
            else:
                # 使用现有分类
                industry_classifications[company_id] = company_data.get('industry', [])
        
        # 批量处理分类请求
        if classification_requests:
            batch_results = self._process_industry_classification(classification_requests)
            
            # 应用分类结果
            for result in batch_results:
                company_id = result['company_id']
                industries = result['industries']
                
                industry_classifications[company_id] = industries
        
        self.optimization_stats['entities_processed'] += len(classification_requests)
        logger.info(f"行业分类优化完成: {len(classification_requests)} 个公司分类")
        
        return industry_classifications
    
    def optimize_investor_name_standardization(self, investor_names: Set[str]) -> Dict[str, str]:
        """优化投资方名称标准化"""
        logger.info(f"开始优化投资方名称标准化，共 {len(investor_names)} 个名称")
        
        standardized_names = {}
        standardization_requests = []
        
        # 分组相似名称
        name_groups = self._group_similar_investor_names(investor_names)
        
        for group in name_groups:
            if len(group) > 1:
                # 需要标准化的名称组
                for name in group:
                    request_data = {
                        'investor_name': name,
                        'context': {
                            'similar_names': [n for n in group if n != name],
                            'group_size': len(group)
                        }
                    }
                    
                    request_id = self.enhancement_tracker.add_enhancement_request(
                        'standardize_name',
                        request_data,
                        priority='high' if len(group) > 3 else 'medium'
                    )
                    
                    standardization_requests.append({
                        'original_name': name,
                        'request_id': request_id
                    })
            else:
                # 单个名称，直接标准化（简单清理）
                name = group[0]
                standardized_names[name] = self._simple_standardize_name(name)
        
        # 批量处理标准化请求
        if standardization_requests:
            batch_results = self._process_name_standardization(standardization_requests)
            
            # 应用标准化结果
            for result in batch_results:
                original_name = result['original_name']
                standardized_name = result['standardized_name']
                
                standardized_names[original_name] = standardized_name
        
        self.optimization_stats['entities_processed'] += len(standardization_requests)
        logger.info(f"投资方名称标准化优化完成: {len(standardization_requests)} 个名称")
        
        return standardized_names
    
    def process_all_pending_enhancements(self) -> Dict:
        """处理所有待处理的增强请求"""
        logger.info("开始处理所有待处理的LLM增强请求")
        
        start_time = datetime.now()
        
        # 获取待处理请求统计
        pending_stats = self.enhancement_tracker.get_stats()
        
        if pending_stats['pending_requests'] == 0:
            logger.info("没有待处理的增强请求")
            return {
                'processed': 0,
                'results': [],
                'processing_time': 0.0
            }
        
        logger.info(f"待处理请求: {pending_stats['pending_requests']} 个")
        
        # 批量处理
        batch_size = self.config['default_batch_size']
        all_results = []
        total_processed = 0
        
        while pending_stats['pending_requests'] > 0:
            batch_result = self.enhancement_tracker.process_batch_requests(
                self.llm_client,
                batch_size=batch_size
            )
            
            all_results.extend(batch_result['results'])
            total_processed += batch_result['processed']
            
            # 更新统计
            pending_stats = self.enhancement_tracker.get_stats()
            
            if batch_result['remaining'] == 0:
                break
        
        # 更新统计
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        self.optimization_stats['llm_calls_made'] += total_processed
        self.optimization_stats['processing_time'] += processing_time
        
        logger.info(f"批量处理完成: {total_processed} 个请求，耗时 {processing_time:.2f} 秒")
        
        return {
            'processed': total_processed,
            'results': all_results,
            'processing_time': processing_time
        }
    
    def get_optimization_stats(self) -> Dict:
        """获取优化统计"""
        return {
            'entities_processed': self.optimization_stats['entities_processed'],
            'relationships_processed': self.optimization_stats['relationships_processed'],
            'llm_calls_made': self.optimization_stats['llm_calls_made'],
            'batch_sizes': dict(self.optimization_stats['batch_sizes']),
            'processing_time': self.optimization_stats['processing_time'],
            'efficiency_ratio': self._calculate_efficiency_ratio()
        }
    
    def export_enhancement_queue(self, filepath: str) -> None:
        """导出增强队列"""
        self.enhancement_tracker.export_pending_requests(filepath)
    
    def _needs_description_enhancement(self, entity_data: Dict) -> bool:
        """检查是否需要描述增强"""
        description = entity_data.get('description', '')
        
        # 如果描述为空或太短
        if not description or len(description.strip()) < 20:
            return True
        
        # 如果置信度低于阈值
        confidence = entity_data.get('metadata', {}).get('confidence', 0)
        if confidence < CONFIDENCE_THRESHOLDS['entity_linking']:
            return True
        
        return False
    
    def _get_enhancement_priority(self, entity_data: Dict) -> str:
        """获取增强优先级"""
        confidence = entity_data.get('metadata', {}).get('confidence', 0)
        
        if confidence < 0.3:
            return 'high'
        elif confidence < 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _group_conflicts_by_similarity(self, conflicts: List[Dict]) -> Dict[str, List[Dict]]:
        """按相似性分组冲突"""
        # 简单的分组逻辑（实际可以更复杂）
        groups = defaultdict(list)
        
        for conflict in conflicts:
            # 基于名称相似性分组
            name = conflict.get('name', '')
            group_key = self._get_similarity_group_key(name)
            groups[group_key].append(conflict)
        
        return dict(groups)
    
    def _get_similarity_group_key(self, name: str) -> str:
        """获取相似性分组键"""
        # 简单的关键词提取
        keywords = ['科技', '投资', '资本', '基金', '创业', '创新']
        
        for keyword in keywords:
            if keyword in name:
                return keyword
        
        return name[:2]  # 前两个字符作为分组键
    
    def _group_texts_for_processing(self, text_sources: List[Dict]) -> List[List[Dict]]:
        """为处理分组文本"""
        # 按来源类型分组
        groups = defaultdict(list)
        
        for source in text_sources:
            source_type = source.get('type', 'unknown')
            groups[source_type].append(source)
        
        return list(groups.values())
    
    def _combine_related_texts(self, text_group: List[Dict]) -> str:
        """合并相关文本"""
        combined_text = []
        
        for source in text_group:
            text = source.get('text', '')
            if text and len(text.strip()) > 10:
                combined_text.append(text.strip())
        
        return '\\n'.join(combined_text)
    
    def _needs_industry_classification(self, company_data: Dict) -> bool:
        """检查是否需要行业分类"""
        industry = company_data.get('industry', [])
        
        # 如果没有行业信息
        if not industry:
            return True
        
        # 如果行业信息置信度低
        confidence = company_data.get('metadata', {}).get('industry_confidence', 0)
        if confidence < CONFIDENCE_THRESHOLDS['entity_linking']:
            return True
        
        return False
    
    def _get_classification_priority(self, company_data: Dict) -> str:
        """获取分类优先级"""
        # 基于公司规模、融资情况等确定优先级
        capital = company_data.get('capital', '')
        
        if capital and ('亿' in capital or '万' in capital):
            return 'high'
        else:
            return 'medium'
    
    def _group_similar_investor_names(self, names: Set[str]) -> List[List[str]]:
        """分组相似的投资方名称"""
        groups = []
        processed = set()
        
        for name in names:
            if name in processed:
                continue
            
            # 找到相似的名称
            similar_names = [name]
            for other_name in names:
                if other_name != name and self._are_names_similar(name, other_name):
                    similar_names.append(other_name)
                    processed.add(other_name)
            
            groups.append(similar_names)
            processed.add(name)
        
        return groups
    
    def _are_names_similar(self, name1: str, name2: str) -> bool:
        """检查名称是否相似"""
        # 简单的相似性检查
        if name1 == name2:
            return True
        
        # 检查是否包含相同的关键词
        keywords1 = set(name1.replace('有限公司', '').replace('股份有限公司', '').split())
        keywords2 = set(name2.replace('有限公司', '').replace('股份有限公司', '').split())
        
        # 如果有共同的关键词
        if keywords1 & keywords2:
            return True
        
        return False
    
    def _simple_standardize_name(self, name: str) -> str:
        """简单标准化名称"""
        # 移除多余空格
        standardized = ' '.join(name.split())
        
        # 统一括号格式
        standardized = standardized.replace('（', '(').replace('）', ')')
        
        return standardized
    
    def _process_description_enhancements(self, requests: List[Dict]) -> List[Dict]:
        """处理描述增强请求"""
        results = []
        
        # 批量处理所有请求
        batch_result = self.enhancement_tracker.process_batch_requests(
            self.llm_client,
            batch_size=len(requests)
        )
        
        # 整理结果
        for request in requests:
            # 找到对应的结果
            result = next((r for r in batch_result['results'] if isinstance(r, str)), None)
            
            if result:
                results.append({
                    'entity_id': request['entity_id'],
                    'enhanced_description': result,
                    'confidence': 0.8
                })
        
        return results
    
    def _process_conflict_resolution(self, conflicts: List[Dict]) -> Dict[str, Any]:
        """处理冲突解决"""
        # 批量处理冲突解决
        batch_result = self.enhancement_tracker.process_batch_requests(
            self.llm_client,
            batch_size=1  # 一次处理一个冲突组
        )
        
        if batch_result['results']:
            result = batch_result['results'][0]
            if isinstance(result, dict):
                return result
            else:
                return {'resolved_entity': result, 'confidence': 0.7}
        
        return {'resolved_entity': conflicts[0], 'confidence': 0.5}
    
    def _process_relationship_extraction(self) -> List[Dict]:
        """处理关系提取"""
        # 批量处理所有关系提取请求
        batch_result = self.enhancement_tracker.process_batch_requests(
            self.llm_client,
            batch_size=self.config['default_batch_size']
        )
        
        # 整理结果
        results = []
        for result in batch_result['results']:
            if isinstance(result, list):
                results.append({
                    'relationships': result,
                    'confidence': 0.7
                })
        
        return results
    
    def _process_industry_classification(self, requests: List[Dict]) -> List[Dict]:
        """处理行业分类"""
        # 批量处理分类请求
        batch_result = self.enhancement_tracker.process_batch_requests(
            self.llm_client,
            batch_size=len(requests)
        )
        
        # 整理结果
        results = []
        for request, result in zip(requests, batch_result['results']):
            if isinstance(result, list):
                results.append({
                    'company_id': request['company_id'],
                    'industries': result
                })
        
        return results
    
    def _process_name_standardization(self, requests: List[Dict]) -> List[Dict]:
        """处理名称标准化"""
        # 批量处理标准化请求
        batch_result = self.enhancement_tracker.process_batch_requests(
            self.llm_client,
            batch_size=len(requests)
        )
        
        # 整理结果
        results = []
        for request, result in zip(requests, batch_result['results']):
            if isinstance(result, str):
                results.append({
                    'original_name': request['original_name'],
                    'standardized_name': result
                })
        
        return results
    
    def _calculate_efficiency_ratio(self) -> float:
        """计算效率比率"""
        llm_calls_made = self.optimization_stats['llm_calls_made']
        if llm_calls_made == 0:
            return 0.0
        
        total_processed = (self.optimization_stats['entities_processed'] + 
                          self.optimization_stats['relationships_processed'])
        
        return float(total_processed) / float(llm_calls_made)