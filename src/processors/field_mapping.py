"""
字段映射配置
用于解决数据解析器和知识图谱构建器之间的字段名称不匹配问题
"""

# 公司数据字段映射
COMPANY_FIELD_MAPPING = {
    # 解析器生成的字段 -> 构建器期望的字段
    'short_name': '公司名称',
    'full_name': '公司全称',
    'description': '公司介绍',
    'legal_representative': '法人代表',
    'registered_capital': '注册资金',
    'establishment_date': '成立时间',
    'address': '地址',
    'website': '网址',
    'registration_id': '工商注册id',
    'unified_credit_code': '统一信用代码'
}

# 投资事件字段映射
INVESTMENT_EVENT_FIELD_MAPPING = {
    'description': '事件描述',
    'investors': '投资方',
    'investee': '融资方',
    'amount': '金额',
    'round': '轮次',
    'date': '融资时间',
    'industry': '行业',
    'location': '地区'
}

# 投资方字段映射
INVESTOR_FIELD_MAPPING = {
    'name': '机构名称',
    'description': '介绍',
    'scale': '规模',
    'preferred_rounds': '轮次',
    'investment_focus': '投资领域',
    'established_date': '成立时间',
    'location': '所在地'
}

def apply_field_mapping(data_list, field_mapping):
    """
    应用字段映射到数据列表
    
    Args:
        data_list: 原始数据列表
        field_mapping: 字段映射字典
        
    Returns:
        映射后的数据列表
    """
    if not data_list:
        return data_list
    
    mapped_data = []
    for item in data_list:
        if not isinstance(item, dict):
            mapped_data.append(item)
            continue
            
        mapped_item = {}
        for old_field, new_field in field_mapping.items():
            if old_field in item:
                mapped_item[new_field] = item[old_field]
            else:
                mapped_item[new_field] = ''
        
        # 保留原始字段作为备份
        mapped_item['_original_data'] = item.copy()
        mapped_data.append(mapped_item)
    
    return mapped_data

def reverse_field_mapping(data_list, field_mapping):
    """
    反向字段映射（从构建器格式转换回解析器格式）
    
    Args:
        data_list: 构建器格式的数据列表
        field_mapping: 字段映射字典
        
    Returns:
        解析器格式的数据列表
    """
    if not data_list:
        return data_list
    
    reversed_mapping = {v: k for k, v in field_mapping.items()}
    return apply_field_mapping(data_list, reversed_mapping)