#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * 过滤CSV文件中包含错误字符的行
 * @param {string} inputFile - 输入CSV文件路径
 * @param {string} outputFile - 输出CSV文件路径
 * @param {string} errorChar - 需要过滤的错误字符，默认为�
 */
function filterCSVWithErrorChars(inputFile, outputFile, errorChar = '�') {
    try {
        // 检查输入文件是否存在
        if (!fs.existsSync(inputFile)) {
            console.error(`错误：输入文件不存在 - ${inputFile}`);
            process.exit(1);
        }

        // 读取原始CSV文件
        console.log(`正在读取文件：${inputFile}`);
        const fileContent = fs.readFileSync(inputFile, 'utf8');
        
        // 按行分割内容
        const lines = fileContent.split(/\r?\n/);
        const originalLineCount = lines.length;
        
        console.log(`原始文件共有 ${originalLineCount} 行`);
        
        // 过滤包含错误字符的行
        const filteredLines = lines.filter(line => {
            return !line.includes(errorChar);
        });
        
        const filteredLineCount = filteredLines.length;
        const removedLineCount = originalLineCount - filteredLineCount;
        
        console.log(`过滤后共有 ${filteredLineCount} 行`);
        console.log(`删除了 ${removedLineCount} 行包含错误字符的记录`);
        
        if (removedLineCount > 0) {
            // 显示一些被删除的行作为示例（只显示前5行）
            const removedLines = lines.filter(line => line.includes(errorChar));
            console.log('\n被删除的行示例：');
            removedLines.slice(0, 5).forEach((line, index) => {
                console.log(`  ${index + 1}: ${line}`);
            });
            if (removedLines.length > 5) {
                console.log(`  ... 还有 ${removedLines.length - 5} 行`);
            }
        }
        
        // 写入过滤后的内容
        const outputDir = path.dirname(outputFile);
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        fs.writeFileSync(outputFile, filteredLines.join('\n'), 'utf8');
        console.log(`\n过滤后的文件已保存至：${outputFile}`);
        
        // 统计信息
        const successRate = ((filteredLineCount / originalLineCount) * 100).toFixed(2);
        console.log(`数据保留率：${successRate}%`);
        
        return {
            originalCount: originalLineCount,
            filteredCount: filteredLineCount,
            removedCount: removedLineCount,
            successRate: successRate
        };
        
    } catch (error) {
        console.error('处理文件时发生错误：', error.message);
        process.exit(1);
    }
}

// 主函数
function main() {
    // 配置输入和输出文件路径
    const inputFile = path.resolve(__dirname, 'SmoothNLP工商数据集样本10K.csv');
    const outputFile = path.resolve(__dirname, 'SmoothNLP工商数据集样本10K_filtered.csv');
    
    console.log('=== CSV文件错误字符过滤工具 ===\n');
    
    // 执行过滤
    const result = filterCSVWithErrorChars(inputFile, outputFile, '�');
    
    console.log('\n=== 过滤完成 ===');
}

// 如果直接运行此脚本
if (require.main === module) {
    main();
}

// 导出函数供其他模块使用
module.exports = {
    filterCSVWithErrorChars
};