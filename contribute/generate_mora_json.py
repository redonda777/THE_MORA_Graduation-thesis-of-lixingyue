#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成完整的 mora.json 文件
包含12个版本，每个版本76个节点（1个序言 + 75章）
"""

import json

def generate_mora_json():
    # 版本信息
    versions = [
        {"id": "version1", "name": "版本1", "description": "hj", "line": "C"},
        {"id": "version2", "name": "版本2", "description": "gd", "line": "D"},
        {"id": "version3", "name": "版本3", "description": "gd1", "line": "E"},
        {"id": "version4", "name": "版本4", "description": "gd2", "line": "F"},
        {"id": "version5", "name": "版本5", "description": "gd3", "line": "G"},
        {"id": "version6", "name": "版本6", "description": "ba", "line": "h"},
        {"id": "version7", "name": "版本7", "description": "bb", "line": "I"},
        {"id": "version8", "name": "版本8", "description": "wb", "line": "J"},
        {"id": "version9", "name": "版本9", "description": "hs", "line": "K"},
        {"id": "version10", "name": "版本10", "description": "yz", "line": "L"},
        {"id": "version11", "name": "版本11", "description": "xr", "line": "M"},
        {"id": "version12", "name": "版本12", "description": "fy", "line": "N"},
    ]
    
    # 初始化节点列表
    nodes = [
        {"id": "root", "name": "道德经", "description": "作者：老子"}
    ]
    
    # 添加版本节点
    for version in versions:
        nodes.append(version)
    
    # 为每个版本生成76个章节节点（1个序言 + 75章）
    for version_num in range(1, 13):
        # 序言

        
        # 75章
        for chapter_num in range(1, 76):
            nodes.append({
                "id": f"chapter{version_num}_{chapter_num + 1}",
                "name": f"版本{version_num}第{chapter_num}章",
                "description": f"版本{version_num}第{chapter_num}章",
                "line": "A"
            })
    
    # 生成链接
    links = []
    
    # root 连接到各个版本
    for version_num in range(1, 13):
        links.append({
            "source": "root",
            "target": f"version{version_num}"
        })
    
    # 每个版本连接到其所有章节（76个）
    for version_num in range(1, 13):
        for chapter_num in range(0, 76):  # 1到76
            links.append({
                "source": f"version{version_num}",
                "target": f"chapter{version_num}_{chapter_num}"
            })
    
    # 构建完整的JSON结构
    mora_data = {
        "nodes": nodes,
        "links": links
    }
    
    return mora_data

def main():
    import sys
    # 设置输出编码为UTF-8
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("正在生成 mora.json 文件...")
    
    mora_data = generate_mora_json()
    
    # 统计信息
    print(f"[OK] 节点总数: {len(mora_data['nodes'])}")
    print(f"  - 根节点: 1")
    print(f"  - 版本节点: 12")
    print(f"  - 章节节点: {len(mora_data['nodes']) - 13} (12个版本 x 76章)")
    print(f"[OK] 链接总数: {len(mora_data['links'])}")
    print(f"  - root到版本: 12")
    print(f"  - 版本到章节: {len(mora_data['links']) - 12} (12个版本 x 76章)")
    
    # 保存到文件
    output_file = "mora.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mora_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n[OK] 文件已保存到: {output_file}")

if __name__ == "__main__":
    main()

