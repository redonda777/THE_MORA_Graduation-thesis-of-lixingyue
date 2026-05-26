import json

# 1. 定义你的JSON文件路径（把这里改成你的文件实际路径）
json_file_path = r"D:\The_Mora\connect_edit_distance\sentence_edit_distance.json"

try:
    # 2. 打开并读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        # 加载JSON数据（自动解析成Python列表）
        data = json.load(f)
    
    # 3. 检查数据是否是列表（适配你的格式）
    if isinstance(data, list):
        record_count = len(data)
        print(f"✅ JSON文件中共有 {record_count} 条记录")
    else:
        print("⚠️ 警告：JSON文件不是数组格式，无法统计记录数")

# 处理文件不存在的情况
except FileNotFoundError:
    print(f"❌ 错误：未找到文件 {json_file_path}，请检查文件路径是否正确")
# 处理JSON格式错误的情况
except json.JSONDecodeError:
    print(f"❌ 错误：{json_file_path} 不是合法的JSON文件")
# 其他未知错误
except Exception as e:
    print(f"❌ 未知错误：{str(e)}")