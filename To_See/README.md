# 交互式节点网络可视化

这是一个基于 D3.js 的交互式节点链接图可视化工具，支持鼠标悬停放大效果。

## 功能特点

- 🎯 **鼠标悬停放大**：鼠标移动到节点上时，节点会自动放大并高亮显示
- 🔗 **连接高亮**：悬停节点时，相关的连接线也会高亮显示
- 🎨 **美观界面**：现代化的渐变背景和流畅的动画效果
- 🖱️ **拖拽交互**：可以拖拽节点来调整布局
- 📊 **实时统计**：显示节点数和连接数
- 🏷️ **标签切换**：可以切换显示/隐藏节点标签

## 运行方法

### 方法一：使用 Python 简单 HTTP 服务器（推荐）

1. 打开终端/命令提示符
2. 进入 `To_See` 目录：
   ```bash
   cd To_See
   ```

3. 启动 HTTP 服务器：

   **Python 3:**
   ```bash
   python -m http.server 8000
   ```

   **Python 2:**
   ```bash
   python -m http.server 8000   
   ```

4. 在浏览器中打开：
   ```
   http://localhost:8000
   ```

### 方法二：使用 Node.js http-server

1. 安装 http-server（如果还没有安装）：
   ```bash
   npm install -g http-server
   ```

2. 进入 `To_See` 目录：
   ```bash
   cd To_See
   ```

3. 启动服务器：
   ```bash
   http-server -p 8000
   ```

4. 在浏览器中打开：
   ```
   http://localhost:8000
   ```

### 方法三：直接打开（可能有限制）

由于浏览器的安全策略，直接双击打开 `index.html` 可能无法加载 JSON 数据文件。建议使用方法一或方法二。

## 文件结构

```
To_See/
├── index.html          # 主 HTML 文件
├── style.css           # 样式文件
├── visualization.js    # 可视化逻辑
├── data.json           # 节点和链接数据（可选）
└── README.md           # 说明文档
```

## 自定义数据

你可以编辑 `data.json` 文件来自定义节点和链接：

```json
{
  "nodes": [
    {"id": "node_1", "name": "节点名称", "description": "节点描述", "color": "#ff6b6b"}
  ],
  "links": [
    {"source": "node_1", "target": "node_2"}
  ]
}
```

如果没有 `data.json` 文件，程序会自动生成示例数据。

## 使用技巧

- **拖拽节点**：点击并拖拽节点可以调整布局
- **重置视图**：点击"重置视图"按钮可以重新开始模拟
- **切换标签**：点击"切换标签"按钮可以显示/隐藏节点名称
- **鼠标悬停**：将鼠标移到节点上可以看到放大效果和相关信息

## 技术栈

- D3.js v7：用于数据可视化和力导向图
- HTML5 / CSS3：页面结构和样式
- JavaScript (ES6+)：交互逻辑

## 浏览器兼容性

- Chrome / Edge（推荐）
- Firefox
- Safari
- Opera

建议使用最新版本的现代浏览器以获得最佳体验。
