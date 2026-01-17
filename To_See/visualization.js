// 配置参数
const config = {
    width: window.innerWidth,
    height: window.innerHeight - 120, // 减去 header 高度
    nodeRadius: 8,
    nodeRadiusHover: 20,
    linkDistance: 100,
    chargeStrength: -300,
    showLabels: true,
    zoomScale: 2.5, // 悬停时的缩放倍数
    zoomDuration: 800 // 缩放动画时长（毫秒）
};

// 全局变量
let svg, container, simulation, nodes, links, nodeElements, linkElements, labelElements;
let tooltip;
let hoveredNode = null;
let zoom = d3.zoom();
let currentTransform = d3.zoomIdentity;
let isZooming = false; // 是否正在放大动画中（放大期间暂停鼠标位置判定）
let mousePosition = null; // 记录鼠标位置，用于检测鼠标是否移动
let allowLeave = false; // 是否允许离开节点（放大完成后，如果鼠标移动了则允许）

// 初始化可视化
function initVisualization() {
    console.log('🚀 初始化可视化系统...');
    console.log(`📅 当前时间: ${new Date().toLocaleString()}`);
    console.log(`📁 代码文件版本: visualization.js (请检查浏览器是否缓存了旧版本)`);
    
    // 创建 SVG
    svg = d3.select('#visualization')
        .append('svg')
        .attr('width', config.width)
        .attr('height', config.height);

    // 创建工具提示
    tooltip = d3.select('body')
        .append('div')
        .attr('class', 'tooltip');

    // 加载数据并创建可视化
    loadData();
}

// 加载数据
function loadData() {
    //const dataFile = 'mora_tree_node_link.json';
     const dataFile = 'mora_test.json';
    console.log('='.repeat(60));
    console.log('📊 开始加载数据');
    console.log('='.repeat(60));
    console.log(`📁 尝试加载文件: ${dataFile}`);
    console.log(`⏰ 加载开始时间: ${new Date().toLocaleTimeString()}`);
    
    // 显示加载提示
    const loadingStartTime = Date.now();
    
    // 尝试从 data.json 加载数据，如果没有则使用示例数据
    d3.json(dataFile)
        .then(data => {
            const loadingTime = ((Date.now() - loadingStartTime) / 1000).toFixed(2);
            console.log('✅ 成功加载数据文件:', dataFile);
            console.log(`⏱️ 加载耗时: ${loadingTime} 秒`);
            console.log('📈 数据统计:');
            console.log(`   - 节点数量: ${data.nodes ? data.nodes.length : 0}`);
            console.log(`   - 链接数量: ${data.links ? data.links.length : 0}`);
            
            // 检查文件大小（估算）
            const estimatedSize = JSON.stringify(data).length;
            const sizeInMB = (estimatedSize / 1024 / 1024).toFixed(2);
            console.log(`   - 数据大小（估算）: ${sizeInMB} MB`);
            
            if (data.nodes && data.nodes.length > 0) {
                console.log('📋 前5个节点示例:');
                data.nodes.slice(0, 5).forEach((node, i) => {
                    console.log(`   ${i + 1}. ID: ${node.id}, Name: ${node.description || 'N/A'}, Type: ${node.type || 'N/A'}`);
                });
            }
            
            if (data.links && data.links.length > 0) {
                console.log('🔗 前5个链接示例:');
                data.links.slice(0, 5).forEach((link, i) => {
                    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                    const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                    console.log(`   ${i + 1}. ${sourceId} → ${targetId}`);
                });
            }
            
            console.log('='.repeat(60));
            createVisualization(data);
        })
        .catch(error => {
            console.warn('⚠️ 无法加载数据文件:', dataFile);
            console.warn('❌ 错误详情:', error);
            console.log('🔄 切换到使用示例数据...');
            console.log('='.repeat(60));
            const sampleData = generateSampleData();
            console.log('✅ 示例数据生成完成');
            console.log(`📈 示例数据统计: ${sampleData.nodes.length} 个节点, ${sampleData.links.length} 个链接`);
            console.log('='.repeat(60));
            createVisualization(sampleData);
        });
}

// 创建可视化
function createVisualization(data) {
    console.log('🎨 开始创建可视化...');
    console.log(`📊 输入数据: ${data.nodes ? data.nodes.length : 0} 个节点, ${data.links ? data.links.length : 0} 个链接`);
    
    // 更新节点和链接数据
    nodes = data.nodes.map(d => Object.assign(d, { 
        radius: config.nodeRadius,
        originalRadius: config.nodeRadius 
    }));
    links = data.links.map(d => Object.assign({}, d));

    console.log(`✅ 数据处理完成: ${nodes.length} 个节点, ${links.length} 个链接`);
    
    // 验证数据：检查所有链接引用的节点是否存在
    console.log('🔍 开始验证数据完整性...');
    const nodeIds = new Set(nodes.map(n => n.id));
    const missingNodes = [];
    
    links.forEach((link, index) => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;
        
        if (!nodeIds.has(sourceId)) {
            missingNodes.push({ type: 'source', id: sourceId, linkIndex: index });
        }
        if (!nodeIds.has(targetId)) {
            missingNodes.push({ type: 'target', id: targetId, linkIndex: index });
        }
    });
    
    if (missingNodes.length > 0) {
        console.error('❌ 数据验证失败！发现以下问题：');
        missingNodes.forEach(missing => {
            console.error(`   - 链接 #${missing.linkIndex} 的${missing.type}节点 "${missing.id}" 在节点列表中不存在`);
        });
        console.error('💡 建议：请检查 JSON 文件中的链接数据，确保所有 source 和 target 的 ID 都在节点列表中存在');
        console.error('📋 可用的节点 ID 示例（前10个）：');
        Array.from(nodeIds).slice(0, 10).forEach(id => console.error(`   - ${id}`));
        throw new Error(`数据验证失败：发现 ${missingNodes.length} 个无效的节点引用`);
    }
    
    console.log('✅ 数据验证通过：所有链接引用的节点都存在');
    
    // 更新统计信息
    updateStats();
    
    console.log('🔧 开始创建SVG和力导向图...');

    // 创建容器组（用于缩放和平移）
    container = svg.append('g')
        .attr('class', 'container');

    // 设置缩放行为
    zoom
        .scaleExtent([0.5, 5])
        .on('zoom', (event) => {
            currentTransform = event.transform;
            container.attr('transform', event.transform);
        });

    svg.call(zoom);

    // 创建力导向模拟
    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(config.linkDistance))
        .force('charge', d3.forceManyBody().strength(config.chargeStrength))
        .force('center', d3.forceCenter(config.width / 2, config.height / 2))
        .force('collision', d3.forceCollide().radius(d => d.radius + 5));

    // 创建链接
    linkElements = container.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('class', 'link')
        .attr('stroke-opacity', 0.6);

    // 创建节点组
    const nodeGroups = container.append('g')
        .attr('class', 'nodes')
        .selectAll('g')
        .data(nodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragStarted)
            .on('drag', dragged)
            .on('end', dragEnded));

    // 创建节点圆圈
    nodeElements = nodeGroups.append('circle')
        .attr('r', d => d.radius)
        .attr('fill', d => d.color || '#fff');

    // 创建节点标签
    labelElements = nodeGroups.append('text')
        .attr('dy', d => d.radius + 15)
        .text(d => d.text || d.id)
        .attr('class', d => config.showLabels ? 'show-label' : '');

    // 添加鼠标事件
    nodeGroups
        .on('mouseenter', handleMouseEnter)
        .on('mouseleave', handleMouseLeave)
        .on('mousemove', handleMouseMove);

    // 更新位置
    simulation.on('tick', ticked);

    // 添加控制按钮事件
    setupControls();
    
    console.log('✅ 可视化创建完成！');
    console.log('='.repeat(60));
    console.log('💡 提示: 将鼠标悬停在节点上可以查看放大效果');
    console.log('='.repeat(60));
}

// 更新位置
function ticked() {
    linkElements
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

    nodeElements
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);

    labelElements
        .attr('x', d => d.x)
        .attr('y', d => d.y);
}

// 鼠标进入节点
function handleMouseEnter(event, d) {
    // 如果正在放大动画中，忽略新的鼠标进入事件
    if (isZooming) {
        console.log('⏸️ 鼠标进入节点被忽略（正在放大动画中）:', d.id);
        return;
    }
    
    console.log('🖱️ 鼠标进入节点:', d.id, d.description || 'N/A');
    
    // 如果已经有其他节点被选中，先清除所有高亮
    if (hoveredNode && hoveredNode !== d) {
        console.log('🔄 清除之前选中的节点:', hoveredNode.id);
        clearAllHighlights();
    }
    
    hoveredNode = d;
    
    // 记录鼠标位置（用于检测放大完成后鼠标是否移动）
    mousePosition = { x: event.pageX, y: event.pageY };
    allowLeave = false; // 重置允许离开标志
    
    // 设置放大标志，停止判定鼠标位置
    isZooming = true;
    
    // 放大当前节点
    d3.select(event.currentTarget)
        .classed('highlighted', true)
        .transition()
        .duration(300)
        .ease(d3.easeCubicOut);

    nodeElements
        .filter(node => node === d)
        .transition()
        .duration(300)
        .attr('r', config.nodeRadiusHover);

    // 高亮连接的节点和链接
    const connectedNodeIds = new Set([d.id]);
    links.forEach(link => {
        if (link.source.id === d.id || link.target.id === d.id) {
            connectedNodeIds.add(link.source.id === d.id ? link.target.id : link.source.id);
        }
    });

    nodeElements
        .filter(node => connectedNodeIds.has(node.id) && node !== d)
        .transition()
        .duration(300)
        .attr('r', d => d.radius * 1.5);

    linkElements
        .filter(link => link.source.id === d.id || link.target.id === d.id)
        .classed('highlighted', true);

    // 聚焦并放大到当前节点
    console.log('🔍 开始聚焦并放大节点:', d.id);
    focusOnNode(d);

    // 淡化其他节点
    nodeElements
        .filter(node => !connectedNodeIds.has(node.id))
        .transition()
        .duration(300)
        .attr('opacity', 0.3);

    linkElements
        .filter(link => link.source.id !== d.id && link.target.id !== d.id)
        .transition()
        .duration(300)
        .attr('opacity', 0.1);

    // 显示工具提示
    showTooltip(event, d);
}

// 鼠标离开节点
function handleMouseLeave(event, d) {
    // 如果正在放大动画中，忽略鼠标离开事件
    // 等待放大完成后再判定
    if (isZooming) return;
    
    // 如果离开的不是当前悬停的节点，也忽略
    if (hoveredNode !== d) return;
    
    // 放大完成后，只有鼠标移动了才允许离开
    // 如果鼠标已经移动过，clearAllHighlights 已经在 handleMouseMove 中调用了
    if (!allowLeave) return;
    
    // 恢复到初始状态
    clearAllHighlights();
}

// 聚焦到节点（缩放和平移）
function focusOnNode(d) {
    if (!d.x || !d.y) {
        console.warn('⚠️ 无法聚焦节点（节点位置未定义）:', d.id);
        return;
    }
    
    console.log(`🔍 聚焦节点: ${d.id}, 位置: (${d.x.toFixed(2)}, ${d.y.toFixed(2)})`);

    // 计算目标位置（将节点移到视图中心）
    const targetX = config.width / 2 - d.x * config.zoomScale;
    const targetY = config.height / 2 - d.y * config.zoomScale;

    // 创建新的变换
    const newTransform = d3.zoomIdentity
        .translate(targetX, targetY)
        .scale(config.zoomScale);

    // 平滑过渡到新位置
    svg.transition()
        .duration(config.zoomDuration)
        .ease(d3.easeCubicOut)
        .call(zoom.transform, newTransform)
        .on('end', () => {
            // 动画完成后，重新启用鼠标判定
            // 此时如果鼠标移动了，则允许离开节点
            isZooming = false;
            console.log('✅ 放大动画完成，重新启用鼠标判定');
            // 鼠标移动检测在 handleMouseMove 中处理
        });
}

// 重置视图
function resetView() {
    // 清理状态
    mousePosition = null;
    allowLeave = false;
    svg.on('mousemove.zoomComplete', null);
    
    // 平滑恢复到原始视图
    svg.transition()
        .duration(config.zoomDuration)
        .ease(d3.easeCubicOut)
        .call(zoom.transform, d3.zoomIdentity);
}

// 清除所有高亮状态（恢复到初始状态）
function clearAllHighlights() {
    console.log('🔄 清除所有高亮状态，恢复到初始状态');
    // 清除所有节点高亮
    nodeElements
        .classed('highlighted', false)
        .transition()
        .duration(300)
        .attr('r', d => d.radius)
        .attr('opacity', 1);

    // 清除所有链接高亮
    linkElements
        .classed('highlighted', false)
        .transition()
        .duration(300)
        .attr('opacity', 0.6);

    // 恢复视图
    resetView();

    // 隐藏工具提示
    hideTooltip();

    // 清除状态
    hoveredNode = null;
    mousePosition = null;
    allowLeave = false;
}

// 鼠标移动
function handleMouseMove(event, d) {
    if (hoveredNode === d) {
        updateTooltipPosition(event);
        
        // 如果放大已完成且鼠标移动了，先恢复到初始状态
        if (!isZooming && mousePosition) {
            const moved = Math.abs(event.pageX - mousePosition.x) > 5 || 
                         Math.abs(event.pageY - mousePosition.y) > 5;
            if (moved) {
                console.log('🖱️ 检测到鼠标移动，恢复到初始状态');
                // 先恢复到初始无高亮状态
                clearAllHighlights();
                // 清除位置记录
                mousePosition = null;
            }
        }
    }
}

// 显示工具提示
function showTooltip(event, d) {
    tooltip
        .html(`<strong>${d.description || d.id}</strong>${d.text ? '<br>' + d.text : ''}`)
        .classed('visible', true);
    
    updateTooltipPosition(event);
}

// 更新工具提示位置
function updateTooltipPosition(event) {
    tooltip
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
}

// 隐藏工具提示
function hideTooltip() {
    tooltip.classed('visible', false);
}

// 拖拽开始
function dragStarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

// 拖拽中
function dragged(event, d) {
    // 获取相对于 container 的坐标
    // 使用 d3.pointer 获取相对于 container 的坐标
    const pointer = d3.pointer(event, container.node());
    const transform = currentTransform;
    // 将屏幕坐标转换为数据坐标
    const x = (pointer[0] - transform.x) / transform.k;
    const y = (pointer[1] - transform.y) / transform.k;
    d.fx = x;
    d.fy = y;
}

// 拖拽结束
function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// 设置控制按钮
function setupControls() {
    // 重置视图
    d3.select('#resetBtn').on('click', () => {
        simulation.alpha(1).restart();
        nodes.forEach(d => {
            d.fx = null;
            d.fy = null;
        });
        resetView();
    });

    // 切换标签
    d3.select('#toggleLabels').on('click', () => {
        config.showLabels = !config.showLabels;
        labelElements.classed('show-label', config.showLabels);
    });
}

// 更新统计信息
function updateStats() {
    const nodeCount = nodes.length;
    const linkCount = links.length;
    
    d3.select('#nodeCount').text(nodeCount);
    d3.select('#linkCount').text(linkCount);
    
    console.log(`📊 统计信息已更新: ${nodeCount} 个节点, ${linkCount} 个链接`);
}

// 生成示例数据（树状结构）
function generateSampleData() {
    console.log('🔧 开始生成示例数据（树状结构）...');
    const nodes = [];
    const links = [];
    const colors = d3.schemeCategory10;
    let nodeIdCounter = 0;

    // 创建根节点
    const rootId = `node_${nodeIdCounter++}`;
    nodes.push({
        id: rootId,
        name: '根节点',
        description: '树的根节点',
        color: colors[0]
    });

    // 树状结构配置：每层的分支数和最大深度
    const branchesPerLevel = [3, 3, 2, 2]; // 每层每个节点有几个子节点
    const maxDepth = branchesPerLevel.length;

    // 递归生成树状结构
    function generateTree(parentId, level, path) {
        if (level >= maxDepth) return;

        const branchCount = branchesPerLevel[level] || 2;
        const currentPath = path ? `${path}_` : '';
        
        for (let i = 0; i < branchCount; i++) {
            const nodeId = `node_${nodeIdCounter++}`;
            const nodeName = `第${level + 1}层节点${String.fromCharCode(65 + i)}`;
            
            nodes.push({
                id: nodeId,
                name: nodeName,
                description: `第 ${level + 1} 层的第 ${i + 1} 个子节点`,
                color: colors[(level + 1) % colors.length]
            });

            // 创建父子链接
            links.push({
                source: parentId,
                target: nodeId
            });

            // 递归生成子节点
            generateTree(nodeId, level + 1, `${currentPath}${i}`);
        }
    }

    // 从根节点开始生成树
    generateTree(rootId, 0, '');

    console.log(`✅ 示例数据生成完成: ${nodes.length} 个节点, ${links.length} 个链接`);
    return { nodes, links };
}

// 窗口大小改变时调整
window.addEventListener('resize', () => {
    config.width = window.innerWidth;
    config.height = window.innerHeight - 120;
    
    if (svg) {
        svg.attr('width', config.width).attr('height', config.height);
        if (simulation) {
            simulation.force('center', d3.forceCenter(config.width / 2, config.height / 2));
            simulation.alpha(1).restart();
        }
    }
});

// 初始化
initVisualization();
