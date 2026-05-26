const COLORS = [
  '#56c2ff',
  '#47d18c',
  '#f2c14e',
  '#c084fc',
  '#f87171',
  '#38bdf8',
  '#fb923c',
  '#a3e635',
];

const RANK_CLASS_COUNT = 8;
const BLANK_CLICK_DRIFT = 4;
const INITIAL_CHAPTER_DISPLAY = 1;
const BRIGHTNESS_COLORS = [
  '#ffffff',
  '#fef08a',
  '#fde68a',
  '#fcd34d',
  '#fbbf24',
  '#f59e0b',
  '#d97706',
  '#b45309',
  '#92400e',
];

const state = {
  mode: 'version',
  graph: { nodes: [], links: [] },
  selected: null,
  fixed: null,
  overview: null,
  ranking: [],
  chapter: 0,
  chapterData: null,
  bookAffinity: null,
  searchIndex: null,
  zoom: 1,
  panX: 0,
  panY: 0,
};

let layoutFrame = null;
let layoutTicks = 0;

const svg = document.getElementById('graphSvg');
const heatmap = document.getElementById('heatmap');
const appShell = document.querySelector('.app-shell');
const detailAside = document.querySelector('.detail');
const detailPanel = document.getElementById('detailPanel');
const sentencesPanel = document.getElementById('sentencesPanel');
const statsPanel = document.getElementById('statsPanel');
const rankingTitle = document.getElementById('rankingTitle');
const rankingList = document.getElementById('rankingList');
const legend = document.getElementById('legend');
const viewTitle = document.getElementById('viewTitle');
const viewSubtitle = document.getElementById('viewSubtitle');
const chapterControls = document.getElementById('chapterControls');
const chapterInput = document.getElementById('chapterInput');
const chapterSelect = document.getElementById('chapterSelect');
const sentenceSearch = document.getElementById('sentenceSearch');
const sentenceSearchInput = document.getElementById('sentenceSearchInput');
const sentenceSearchBtn = document.getElementById('sentenceSearchBtn');
const sentenceSearchStatus = document.getElementById('sentenceSearchStatus');

function showDetailAside() {
  appShell.classList.remove('matrix-view');
  if (!detailAside.isConnected) appShell.appendChild(detailAside);
}

function removeDetailAside() {
  appShell.classList.add('matrix-view');
  if (detailAside.isConnected) detailAside.remove();
}

function staticJsonPath(url) {
  const [path] = String(url).split('?');
  const chapterMatch = path.match(/^\/api\/chapter\/(\d+)$/);
  if (chapterMatch) return `/api/chapter/${chapterMatch[1]}.json`;
  const routes = {
    '/api/overview': '/api/overview.json',
    '/api/version-graph': '/api/version-graph.json',
    '/api/version-ranking': '/api/version-ranking.json',
    '/api/chapters': '/api/chapters.json',
    '/api/book-affinity': '/api/book-affinity.json',
    '/api/search-index': '/api/search-index.json',
  };
  return routes[path] || url;
}

async function getJson(url) {
  const res = await fetch(staticJsonPath(url));
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function normalizeSearchText(value) {
  return String(value ?? '').replace(/[\s#◆◇■□*·,，.。:：;；!?！？'"“”‘’()[\]（）【】<>《》]+/g, '').toLowerCase();
}

function subsequenceScore(query, text) {
  if (!query || !text) return 0;
  let pos = 0;
  let matched = 0;
  for (const char of query) {
    const found = text.indexOf(char, pos);
    if (found < 0) continue;
    matched += 1;
    pos = found + 1;
  }
  return matched / query.length;
}

function searchScore(query, text) {
  if (!query) return 0;
  if (text.includes(query)) return 2 + query.length / Math.max(1, text.length);
  return subsequenceScore(query, text);
}

async function searchSentences(query, limit = 12) {
  if (!state.searchIndex) {
    const data = await getJson('/api/search-index');
    state.searchIndex = data.items || [];
  }
  const normalizedQuery = normalizeSearchText(query);
  if (!normalizedQuery) return { query, items: [], total: 0 };

  const items = state.searchIndex
    .map(item => {
      const normalizedText = item.normalizedText || normalizeSearchText(item.text);
      const score = searchScore(normalizedQuery, normalizedText);
      return {
        ...item,
        score,
        matchType: normalizedText.includes(normalizedQuery) ? 'contains' : 'fuzzy',
      };
    })
    .filter(item => item.score >= 0.55)
    .sort((a, b) =>
      (b.score - a.score) ||
      ((b.sentenceCount || 0) - (a.sentenceCount || 0)) ||
      String(a.version).localeCompare(String(b.version)) ||
      ((a.sentence || 0) - (b.sentence || 0)),
    );

  items.forEach((item, index) => { item.rank = index + 1; });
  return { query, items: items.slice(0, limit), total: items.length };
}

function communityColor(id) {
  if (!id) return '#64748b';
  return COLORS[(id - 1) % COLORS.length];
}

function nodeVisualColor(node) {
  if (node.type === 'version') return '#a78bfa';
  if (node.type === 'chapter') return '#60a5fa';
  if (node.type === 'sentence') return '#3b82f6';
  return communityColor(node.community);
}

function formatScore(value) {
  return Number(value || 0).toFixed(4);
}

function isMissingScore(value) {
  return value === null || value === undefined || Number.isNaN(Number(value));
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function sentenceLabel(id) {
  const node = state.graph.nodes.find(item => item.id === id);
  if (!node) return id;
  if (node.type === 'sentence') return `${node.version}:${node.sentence}`;
  if (node.type === 'chapter') return `${node.version}`;
  return node.label || node.id;
}

function otherEnd(link, nodeId) {
  return link.source === nodeId ? link.target : link.source;
}

function renderStats() {
  const overview = state.overview;
  if (!overview) return;
  statsPanel.innerHTML = `
    <div class="stat"><strong>${overview.versions.length}</strong><span>版本</span></div>
    <div class="stat"><strong>${overview.chapter_count}</strong><span>章节数</span></div>
    <div class="stat"><strong>${overview.sentence_count.toLocaleString()}</strong><span>句子</span></div>
    <div class="stat"><strong>${overview.summary.chapter_count || 0}</strong><span>社区章节</span></div>
  `;
}

function withLocalRanks(items) {
  return items.map((item, index) => ({ ...item, localRank: index + 1 }));
}

function boundedRank(value) {
  return Math.max(1, Math.min(RANK_CLASS_COUNT, Number(value) || RANK_CLASS_COUNT));
}

function itemRankClass(item) {
  return `rank-${boundedRank(item.localRank || item.rank || 1)}`;
}

function relationStyle(item) {
  const rank = boundedRank(item.localRank || item.rank || 1);
  const color = BRIGHTNESS_COLORS[rank] || BRIGHTNESS_COLORS[BRIGHTNESS_COLORS.length - 1];
  return `style="--rank-color:${color}"`;
}

function getVersionRankingForNode(nodeId) {
  const items = nodeId
    ? state.ranking.filter(item => item.source === nodeId || item.target === nodeId)
    : state.ranking;
  return withLocalRanks([...items].sort((a, b) => (b.similarity || 0) - (a.similarity || 0)));
}

function getChapterRankingForNode(node) {
  if (node?.type === 'sentence') {
    const items = (state.chapterData?.sentenceRanking || []).filter(
      item => item.source === node.id || item.target === node.id,
    );
    return withLocalRanks([...items].sort((a, b) => (a.distance ?? 1) - (b.distance ?? 1)));
  }
  const source = state.chapterData?.ranking || [];
  const items = node?.id
    ? source.filter(item => item.source === node.id || item.target === node.id)
    : source;
  return withLocalRanks([...items].sort((a, b) => (b.similarity || 0) - (a.similarity || 0)));
}

function renderRanking(title, items) {
  rankingTitle.textContent = title;
  rankingList.innerHTML = items.slice(0, 24).map(item => {
    const support = item.support ? `<small>${item.support} 对句子</small>` : '';
    const distance = item.distance !== undefined
      ? `<small>normalized_distance ${formatScore(item.distance)}</small>`
      : '';
    return `
      <div class="rank-row ${itemRankClass(item)}" ${relationStyle(item)} data-source="${item.source}" data-target="${item.target}">
        <b>${item.localRank || item.rank}</b>
        <span>${sentenceLabel(item.source)} - ${sentenceLabel(item.target)}</span>
        <span>${formatScore(item.similarity)}</span>
        ${support}
        ${distance}
      </div>`;
  }).join('');
  rankingList.querySelectorAll('.rank-row').forEach(row => {
    row.addEventListener('click', () => highlightPair(row.dataset.source, row.dataset.target));
  });
}

function updateContextRanking() {
  const node = state.fixed;
  if (state.mode === 'chapter') {
    const chapterLabel = `第 ${state.chapter + 1} 章`;
    if (node?.type === 'sentence') {
      renderRanking(`${chapterLabel}：${sentenceLabel(node.id)} 的句子距离排行`, getChapterRankingForNode(node));
      return;
    }
    const title = node
      ? `${chapterLabel}：${node.id} 的章节相似度排行`
      : `${chapterLabel}：版本相似度排行`;
    renderRanking(title, getChapterRankingForNode(node));
    return;
  }
  if (state.mode === 'version') {
    const title = node ? `${node.id} 的版本相似度排行` : '版本相似度排序';
    renderRanking(title, getVersionRankingForNode(node?.id));
    return;
  }
  renderRanking('版本相似度排序', getVersionRankingForNode());
}

function renderLegend(maxCommunity = 6) {
  legend.innerHTML = '';
}

function linkNodes(nodes, links) {
  const nodeMap = new Map(nodes.map(node => [node.id, node]));
  links.forEach(link => {
    link.sourceNode = nodeMap.get(link.source);
    link.targetNode = nodeMap.get(link.target);
  });
}

function prepareVersionLayout(nodes, links) {
  const rect = svg.getBoundingClientRect();
  const width = rect.width || 900;
  const height = rect.height || 700;
  const radius = Math.min(width, height) * 0.34;
  nodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / nodes.length - Math.PI / 2;
    node.anchorX = width / 2 + Math.cos(angle) * radius;
    node.anchorY = height / 2 + Math.sin(angle) * radius;
    node.x = node.anchorX;
    node.y = node.anchorY;
    node.vx = 0;
    node.vy = 0;
  });
  linkNodes(nodes, links);
}

function applyRankAggregation(nodes, links, activeNode, options = {}) {
  const focus = rankedFocus(activeNode, links);
  if (!activeNode || focus.activeIds.size <= 1) return;

  const center = activeNode;
  const neighbors = [...focus.ranks.entries()]
    .filter(([, rank]) => rank > 0)
    .map(([id, rank]) => ({ node: nodes.find(item => item.id === id), rank }))
    .filter(item => item.node);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const baseRing = options.baseRing ?? 26;
  const rankRing = options.rankRing ?? 15;
  const pull = options.pull ?? 0.04;
  neighbors.forEach(({ node, rank }, index) => {
    const angle = index * goldenAngle;
    const ring = baseRing + Math.sqrt(rank) * rankRing;
    const targetX = center.x + Math.cos(angle) * ring;
    const targetY = center.y + Math.sin(angle) * ring;
    node.vx += (targetX - node.x) * pull;
    node.vy += (targetY - node.y) * pull;
  });
}

function prepareChapterLayout(nodes, links) {
  const rect = svg.getBoundingClientRect();
  const width = rect.width || 900;
  const height = rect.height || 700;
  const chapters = nodes.filter(node => node.type === 'chapter');
  const sentences = nodes.filter(node => node.type === 'sentence');
  const centerX = width / 2;
  const centerY = height / 2;
  const clusterRadius = Math.min(width, height) * 0.33;
  const versionRadius = Math.min(width, height) * 0.12;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  chapters.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(1, chapters.length) - Math.PI / 2;
    node.anchorX = centerX + Math.cos(angle) * versionRadius;
    node.anchorY = centerY + Math.sin(angle) * versionRadius;
    node.x = node.anchorX + Math.cos(angle) * 18;
    node.y = node.anchorY + Math.sin(angle) * 18;
    node.vx = 0;
    node.vy = 0;
  });

  const grouped = new Map();
  sentences.forEach(node => {
    if (!grouped.has(node.version)) grouped.set(node.version, []);
    grouped.get(node.version).push(node);
  });

  chapters.forEach((chapterNode, index) => {
    const rows = (grouped.get(chapterNode.version) || []).sort((a, b) => a.sentence - b.sentence);
    const angle = (Math.PI * 2 * index) / Math.max(1, chapters.length) - Math.PI / 2;
    const clusterX = centerX + Math.cos(angle) * clusterRadius;
    const clusterY = centerY + Math.sin(angle) * clusterRadius;
    rows.forEach((node, rowIndex) => {
      const spoke = rowIndex * goldenAngle;
      const radius = 10 + Math.sqrt(rowIndex + 1) * 11;
      node.anchorX = clusterX + Math.cos(spoke) * radius;
      node.anchorY = clusterY + Math.sin(spoke) * radius;
      node.x = node.anchorX + Math.sin(rowIndex * 1.7) * 16;
      node.y = node.anchorY + Math.cos(rowIndex * 1.3) * 16;
      node.vx = (Math.sin(rowIndex + index) - 0.5) * 0.5;
      node.vy = (Math.cos(rowIndex + index) - 0.5) * 0.5;
    });
  });

  linkNodes(nodes, links);
}

function tickChapter(nodes, links) {
  const rect = svg.getBoundingClientRect();
  const width = rect.width || 900;
  const height = rect.height || 700;
  for (const node of nodes) {
    const anchorStrength = node.type === 'chapter' ? 0.045 : 0.018;
    node.vx += ((node.anchorX ?? node.x) - node.x) * anchorStrength;
    node.vy += ((node.anchorY ?? node.y) - node.y) * anchorStrength;
  }
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i];
      const b = nodes[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dist2 = dx * dx + dy * dy || 1;
      const dist = Math.sqrt(dist2);
      const base = a.version && a.version === b.version ? 1100 : 720;
      const force = base / dist2;
      a.vx += (dx / dist) * force;
      a.vy += (dy / dist) * force;
      b.vx -= (dx / dist) * force;
      b.vy -= (dy / dist) * force;
    }
  }
  for (const link of links) {
    const a = link.sourceNode;
    const b = link.targetNode;
    if (!a || !b) continue;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    let target = 120;
    let strength = 0.004;
    if (link.type === 'contains_sentence') {
      target = 155;
      strength = 0.0025;
    } else if (link.type === 'sentence_similarity') {
      target = 42 + (link.distance ?? 0.08) * 220;
      strength = 0.009;
    } else if (link.type === 'chapter_similarity' || link.type === 'version_similarity') {
      target = 105;
      strength = 0.005;
    }
    const force = (dist - target) * strength;
    a.vx += (dx / dist) * force;
    a.vy += (dy / dist) * force;
    b.vx -= (dx / dist) * force;
    b.vy -= (dy / dist) * force;
  }
  applyRankAggregation(nodes, links, state.fixed, {
    baseRing: 26,
    rankRing: 15,
    pull: state.fixed?.type === 'sentence' ? 0.06 : 0.035,
  });
  for (const node of nodes) {
    node.vx *= 0.78;
    node.vy *= 0.78;
    const speed = Math.sqrt(node.vx * node.vx + node.vy * node.vy);
    if (speed > 7) {
      node.vx = (node.vx / speed) * 7;
      node.vy = (node.vy / speed) * 7;
    }
    node.x = Math.max(24, Math.min(width - 24, node.x + node.vx));
    node.y = Math.max(24, Math.min(height - 24, node.y + node.vy));
  }
}

function tick(nodes, links) {
  const rect = svg.getBoundingClientRect();
  const width = rect.width || 900;
  const height = rect.height || 700;
  for (const node of nodes) {
    node.vx += (width / 2 - node.x) * 0.0008;
    node.vy += (height / 2 - node.y) * 0.0008;
    node.vx += ((node.anchorX ?? node.x) - node.x) * 0.006;
    node.vy += ((node.anchorY ?? node.y) - node.y) * 0.006;
  }
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i];
      const b = nodes[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dist2 = dx * dx + dy * dy || 1;
      const force = 1200 / dist2;
      const dist = Math.sqrt(dist2);
      a.vx += (dx / dist) * force;
      a.vy += (dy / dist) * force;
      b.vx -= (dx / dist) * force;
      b.vy -= (dy / dist) * force;
    }
  }
  for (const link of links) {
    const a = link.sourceNode;
    const b = link.targetNode;
    if (!a || !b) continue;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const target = 130 - (link.similarity || 0) * 55;
    const force = (dist - target) * 0.006;
    a.vx += (dx / dist) * force;
    a.vy += (dy / dist) * force;
    b.vx -= (dx / dist) * force;
    b.vy -= (dy / dist) * force;
  }
  applyRankAggregation(nodes, links, state.fixed, {
    baseRing: 34,
    rankRing: 18,
    pull: 0.026,
  });
  for (const node of nodes) {
    node.vx *= 0.76;
    node.vy *= 0.76;
    const speed = Math.sqrt(node.vx * node.vx + node.vy * node.vy);
    if (speed > 5.2) {
      node.vx = (node.vx / speed) * 5.2;
      node.vy = (node.vy / speed) * 5.2;
    }
    node.x = Math.max(28, Math.min(width - 28, node.x + node.vx));
    node.y = Math.max(28, Math.min(height - 28, node.y + node.vy));
  }
}

function stabilize(nodes, links, count = 180) {
  for (let i = 0; i < count; i++) tick(nodes, links);
}

function stabilizeChapter(nodes, links, count = 140) {
  for (let i = 0; i < count; i++) tickChapter(nodes, links);
}

function stopLayoutAnimation() {
  if (layoutFrame) cancelAnimationFrame(layoutFrame);
  layoutFrame = null;
  layoutTicks = 0;
}

function startChapterAnimation() {
  stopLayoutAnimation();
  const step = () => {
    if (state.mode !== 'chapter' || layoutTicks > 420) {
      layoutFrame = null;
      return;
    }
    tickChapter(state.graph.nodes, state.graph.links);
    layoutTicks += 1;
    drawGraph({ skipRanking: true });
    layoutFrame = requestAnimationFrame(step);
  };
  layoutFrame = requestAnimationFrame(step);
}

function startVersionAnimation() {
  stopLayoutAnimation();
  const step = () => {
    if (state.mode !== 'version' || layoutTicks > 220) {
      layoutFrame = null;
      return;
    }
    tick(state.graph.nodes, state.graph.links);
    layoutTicks += 1;
    drawGraph({ skipRanking: true });
    layoutFrame = requestAnimationFrame(step);
  };
  layoutFrame = requestAnimationFrame(step);
}

function renderGraph(data, options = {}) {
  svg.style.display = '';
  heatmap.style.display = '';
  heatmap.classList.add('hidden');
  svg.classList.remove('hidden');
  state.selected = null;
  state.fixed = null;
  const nodes = data.nodes.map(node => ({ ...node }));
  const links = data.links.map(link => ({ ...link }));
  if (state.mode === 'chapter') {
    prepareChapterLayout(nodes, links);
    stabilizeChapter(nodes, links, 120);
  } else {
    stopLayoutAnimation();
    prepareVersionLayout(nodes, links);
    stabilize(nodes, links, 220);
  }
  state.graph = { nodes, links };
  drawGraph(options);
  if (state.mode === 'chapter') startChapterAnimation();
  if (state.mode === 'version') startVersionAnimation();
}

function isLinkActive(link, activeId) {
  return activeId && (link.source === activeId || link.target === activeId);
}

function rankedFocus(activeNode, links) {
  const activeIds = new Set();
  const ranks = new Map();
  const edgeRanks = new Map();
  if (!activeNode) return { activeIds, ranks, edgeRanks };

  activeIds.add(activeNode.id);
  ranks.set(activeNode.id, 0);

  const directLinks = links
    .map((link, index) => ({ link, index }))
    .filter(item => item.link.source === activeNode.id || item.link.target === activeNode.id);

  const sorted = directLinks.sort((a, b) => {
    const aLink = a.link;
    const bLink = b.link;
    if (activeNode.type === 'sentence') {
      return (aLink.distance ?? 1) - (bLink.distance ?? 1);
    }
    const aScore = aLink.type === 'contains_sentence' ? 0.2 : (aLink.similarity || 0);
    const bScore = bLink.type === 'contains_sentence' ? 0.2 : (bLink.similarity || 0);
    return bScore - aScore;
  });

  sorted.forEach((item, index) => {
    const other = otherEnd(item.link, activeNode.id);
    const rank = Math.min(index + 1, RANK_CLASS_COUNT);
    activeIds.add(other);
    if (!ranks.has(other)) ranks.set(other, rank);
    edgeRanks.set(item.index, rank);
  });

  links.forEach((link, index) => {
    if (!activeIds.has(link.source) || !activeIds.has(link.target)) return;
    if (edgeRanks.has(index)) return;
    const sourceRank = ranks.get(link.source) ?? RANK_CLASS_COUNT;
    const targetRank = ranks.get(link.target) ?? RANK_CLASS_COUNT;
    edgeRanks.set(index, Math.min(RANK_CLASS_COUNT, Math.max(sourceRank, targetRank)));
  });

  return { activeIds, ranks, edgeRanks };
}

function drawGraph(options = {}) {
  const { nodes, links } = state.graph;
  const activeNode = state.fixed;
  const activeId = activeNode?.id;
  const focus = rankedFocus(activeNode, links);

  const edgeMarkup = links.map((link, index) => {
    const a = link.sourceNode;
    const b = link.targetNode;
    if (!a || !b) return '';
    const type = link.type || 'version_similarity';
    const rank = focus.edgeRanks.get(index);
    const active = rank !== undefined || isLinkActive(link, activeId);
    const dim = activeId && !active;
    const rankClass = rank ? `rank-${rank}` : '';
    const width = type === 'contains_sentence'
      ? 0.7
      : 1 + Math.max(0, (link.similarity || 0) - 0.6) * (type === 'sentence_similarity' ? 3.2 : 5);
    const distance = link.distance !== undefined ? `data-distance="${formatScore(link.distance)}"` : '';
    return `<line class="edge ${type} ${active ? 'highlight' : ''} ${rankClass} ${dim ? 'dimmed' : ''}" data-edge="${index}" ${distance} x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke-width="${width.toFixed(2)}" />`;
  }).join('');

  const nodeMarkup = nodes.map(node => {
    const rank = focus.ranks.get(node.id);
    const isFocus = rank === 0;
    const isActive = rank !== undefined;
    const dim = activeId && !focus.activeIds.has(node.id);
    const rankClass = isActive ? (isFocus ? 'focus' : `rank-${rank}`) : '';
    const r = node.type === 'version' ? 6 : (node.type === 'chapter' ? 4 : 2);
    const label = node.type === 'sentence' ? String(node.sentence) : node.label;
    return `
      <g class="node ${node.type || 'version'} ${rankClass} ${dim ? 'dimmed' : ''}" data-id="${node.id}" transform="translate(${node.x}, ${node.y})">
        <circle r="${r}" fill="${nodeVisualColor(node)}"></circle>
        <text text-anchor="middle" dy="${node.type === 'sentence' ? 3 : 4}">${label}</text>
      </g>`;
  }).join('');

  svg.innerHTML = `<g transform="translate(${state.panX},${state.panY}) scale(${state.zoom})">${edgeMarkup}${nodeMarkup}</g>`;
  if (!options.skipRanking) updateContextRanking();
}

function highlightPair(source, target) {
  state.fixed = state.graph.nodes.find(node => node.id === source) || { id: source };
  state.selected = null;
  drawGraph({ skipRanking: true });
  const targetNode = state.graph.nodes.find(node => node.id === target);
  if (targetNode) showNodeDetail(targetNode);
  if (state.mode === 'chapter') startChapterAnimation();
  if (state.mode === 'version') startVersionAnimation();
}

function clearHighlight() {
  state.fixed = null;
  state.selected = null;
  drawGraph();
  if (state.mode === 'chapter') startChapterAnimation();
  if (state.mode === 'version') startVersionAnimation();
}

function showNodeDetail(node) {
  if (!node) return;
  if (node.type === 'sentence') {
    showSentenceDetail(node);
    return;
  }
  const edgeRows = state.graph.links
    .filter(link => (link.source === node.id || link.target === node.id) && link.type !== 'contains_sentence')
    .sort((a, b) => (b.similarity || 0) - (a.similarity || 0))
    .slice(0, 12)
    .map((link, index) => {
      const other = otherEnd(link, node.id);
      const distance = link.distance !== undefined ? ` / distance ${formatScore(link.distance)}` : '';
      const item = { localRank: index + 1 };
      return `<div class="metric-row relation-row ${itemRankClass(item)}" ${relationStyle(item)}><span>${sentenceLabel(other)}</span> similarity ${formatScore(link.similarity || 0)}${distance}</div>`;
    }).join('');
  const chapterRow = node.type === 'chapter'
    ? `<div class="metric-row"><span>章节</span> 第 ${Number(node.chapter) + 1} 章</div>`
    : '';
  const versionRow = node.version
    ? `<div class="metric-row"><span>版本</span> ${node.version}</div>`
    : '';
  detailPanel.innerHTML = `
    <div class="detail-title">${node.label || node.id}</div>
    <div class="metric-row"><span>类型</span> ${node.type || 'version'}</div>
    ${versionRow}
    ${chapterRow}
    <div class="metric-row"><span>社区</span> ${node.community || '-'}</div>
    <div class="metric-row"><span>句子数</span> ${node.sentenceCount ?? '-'}</div>
    <div class="section-title" style="margin-top:12px">关联</div>
    ${edgeRows || '<div class="empty-state">没有关联边</div>'}
  `;
}

function showSentenceDetail(node) {
  const edgeRows = (state.chapterData?.sentenceRanking || [])
    .filter(link => link.source === node.id || link.target === node.id)
    .sort((a, b) => (a.distance ?? 1) - (b.distance ?? 1))
    .slice(0, 12)
    .map((link, index) => {
      const otherId = otherEnd(link, node.id);
      const other = state.graph.nodes.find(item => item.id === otherId);
      const item = { localRank: index + 1 };
      return `
        <div class="metric-row relation-row ${itemRankClass(item)}" ${relationStyle(item)}><span>${sentenceLabel(otherId)}</span> distance ${formatScore(link.distance)}</div>
        ${other?.text ? `<div class="sentence-snippet ${itemRankClass(item)}" ${relationStyle(item)}>${escapeHtml(other.text)}</div>` : ''}
      `;
    }).join('');
  detailPanel.innerHTML = `
    <div class="detail-title">${node.version} 第 ${node.sentence} 句</div>
    <div class="metric-row"><span>类型</span> sentence</div>
    <div class="metric-row"><span>版本</span> ${node.version}</div>
    <div class="metric-row"><span>章节</span> 第 ${Number(node.chapter) + 1} 章</div>
    <div class="sentence-focus">${escapeHtml(node.text || '')}</div>
    <div class="section-title" style="margin-top:12px">最近句子</div>
    ${edgeRows || '<div class="empty-state">没有句子相似边</div>'}
  `;
}

function renderSentences(chapterData) {
  const communities = chapterData.communities || {};
  sentencesPanel.innerHTML = `
    <div class="section-title">第 ${Number(chapterData.chapter) + 1} 章文本</div>
    <div class="sentence-columns">
      ${Object.entries(chapterData.sentences).map(([version, rows]) => {
        const firstRawSentence = rows[0]?.sentence;
        const firstDisplaySentence = Number(firstRawSentence) + 1;
        const rawNote = Number(firstRawSentence) > 0
          ? `<small title="该版本本章前 ${firstRawSentence} 个句位在原始数据中缺失">从第 ${firstDisplaySentence} 句起</small>`
          : '';
        return `
          <div class="version-sentences">
            <h3><span>${version} · 社区 ${communities[version] || '-'}</span>${rawNote}</h3>
            <ol start="${Number.isFinite(firstDisplaySentence) ? firstDisplaySentence : 1}">
              ${rows.map(row => `<li title="原始句号：${row.sentence}">${escapeHtml(row.text)}</li>`).join('')}
            </ol>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

async function loadVersionView() {
  state.mode = 'version';
  stopLayoutAnimation();
  state.chapterData = null;
  viewTitle.textContent = '版本社区图';
  viewSubtitle.textContent = '边宽表示版本相似度，颜色表示 Girvan-Newman 社区。';
  chapterControls.style.display = 'none';
  sentenceSearch.classList.add('hidden');
  showDetailAside();
  sentencesPanel.classList.remove('hidden');
  const data = await getJson('/api/version-graph?threshold=0.6&top_k=3');
  renderLegend(4);
  renderGraph(data);
}

async function loadChapterView(chapter = state.chapter) {
  state.mode = 'chapter';
  state.chapter = Number(chapter);
  chapterInput.value = state.chapter + 1;
  chapterSelect.value = String(state.chapter);
  viewTitle.textContent = `第 ${state.chapter + 1} 章句子距离图`;
  viewSubtitle.textContent = '包含版本节点、句子节点、句子-句子 normalized_distance 边；点击句子可查看最近句子。';
  chapterControls.style.display = 'block';
  sentenceSearch.classList.remove('hidden');
  showDetailAside();
  sentencesPanel.classList.remove('hidden');
  const data = await getJson(`/api/chapter/${state.chapter}`);
  state.chapterData = data;
  renderLegend(7);
  renderGraph(data);
  renderSentences(data);
}

async function loadMatrixView() {
  state.mode = 'matrix';
  stopLayoutAnimation();
  legend.innerHTML = '';
  viewTitle.textContent = '书籍级亲缘矩阵';
  viewSubtitle.textContent = '由章节/句子相似边聚合得到，颜色越亮表示越接近。';
  chapterControls.style.display = 'none';
  sentenceSearch.classList.add('hidden');
  removeDetailAside();
  svg.classList.add('hidden');
  heatmap.classList.remove('hidden');
  svg.style.display = 'none';
  heatmap.style.display = 'block';
  state.selected = null;
  state.fixed = null;
  const data = await getJson('/api/book-affinity');
  state.bookAffinity = data;
  renderHeatmap(data);
  updateContextRanking();
}

function renderHeatmap(data) {
  const versions = data.versions;
  const matrix = data.matrix;
  const sources = data.sources || [];
  heatmap.innerHTML = `
    <table>
      <thead><tr><th></th>${versions.map(version => `<th>${version}</th>`).join('')}</tr></thead>
      <tbody>
        ${versions.map((version, i) => `
          <tr>
            <th>${version}</th>
            ${versions.map((_, j) => {
              const value = matrix[i][j];
              if (isMissingScore(value)) {
                return '<td class="missing-affinity" title="原始句子相似数据中没有这一版本对的证据">NA</td>';
              }
              const source = sources[i]?.[j];
              const sourceClass = source === 'no_shared_chapter' ? ' class="no-shared-chapter-affinity"' : '';
              const title = source === 'no_shared_chapter'
                ? '两个版本没有共同章节，因此没有直接句子级亲缘证据，书籍级直接聚合值记为 0'
                : '由原始句子相似边聚合';
              const color = `rgba(86, 194, 255, ${Math.max(0.08, value)})`;
              return `<td${sourceClass} title="${title}" style="background:${color}; color:${value > 0.72 ? '#061018' : '#dbeafe'}">${formatScore(value)}</td>`;
            }).join('')}
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
}

function graphPointFromClient(clientX, clientY) {
  const rect = svg.getBoundingClientRect();
  return {
    x: (clientX - rect.left - state.panX) / state.zoom,
    y: (clientY - rect.top - state.panY) / state.zoom,
  };
}

function nodeHitRadius(node) {
  if (node.type === 'version') return 13;
  if (node.type === 'chapter') return 11;
  return state.fixed ? 9 : 7;
}

function findNodeAtClientPoint(clientX, clientY) {
  const point = graphPointFromClient(clientX, clientY);
  let best = null;
  let bestDist = Infinity;
  for (const node of state.graph.nodes) {
    const dx = node.x - point.x;
    const dy = node.y - point.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const radius = nodeHitRadius(node);
    if (dist <= radius && dist < bestDist) {
      best = node;
      bestDist = dist;
    }
  }
  return best;
}

function toggleNodeFocus(node) {
  state.fixed = state.fixed?.id === node.id ? null : node;
  state.selected = null;
  if (state.fixed) {
    showNodeDetail(state.fixed);
  } else {
    detailPanel.innerHTML = '<div class="empty-state">点击节点查看详情并高亮关联</div>';
  }
  drawGraph();
  if (state.mode === 'chapter') startChapterAnimation();
  if (state.mode === 'version') startVersionAnimation();
}

function centerGraphOnNode(node) {
  const rect = svg.getBoundingClientRect();
  const width = rect.width || 900;
  const height = rect.height || 700;
  state.panX = width / 2 - node.x * state.zoom;
  state.panY = height / 2 - node.y * state.zoom;
}

function focusNode(node, options = {}) {
  state.fixed = node;
  state.selected = null;
  if (options.center) centerGraphOnNode(node);
  showNodeDetail(node);
  drawGraph();
  if (state.mode === 'chapter') startChapterAnimation();
  if (state.mode === 'version') startVersionAnimation();
}

async function searchAndFocusSentence() {
  const query = sentenceSearchInput.value.trim();
  if (!query) {
    sentenceSearchStatus.textContent = '请输入句子文本';
    return;
  }
  sentenceSearchStatus.textContent = '搜索中...';
  const result = await searchSentences(query, 12);
  const best = result.items?.[0];
  if (!best) {
    sentenceSearchStatus.textContent = '未找到匹配句子';
    return;
  }
  if (state.mode !== 'chapter' || state.chapter !== Number(best.chapter)) {
    await loadChapterView(Number(best.chapter));
  }
  const node = state.graph.nodes.find(item => item.id === best.nodeId);
  if (!node) {
    sentenceSearchStatus.textContent = '已找到结果，但节点不在当前图中';
    return;
  }
  focusNode(node, { center: true });
  sentenceSearchStatus.textContent =
    `第 ${best.chapterDisplay} 章 · ${best.version} · 第 ${best.sentenceDisplay} 句 · ${best.matchType}`;
}

async function setup() {
  state.overview = await getJson('/api/overview');
  const ranking = await getJson('/api/version-ranking?limit=66');
  state.ranking = ranking.items;
  renderStats();
  updateContextRanking();
  const chapters = await getJson('/api/chapters');
  chapterSelect.innerHTML = chapters.chapters.map(item =>
    `<option value="${item.chapter}">第 ${Number(item.chapter) + 1} 章 · ${item.edgeCount} 边</option>`,
  ).join('');
  await loadVersionView();
}

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', async () => {
    document.querySelectorAll('.tab').forEach(item => item.classList.remove('active'));
    tab.classList.add('active');
    const mode = tab.dataset.mode;
    if (mode === 'version') await loadVersionView();
    if (mode === 'chapter') await loadChapterView(Math.max(0, Number(chapterInput.value || INITIAL_CHAPTER_DISPLAY) - 1));
    if (mode === 'matrix') await loadMatrixView();
  });
});

document.getElementById('loadChapterBtn').addEventListener('click', () => {
  loadChapterView(Math.max(0, Number(chapterInput.value || INITIAL_CHAPTER_DISPLAY) - 1));
});

chapterSelect.addEventListener('change', () => {
  loadChapterView(Number(chapterSelect.value));
});

sentenceSearchBtn.addEventListener('click', () => {
  searchAndFocusSentence().catch(err => {
    sentenceSearchStatus.textContent = err.message;
  });
});

sentenceSearchInput.addEventListener('keydown', event => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  searchAndFocusSentence().catch(err => {
    sentenceSearchStatus.textContent = err.message;
  });
});

svg.addEventListener('wheel', event => {
  event.preventDefault();
  state.zoom = Math.max(0.35, Math.min(2.5, state.zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
  drawGraph();
}, { passive: false });

let drag = null;
svg.addEventListener('pointerdown', event => {
  if (state.mode === 'matrix') return;
  event.preventDefault();
  if (document.activeElement && document.activeElement !== document.body) {
    document.activeElement.blur();
  }
  drag = {
    x: event.clientX,
    y: event.clientY,
    panX: state.panX,
    panY: state.panY,
    targetIsBlank: event.target === svg,
  };
});

window.addEventListener('pointermove', event => {
  if (!drag) return;
  state.panX = drag.panX + event.clientX - drag.x;
  state.panY = drag.panY + event.clientY - drag.y;
  drawGraph();
});

window.addEventListener('pointerup', event => {
  if (drag && state.mode !== 'matrix') {
    const moved = Math.hypot(event.clientX - drag.x, event.clientY - drag.y);
    if (moved <= BLANK_CLICK_DRIFT) {
      const node = findNodeAtClientPoint(event.clientX, event.clientY);
      if (node) {
        toggleNodeFocus(node);
      } else {
        clearHighlight();
        detailPanel.innerHTML = '<div class="empty-state">点击节点查看详情并高亮关联</div>';
      }
    }
  }
  drag = null;
});

setup().catch(err => {
  detailPanel.innerHTML = `<div class="empty-state">${escapeHtml(err.message)}</div>`;
});
