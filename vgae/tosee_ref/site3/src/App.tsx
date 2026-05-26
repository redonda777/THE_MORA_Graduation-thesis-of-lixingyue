import { useRef, useEffect, useState, useCallback } from 'react'
import moraData from '../../docs/data/mora_v4.1_0406.json'
import editDistanceData1 from '../../docs/data/formal_0-1000_sentence_edit_distance_llm_v2.json'
import editDistanceData2 from '../../docs/data/formal_1000-4000_sentence_edit_distance_llm_v2.json'

// ─── Types ───────────────────────────────────────────────────────────────────

interface GraphNode {
  id: number
  type: 'version' | 'chapter' | 'sentence'
  version?: string
  chapter_number?: number
  sentence_number?: number
  text?: string
  treeChildren: number[]
  editConnections: { nodeId: number; editDistance: number }[]
  x: number; y: number; vx: number; vy: number
  layer: number
  radius: number
  color: string
}

interface GraphState {
  nodes: GraphNode[]
  width: number
  height: number
  scale: number
  offsetX: number
  offsetY: number
}

// ─── Data Loading ─────────────────────────────────────────────────────────────

function loadGraphData() {
  type MoraNode = { type?: string; name?: string; children?: MoraNode[]; chapter_number?: number; sentence_number?: number; version?: string; text?: string; index?: number }
  type EdgeRec = { chapter_number: number; sentence_number: number; original_text_version: string; modified_text_version: string; edit_distance: number }

  const versions: Map<string, GraphNode> = new Map()
  const chapters: Map<string, GraphNode> = new Map()
  const sentences: GraphNode[] = []

  function walk(node: MoraNode) {
    if (node.type === 'version') {
      versions.set(node.name!, {
        id: -1, type: 'version', version: node.name, layer: 0,
        treeChildren: [], editConnections: [],
        x: 0, y: 0, vx: 0, vy: 0, radius: 6, color: '#a78bfa'
      })
    } else if (node.type === 'chapter') {
      const first = node.children?.find(c => c.type === 'sentence')
      const version = first?.version ?? 'unknown'
      chapters.set(`${version}-${node.chapter_number}`, {
        id: -1, type: 'chapter', version, chapter_number: node.chapter_number, layer: 1,
        treeChildren: [], editConnections: [],
        x: 0, y: 0, vx: 0, vy: 0, radius: 4, color: '#60a5fa'
      })
    } else if (node.type === 'sentence') {
      sentences.push({
        id: -1, type: 'sentence',
        version: node.version!, chapter_number: node.chapter_number!,
        sentence_number: node.sentence_number!,
        text: (node.text ?? '').replaceAll('#', '□'),
        layer: 2, treeChildren: [], editConnections: [],
        x: 0, y: 0, vx: 0, vy: 0, radius: 2, color: '#3b82f6'
      })
    }
    node.children?.forEach(walk)
  }
  walk(moraData as MoraNode)

  // Edit-distance edges
  const allSentences = [...sentences]
  const allEdgeFiles = [editDistanceData1, editDistanceData2] as EdgeRec[][]
  allEdgeFiles.forEach(edgeData => {
    edgeData.forEach(rec => {
      const idA = allSentences.findIndex(s => s.version === rec.original_text_version && s.chapter_number === rec.chapter_number && s.sentence_number === rec.sentence_number)
      const idB = allSentences.findIndex(s => s.version === rec.modified_text_version && s.chapter_number === rec.chapter_number && s.sentence_number === rec.sentence_number)
      if (idA >= 0 && idB >= 0 && idA !== idB) {
        if (!allSentences[idA].editConnections.some(c => c.nodeId === idB))
          allSentences[idA].editConnections.push({ nodeId: idB, editDistance: rec.edit_distance })
        if (!allSentences[idB].editConnections.some(c => c.nodeId === idA))
          allSentences[idB].editConnections.push({ nodeId: idA, editDistance: rec.edit_distance })
      }
    })
  })

  const connectedSentences = allSentences.filter(s => s.editConnections.length > 0)
  const relevantChapterKeys = new Set(connectedSentences.map(s => `${s.version}-${s.chapter_number}`))
  const relevantVersionNames = new Set(connectedSentences.map(s => s.version!))

  const finalNodes: GraphNode[] = []
  const versionIndexMap = new Map<string, number>()
  relevantVersionNames.forEach(v => {
    const node = versions.get(v)!; node.id = finalNodes.length
    finalNodes.push(node); versionIndexMap.set(v, node.id)
  })
  const chapterIndexMap = new Map<string, number>()
  relevantChapterKeys.forEach(key => {
    const node = chapters.get(key)!; node.id = finalNodes.length
    finalNodes.push(node); chapterIndexMap.set(key, node.id)
  })
  const sentenceIndexMap = new Map<number, number>()
  connectedSentences.forEach(s => {
    s.id = finalNodes.length; finalNodes.push(s)
    sentenceIndexMap.set(allSentences.indexOf(s), s.id)
  })

  relevantChapterKeys.forEach(key => {
    const [version] = key.split('-')
    const versionId = versionIndexMap.get(version), chapterId = chapterIndexMap.get(key)
    if (versionId !== undefined && chapterId !== undefined)
      finalNodes[versionId].treeChildren.push(chapterId)
  })
  connectedSentences.forEach(s => {
    const key = `${s.version}-${s.chapter_number}`
    const chapterId = chapterIndexMap.get(key), sentenceId = sentenceIndexMap.get(allSentences.indexOf(s))
    if (chapterId !== undefined && sentenceId !== undefined)
      finalNodes[chapterId].treeChildren.push(sentenceId)
  })
  connectedSentences.forEach(s => {
    s.editConnections = s.editConnections
      .map(c => ({ nodeId: sentenceIndexMap.get(c.nodeId) ?? -1, editDistance: c.editDistance }))
      .filter(c => c.nodeId !== -1)
  })

  return finalNodes
}

// ─── Initial Layout ──────────────────────────────────────────────────────────

function computeInitialPositions(nodes: GraphNode[], width: number, height: number) {
  const versionNodes = nodes.filter(n => n.type === 'version')
  const chapterNodes = nodes.filter(n => n.type === 'chapter')
  const sentenceNodes = nodes.filter(n => n.type === 'sentence')
  const vCount = versionNodes.length || 1
  const vWidth = width / vCount

  versionNodes.forEach((v, vi) => {
    v.x = vWidth * vi + vWidth / 2; v.y = height * 0.12; v.vx = 0; v.vy = 0
  })

  const chaptersByVersion = new Map<string, GraphNode[]>()
  chapterNodes.forEach(c => {
    const list = chaptersByVersion.get(c.version!) || []; list.push(c); chaptersByVersion.set(c.version!, list)
  })
  chaptersByVersion.forEach((chapters, vName) => {
    const vNode = versionNodes.find(n => n.version === vName)
    if (!vNode) return
    chapters.forEach((c, ci) => {
      const spacing = Math.min(60, (height * 0.6) / (chapters.length + 1))
      c.x = vNode.x + (Math.random() - 0.5) * 80
      c.y = height * 0.25 + ci * spacing; c.vx = 0; c.vy = 0
    })
  })

  const sentencesByChapter = new Map<string, GraphNode[]>()
  sentenceNodes.forEach(s => {
    const key = `${s.version}-${s.chapter_number}`
    const list = sentencesByChapter.get(key) || []; list.push(s); sentencesByChapter.set(key, list)
  })
  sentencesByChapter.forEach((sents, chKey) => {
    const chNode = chapterNodes.find(n => `${n.version}-${n.chapter_number}` === chKey)
    if (!chNode) return
    const cols = Math.ceil(Math.sqrt(sents.length))
    sents.forEach((s, si) => {
      const col = si % cols, row = Math.floor(si / cols)
      s.x = chNode.x + (col - cols / 2) * 30
      s.y = chNode.y + 30 + row * 25
      s.vx = (Math.random() - 0.5) * 0.3; s.vy = (Math.random() - 0.5) * 0.3
    })
  })
}

// ─── Physics Simulation ──────────────────────────────────────────────────────

const REPULSION = 80
const SPRING_TREE = 0.06
const SPRING_EDIT = 0.025
const DAMPING = 0.88
const MAX_SPEED = 6

function simulate(state: GraphState) {
  const { nodes, width, height } = state
  const n = nodes.length

  const forces = nodes.map(() => ({ fx: 0, fy: 0 }))

  // Pairwise repulsion
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y
      const distSq = dx * dx + dy * dy || 0.01
      const dist = Math.sqrt(distSq)
      const f = REPULSION / distSq
      forces[i].fx += (dx / dist) * f; forces[i].fy += (dy / dist) * f
      forces[j].fx -= (dx / dist) * f; forces[j].fy -= (dy / dist) * f
    }
  }

  // Tree springs
  for (let i = 0; i < n; i++) {
    for (const childId of nodes[i].treeChildren) {
      if (childId < 0 || childId >= n) continue
      const child = nodes[childId]
      const dx = child.x - nodes[i].x, dy = child.y - nodes[i].y
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.5
      const ideal = nodes[i].type === 'version' ? 100 : 35
      const f = (dist - ideal) * SPRING_TREE
      forces[i].fx += (dx / dist) * f; forces[i].fy += (dy / dist) * f
      forces[childId].fx -= (dx / dist) * f; forces[childId].fy -= (dy / dist) * f
    }
  }

  // Edit-distance springs
  for (let i = 0; i < n; i++) {
    for (const conn of nodes[i].editConnections) {
      if (conn.nodeId < 0 || conn.nodeId >= n) continue
      const other = nodes[conn.nodeId]
      const dx = other.x - nodes[i].x, dy = other.y - nodes[i].y
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.5
      const f = (dist - 60) * SPRING_EDIT
      forces[i].fx += (dx / dist) * f; forces[i].fy += (dy / dist) * f
      forces[conn.nodeId].fx -= (dx / dist) * f; forces[conn.nodeId].fy -= (dy / dist) * f
    }
  }

  // Apply
  for (let i = 0; i < n; i++) {
    nodes[i].vx = (nodes[i].vx + forces[i].fx) * DAMPING
    nodes[i].vy = (nodes[i].vy + forces[i].fy) * DAMPING
    const speed = Math.sqrt(nodes[i].vx ** 2 + nodes[i].vy ** 2)
    if (speed > MAX_SPEED) { nodes[i].vx = (nodes[i].vx / speed) * MAX_SPEED; nodes[i].vy = (nodes[i].vy / speed) * MAX_SPEED }
    nodes[i].x += nodes[i].vx; nodes[i].y += nodes[i].vy
    if (nodes[i].x < 10) { nodes[i].x = 10; nodes[i].vx *= -0.3 }
    if (nodes[i].x > width - 10) { nodes[i].x = width - 10; nodes[i].vx *= -0.3 }
    if (nodes[i].y < 10) { nodes[i].y = 10; nodes[i].vy *= -0.3 }
    if (nodes[i].y > height - 10) { nodes[i].y = height - 10; nodes[i].vy *= -0.3 }
  }
}

// ─── Constants ────────────────────────────────────────────────────────────────

const CANVAS_COLOR = '#0a0a0f'
const EDGE_COLOR = 'rgba(59, 130, 246, 0.10)'
const TREE_EDGE_COLOR = 'rgba(139, 92, 246, 0.15)'
const EDGE_HIGHLIGHT_COLOR = 'rgba(234, 179, 8, 0.45)'

const BRIGHTNESS_COLORS = [
  '#ffffff', '#fef08a', '#fde68a', '#fcd34d',
  '#fbbf24', '#f59e0b', '#d97706', '#b45309', '#92400e',
]

// ─── Component ───────────────────────────────────────────────────────────────

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stateRef = useRef<GraphState | null>(null)
  const hoveredNodeRef = useRef<number | null>(null)
  const focusedNodeRef = useRef<number | null>(null)
  const animationRef = useRef<number>(0)

  const [stats, setStats] = useState({ nodes: 0, edges: 0, fps: 0 })
  const [focusedNode, setFocusedNode] = useState<GraphNode | null>(null)
  const [graphReady] = useState(() => loadGraphData())
  const [brightnessMap, setBrightnessMap] = useState<Map<number, number>>(new Map())

  // ── Draw ────────────────────────────────────────────────────────────────

  const draw = useCallback((ctx: CanvasRenderingContext2D, state: GraphState) => {
    const { nodes, width, height, scale, offsetX, offsetY } = state
    ctx.fillStyle = CANVAS_COLOR
    ctx.fillRect(0, 0, width, height)
    ctx.save()
    ctx.translate(offsetX, offsetY)
    ctx.scale(scale, scale)

    const focused = focusedNodeRef.current
    const hovered = hoveredNodeRef.current
    const bmap = brightnessMap
    const highlightedIds = new Set<number>()
    if (focused !== null) {
      highlightedIds.add(focused)
      bmap.forEach((_, id) => highlightedIds.add(id))
    }

    // Tree edges
    nodes.forEach(node => {
      if (node.type === 'version' || node.type === 'chapter') {
        node.treeChildren.forEach(childId => {
          const child = nodes[childId]
          if (!child) return
          const isHl = highlightedIds.has(node.id) && highlightedIds.has(childId)
          ctx.beginPath()
          ctx.strokeStyle = isHl ? 'rgba(167,139,250,0.5)' : TREE_EDGE_COLOR
          ctx.lineWidth = isHl ? 1.0 : 0.5
          ctx.moveTo(node.x, node.y); ctx.lineTo(child.x, child.y); ctx.stroke()
        })
      }
    })

    // Edit-distance edges
    nodes.forEach(node => {
      node.editConnections.forEach(conn => {
        if (conn.nodeId <= node.id) return
        const target = nodes[conn.nodeId]
        if (!target) return
        const isHl = focused !== null && highlightedIds.has(node.id) && highlightedIds.has(conn.nodeId)
        ctx.beginPath()
        ctx.strokeStyle = isHl ? EDGE_HIGHLIGHT_COLOR : EDGE_COLOR
        ctx.lineWidth = isHl ? 1.2 : 0.4
        ctx.moveTo(node.x, node.y); ctx.lineTo(target.x, target.y); ctx.stroke()
      })
    })

    // Nodes
    nodes.forEach(node => {
      const isHovered = hovered === node.id
      const isFocused = focused === node.id
      const brightness = bmap.get(node.id) ?? -1
      let radius = node.radius, color = node.color
      if (isFocused) { radius = node.type === 'sentence' ? 9 : 12; color = BRIGHTNESS_COLORS[0] }
      else if (brightness >= 0 && brightness < BRIGHTNESS_COLORS.length) { radius = node.type === 'sentence' ? 5 : 7; color = BRIGHTNESS_COLORS[brightness] }
      else if (isHovered) { radius *= 1.8; color = '#94a3b8' }
      ctx.beginPath()
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2)
      ctx.fillStyle = color; ctx.fill()
    })

    ctx.restore()
  }, [brightnessMap])

  // ── Hit test ───────────────────────────────────────────────────────────

  const findNodeAtPosition = useCallback((state: GraphState, clientX: number, clientY: number): number | null => {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    const x = (clientX - rect.left - state.offsetX) / state.scale
    const y = (clientY - rect.top - state.offsetY) / state.scale
    for (const node of state.nodes) {
      const dx = node.x - x, dy = node.y - y
      if (dx * dx + dy * dy < (node.radius + 8) ** 2) return node.id
    }
    return null
  }, [])

  // ── Click ──────────────────────────────────────────────────────────────

  const handleClick = useCallback((state: GraphState, clientX: number, clientY: number) => {
    const nodeId = findNodeAtPosition(state, clientX, clientY)
    if (nodeId !== null) {
      focusedNodeRef.current = focusedNodeRef.current === nodeId ? null : nodeId
      const focused = focusedNodeRef.current
      if (focused !== null) {
        const node = state.nodes[focused]
        setFocusedNode(node)
        const sorted = [...node.editConnections].sort((a, b) => a.editDistance - b.editDistance)
        const newMap = new Map<number, number>()
        sorted.forEach((conn, idx) => { if (idx + 1 < BRIGHTNESS_COLORS.length) newMap.set(conn.nodeId, idx + 1) })
        setBrightnessMap(newMap)
      } else {
        setFocusedNode(null); setBrightnessMap(new Map())
      }
    }
  }, [findNodeAtPosition])

  // ── Lifecycle ─────────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const nodes = graphReady

    const resize = () => {
      canvas.width = window.innerWidth; canvas.height = window.innerHeight - 60
      if (!stateRef.current) {
        computeInitialPositions(nodes, canvas.width, canvas.height)
        stateRef.current = { nodes, width: canvas.width, height: canvas.height, scale: 1, offsetX: 0, offsetY: 0 }
      } else {
        stateRef.current.width = canvas.width; stateRef.current.height = canvas.height
        computeInitialPositions(nodes, canvas.width, canvas.height)
      }
    }
    resize()
    window.addEventListener('resize', resize)

    let lastTime = performance.now()
    let frameCount = 0, fps = 0

    const animate = (time: number) => {
      frameCount++
      if (time - lastTime >= 1000) {
        fps = frameCount; frameCount = 0; lastTime = time
        if (stateRef.current) {
          let edgeCount = 0
          stateRef.current.nodes.forEach(n => { edgeCount += n.editConnections.length })
          setStats({ nodes: stateRef.current.nodes.length, edges: Math.floor(edgeCount / 2), fps })
        }
      }
      if (stateRef.current) { simulate(stateRef.current); draw(ctx, stateRef.current) }
      animationRef.current = requestAnimationFrame(animate)
    }
    animationRef.current = requestAnimationFrame(animate)

    const onMouseMove = (e: MouseEvent) => { if (stateRef.current) hoveredNodeRef.current = findNodeAtPosition(stateRef.current, e.clientX, e.clientY) }
    const onClick = (e: MouseEvent) => { if (stateRef.current) handleClick(stateRef.current, e.clientX, e.clientY) }
    const onWheel = (e: WheelEvent) => {
      if (stateRef.current) {
        e.preventDefault()
        const delta = e.deltaY > 0 ? 0.9 : 1.1
        stateRef.current.scale = Math.max(0.1, Math.min(5, stateRef.current.scale * delta))
      }
    }
    const onMouseDown = (e: MouseEvent) => {
      if (e.button === 1 && stateRef.current) {
        stateRef.current.offsetX = e.clientX; stateRef.current.offsetY = e.clientY
        const startX = e.clientX, startY = e.clientY
        const onMove = (me: MouseEvent) => { if (stateRef.current) { stateRef.current.offsetX += me.clientX - startX; stateRef.current.offsetY += me.clientY - startY } }
        const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
        window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp)
      }
    }

    canvas.addEventListener('mousemove', onMouseMove)
    canvas.addEventListener('click', onClick)
    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('mousedown', onMouseDown)

    return () => {
      window.removeEventListener('resize', resize)
      cancelAnimationFrame(animationRef.current)
      canvas.removeEventListener('mousemove', onMouseMove)
      canvas.removeEventListener('click', onClick)
      canvas.removeEventListener('wheel', onWheel)
      canvas.removeEventListener('mousedown', onMouseDown)
    }
  }, [graphReady, draw, findNodeAtPosition, handleClick])

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden', background: CANVAS_COLOR }}>
      <div style={{
        position: 'fixed', top: 16, left: 16, zIndex: 10,
        color: '#e5e7eb', fontFamily: 'JetBrains Mono, monospace', fontSize: 12,
        background: 'rgba(0,0,0,0.75)', padding: '12px 16px', borderRadius: 8,
        border: '1px solid rgba(255,255,255,0.1)'
      }}>
        <div style={{ marginBottom: 8, fontSize: 14, fontWeight: 600, color: '#22c55e' }}>
          树结构骨架 · 版本章节 · 按编辑距离排序亮度
        </div>
        <div>节点: <span style={{ color: '#3b82f6' }}>{stats.nodes.toLocaleString()}</span></div>
        <div>边: <span style={{ color: '#3b82f6' }}>{stats.edges.toLocaleString()}</span></div>
        <div>FPS: <span style={{ color: stats.fps < 30 ? '#ef4444' : '#22c55e' }}>{stats.fps}</span></div>
        <div style={{ marginTop: 8, fontSize: 10, color: '#9ca3af', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 8 }}>
          版本(紫) → 章节(蓝) → 句子(亮蓝) | 点击句子聚焦
        </div>
      </div>

      {focusedNode !== null && (
        <div style={{
          position: 'fixed', top: 16, right: 16, zIndex: 10,
          color: '#e5e7eb', fontFamily: 'JetBrains Mono, monospace', fontSize: 12,
          background: 'rgba(0,0,0,0.88)', padding: '12px 16px', borderRadius: 8,
          border: '1px solid #22c55e', minWidth: 220, maxWidth: 360
        }}>
          <div style={{ color: '#22c55e', fontWeight: 600, marginBottom: 8 }}>
            #{focusedNode.id} ·{' '}
            {focusedNode.type === 'version' && `版本 · ${focusedNode.version}`}
            {focusedNode.type === 'chapter' && `章节 · ${focusedNode.version} Ch${focusedNode.chapter_number}`}
            {focusedNode.type === 'sentence' && `Ch${focusedNode.chapter_number}·S${focusedNode.sentence_number} · ${focusedNode.version}`}
          </div>
          {focusedNode.type === 'sentence' && (
            <div style={{ color: '#fef08a', fontSize: 13, marginBottom: 8, lineHeight: 1.6 }}>「{focusedNode.text}」</div>
          )}
          {focusedNode.type === 'version' && (
            <>
              <div style={{ color: '#9ca3af', marginBottom: 6, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 8 }}>
                包含章节（{focusedNode.treeChildren.length} 个）:
              </div>
              <div style={{ maxHeight: 220, overflowY: 'auto' }}>
                {focusedNode.treeChildren.map(childId => {
                  const ch = graphReady[childId]
                  if (!ch) return null
                  return (
                    <div key={childId} style={{ display: 'flex', gap: 6, marginBottom: 3, alignItems: 'baseline' }}>
                      <span style={{ color: '#60a5fa', fontSize: 10 }}>●</span>
                      <span style={{ color: '#e5e7eb' }}>#{childId}</span>
                      <span style={{ color: '#9ca3af', fontSize: 10 }}>Ch{ch.chapter_number}</span>
                      <span style={{ color: '#6b7280', fontSize: 10 }}>句子 {ch.treeChildren.length} 个</span>
                    </div>
                  )
                })}
              </div>
            </>
          )}
          {focusedNode.type === 'chapter' && (
            <>
              <div style={{ color: '#9ca3af', marginBottom: 6, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 8 }}>
                包含句子（{focusedNode.treeChildren.length} 个）:
              </div>
              <div style={{ maxHeight: 220, overflowY: 'auto' }}>
                {focusedNode.treeChildren.map(childId => {
                  const s = graphReady[childId]
                  if (!s) return null
                  return (
                    <div key={childId} style={{ display: 'flex', gap: 6, marginBottom: 3, alignItems: 'baseline', flexWrap: 'wrap' }}>
                      <span style={{ color: '#3b82f6', fontSize: 10 }}>●</span>
                      <span style={{ color: '#e5e7eb' }}>#{childId}</span>
                      <span style={{ color: '#9ca3af', fontSize: 10 }}>S{s.sentence_number}</span>
                      <span style={{ color: '#4b5563', fontSize: 10, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 160 }}>「{s.text}」</span>
                    </div>
                  )
                })}
              </div>
            </>
          )}
          {focusedNode.type === 'sentence' && focusedNode.editConnections.length > 0 && (
            <>
              <div style={{ color: '#9ca3af', marginBottom: 6, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 8 }}>
                相邻句子（按编辑距离排序）:
              </div>
              <div style={{ maxHeight: 220, overflowY: 'auto' }}>
                {focusedNode.editConnections.slice().sort((a, b) => a.editDistance - b.editDistance).map((conn, idx) => {
                  const rank = idx + 1, color = rank < BRIGHTNESS_COLORS.length ? BRIGHTNESS_COLORS[rank] : '#6b7280'
                  const n = graphReady[conn.nodeId]
                  if (!n) return null
                  return (
                    <div key={conn.nodeId} style={{ display: 'flex', gap: 6, marginBottom: 3, alignItems: 'baseline', flexWrap: 'wrap' }}>
                      <span style={{ color, fontSize: 10 }}>●</span>
                      <span style={{ color: '#e5e7eb' }}>#{conn.nodeId}</span>
                      <span style={{ color: '#6b7280', fontSize: 10 }}>Ch{n.chapter_number}·S{n.sentence_number}</span>
                      <span style={{ color: '#9ca3af', fontSize: 10 }}>{n.version}</span>
                      <span style={{ color, fontSize: 10 }}>dist={conn.editDistance}</span>
                      <span style={{ color: '#4b5563', fontSize: 10, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 130 }}>「{n.text}」</span>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </div>
      )}

      <canvas ref={canvasRef} style={{ display: 'block' }} />
    </div>
  )
}
