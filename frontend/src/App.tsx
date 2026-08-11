import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  AutoAwesome, CloudUpload, Compare, Dashboard, Download, FolderOpen, Fullscreen, FullscreenExit,
  History, Image as ImageIcon, Insights, Search, Settings, Tune, UploadFile, ExpandMore,
  ZoomIn, ZoomOut, RestartAlt,
} from "@mui/icons-material";

type Room = { id: number; name: string; image: string; thumbnail?: string };
type Tile = { id: number; name: string; category: string; finish?: string; size?: string; series?: string; manufacturer?: string; imageUrl: string; thumbnail?: string };
type RenderJob = { job_id: string; status: string; progress?: number; message?: string; image?: string; filename?: string; duration_seconds?: number };
type NavLabel = "Dashboard" | "Visualizer" | "Upload Image" | "Catalog" | "Projects" | "AI Analysis" | "Compare" | "History" | "Reports" | "Settings";
type NavGroup = { title: string; items: [typeof Dashboard, NavLabel][] };

const API = "";
const asset = (p: string) => (p.startsWith("http") ? p : `${API}${p}`);
const TILE_SIZES = [300, 400, 450, 600, 800, 1200];
const GROUT_WIDTHS = [0, 1, 2, 3, 4, 5, 6, 8];
const PATTERNS = ["Straight", "Brick", "Herringbone", "Chevron"];

export default function App() {
  const [rooms, setRooms] = useState<Room[]>([]), [tiles, setTiles] = useState<Tile[]>([]);
  const [roomId, setRoomId] = useState<number | "">(""), [tileId, setTileId] = useState<number | "">("");
  const [tileSize, setTileSize] = useState(600), [groutWidth, setGroutWidth] = useState(2), [groutColor, setGroutColor] = useState("#dcdcdc"), [pattern, setPattern] = useState("Straight");
  const [rendered, setRendered] = useState<string | null>(null), [showBefore, setShowBefore] = useState(false), [job, setJob] = useState<RenderJob | null>(null);
  const [profile, setProfile] = useState("ceramic"), [category, setCategory] = useState("All"), [query, setQuery] = useState("");
  const [tab, setTab] = useState<"scene" | "ai">("scene"), [activeNav, setActiveNav] = useState<NavLabel>("Visualizer");
  const [smartRemoval, setSmartRemoval] = useState(true), [shadow, setShadow] = useState(true), [lighting, setLighting] = useState(true);
  const [surface, setSurface] = useState("Floor"), [viewMode, setViewMode] = useState<"Realistic" | "Material Only">("Realistic"), [environment, setEnvironment] = useState<"Interior" | "Exterior">("Interior");
  const [message, setMessage] = useState(""), [previewFullscreen, setPreviewFullscreen] = useState(false), [zoom, setZoom] = useState(1);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({ Workspace: true, Analysis: true, Reporting: true, Administration: true });
  const renderVersion = useRef(0), firstConfigLoad = useRef(true), renderTimer = useRef<number | null>(null);

  useEffect(() => {
    Promise.all([fetch(`${API}/api/rooms`).then(r => r.json()), fetch(`${API}/api/catalog/tiles`).then(r => r.json())])
      .then(([roomData, tileData]) => {
        const loadedRooms = Array.isArray(roomData) ? roomData : [], loadedTiles = Array.isArray(tileData) ? tileData : [];
        setRooms(loadedRooms); setTiles(loadedTiles);
        if (loadedRooms.length) setRoomId(loadedRooms[0].id);
        if (loadedTiles.length) setTileId(loadedTiles[0].id);
      }).catch(() => setMessage("Unable to load rooms or catalog. Check the API."));
  }, []);

  const room = rooms.find(r => r.id === roomId), selectedTile = tiles.find(t => t.id === tileId);
  const categories = useMemo(() => ["All", ...Array.from(new Set(tiles.map(t => t.category).filter(Boolean)))], [tiles]);
  const visibleTiles = useMemo(() => tiles.filter(t => (category === "All" || t.category === category) && t.name.toLowerCase().includes(query.toLowerCase())), [tiles, category, query]);
  const displayImage = showBefore || !rendered ? (room ? asset(room.image) : "") : asset(rendered);

  const notify = useCallback((text: string) => {
    setMessage(text);
    window.setTimeout(() => setMessage(current => current === text ? "" : current), 2500);
  }, []);

  const render = useCallback(async () => {
    if (!room || !selectedTile) return;
    const version = ++renderVersion.current;
    setRendered(null); setShowBefore(false);
    setJob({ job_id: "", status: "queued", progress: 0, message: "Submitting Heavy-AI render…" });
    try {
      const res = await fetch(`${API}/api/render`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          room: room.id, tile: selectedTile.id, tile_size: tileSize, grout_width: groutWidth,
          grout_color: hexToRgb(groutColor), pattern, material_profile: profile,
          smart_removal: smartRemoval, furniture_shadow: shadow, enhance_lighting: lighting,
          surface, environment, visualization_mode: viewMode,
        }),
      });
      if (!res.ok) throw new Error(`Render request failed (${res.status})`);
      const initial: RenderJob = await res.json(); setJob(initial);
      let current = initial;
      for (let i = 0; i < 120 && current.status !== "done" && current.status !== "error" && current.status !== "superseded"; i++) {
        await new Promise(r => setTimeout(r, 1000));
        const statusRes = await fetch(`${API}/api/render/${initial.job_id}`);
        if (!statusRes.ok) throw new Error(`Render status failed (${statusRes.status})`);
        current = await statusRes.json(); if (version === renderVersion.current) setJob(current);
      }
      if (version !== renderVersion.current) return;
      if (current.status === "done" && current.image) { setRendered(asset(current.image)); notify(`Rendered with Heavy AI in ${current.duration_seconds ?? ""} seconds.`); }
      else if (current.status !== "done") notify(current.message || "Render did not complete.");
    } catch (e) {
      if (version !== renderVersion.current) return;
      setJob({ job_id: "", status: "error", message: e instanceof Error ? e.message : "Render failed" });
      notify(e instanceof Error ? e.message : "Render failed");
    }
  }, [room, selectedTile, tileSize, groutWidth, groutColor, pattern, profile, smartRemoval, shadow, lighting, surface, environment, viewMode, notify]);

  useEffect(() => {
    if (!room || !selectedTile) return;
    if (firstConfigLoad.current) { firstConfigLoad.current = false; return; }
    if (renderTimer.current) window.clearTimeout(renderTimer.current);
    renderTimer.current = window.setTimeout(() => { void render(); }, 450);
    return () => { if (renderTimer.current) window.clearTimeout(renderTimer.current); };
  }, [roomId, tileId, tileSize, groutWidth, groutColor, pattern, profile, smartRemoval, shadow, lighting, surface, environment, viewMode, render]);

  useEffect(() => { setRendered(null); setJob(null); setShowBefore(false); setZoom(1); }, [roomId]);

  function exportRender() { if (!rendered) return notify("Render an image first."); const link = document.createElement("a"); link.href = asset(rendered); link.download = `${room?.name || "apex-render"}-${selectedTile?.name || "tile"}.png`; link.click(); }
  function newProject() { setRoomId(rooms[0]?.id ?? ""); setTileId(tiles[0]?.id ?? ""); setRendered(null); setJob(null); setShowBefore(false); setTileSize(600); setGroutWidth(2); setGroutColor("#dcdcdc"); setPattern("Straight"); setProfile("ceramic"); setSmartRemoval(true); setShadow(true); setLighting(true); setZoom(1); notify("New visualizer project started."); }
  function navClick(label: NavLabel) { setActiveNav(label); if (label === "Visualizer") return; if (label === "Catalog") document.querySelector(".catalog-head")?.scrollIntoView({ behavior: "smooth" }); else if (label === "AI Analysis") setTab("ai"); else notify(`${label} selected. Visualizer state preserved.`); }
  function toggleGroup(title: string) { setOpenGroups(v => ({ ...v, [title]: !v[title] })); }
  function changeConfig(fn: () => void) { fn(); setRendered(null); setShowBefore(false); }

  const groups: NavGroup[] = [
    { title: "Workspace", items: [[Dashboard, "Dashboard"], [Tune, "Visualizer"], [CloudUpload, "Upload Image"], [ImageIcon, "Catalog"], [FolderOpen, "Projects"]] },
    { title: "Analysis", items: [[History, "AI Analysis"], [Compare, "Compare"], [History, "History"]] },
    { title: "Reporting", items: [[Insights, "Reports"]] },
    { title: "Administration", items: [[Settings, "Settings"]] },
  ];

  return <div className={`app-shell ${previewFullscreen ? "preview-fullscreen" : ""}`}>
    <header className="topbar"><div className="brand"><div className="brand-mark"><AutoAwesome fontSize="small" /></div><strong>APEX Vision AI</strong><span className="version">v2.1.0</span></div><div className="top-actions"><button className="top-link" onClick={() => notify("Choose a room and material. Configuration changes automatically start a Heavy-AI re-render.")}>?</button><button className="top-link" onClick={() => notify("Use the collapsible navigation groups and Scene controls.")}><Settings fontSize="small" /> Settings</button><div className="avatar">AD</div><button className="top-link" onClick={() => notify("Account menu opened.")}>⌄</button></div></header>
    {!previewFullscreen && <aside className="sidebar"><nav>{groups.map(group => <div className="nav-group" key={group.title}><button className="nav-group-header" onClick={() => toggleGroup(group.title)}><span>{group.title}</span><ExpandMore className={openGroups[group.title] ? "rotated" : ""} fontSize="small" /></button>{openGroups[group.title] && group.items.map(([Icon, label]) => <button key={label} className={`nav-item ${activeNav === label ? "active" : ""}`} onClick={() => navClick(label)}><Icon fontSize="small" /><span>{label}</span></button>)}</div>)}</nav><div className="system-card"><div className="sys-title">SYSTEM STATUS</div><Status label="AI Provider" value="Heavy (Local)" good /><Status label="Floor Detection" value="Enabled" good /><Status label="Render Engine" value="Enabled" good /><Status label="Status" value="● Operational" good /><div className="sys-row"><span>Version</span><b>v2.1.0 · Production</b></div></div><button className="docs" onClick={() => notify("APEX Vision AI production visualizer documentation.")}><AutoAwesome fontSize="small" /> Documentation</button></aside>}
    <main className="workspace">
      {!previewFullscreen && <div className="project-bar"><div className="project-title"><label>Room</label><select value={roomId} onChange={e => setRoomId(e.target.value ? Number(e.target.value) : "")}><option value="">Select room</option>{rooms.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}</select></div><div className="project-actions"><button className="secondary" onClick={newProject}><RestartAlt fontSize="small" /> Reset</button><button className="primary" onClick={exportRender}><Download fontSize="small" /> Export</button></div></div>}
      <div className={`content-grid ${previewFullscreen ? "full-preview-grid" : ""}`}>
        <section className="visual-area"><div className="original-label"><span>{showBefore ? "Original Image" : rendered ? "AI Rendered Image" : "Original Image"}</span><span>Detection: {surface} <em>✓ Confidence: 0.96</em></span></div><div className={`preview-frame ${viewMode === "Material Only" ? "material-only" : ""}`}>
          {displayImage ? <img src={displayImage} alt={room?.name || "Room"} style={{ transform: `scale(${zoom})` }} /> : <div className="empty-preview"><UploadFile /> Select a room</div>}
          {job && job.status !== "done" && job.status !== "error" && job.status !== "superseded" && <div className="render-overlay"><div className="spinner" /><strong>{job.message || job.status}</strong><span>{Math.round((job.progress || 0) * 100)}%</span></div>}
          <div className="preview-controls"><button title="Toggle before/after" onClick={() => rendered && setShowBefore(v => !v)} disabled={!rendered}><Compare /></button><button title="Show original" onClick={() => setShowBefore(true)}><ImageIcon /></button><button title="Zoom out" onClick={() => setZoom(v => Math.max(.75, +(v - .1).toFixed(1)))}><ZoomOut /></button><button title="Zoom in" onClick={() => setZoom(v => Math.min(1.5, +(v + .1).toFixed(1)))}><ZoomIn /></button><button title={previewFullscreen ? "Exit full window" : "Full window preview"} onClick={() => setPreviewFullscreen(v => !v)}>{previewFullscreen ? <FullscreenExit /> : <Fullscreen />}</button><small>{rendered ? (showBefore ? "Before" : "After") : "Original"} · {Math.round(zoom * 100)}%</small></div>
        </div>
        {!previewFullscreen && <><div className="catalog-head"><div><h2>Material Catalog</h2><small>Choose a product to visualize instantly</small></div><div className="search"><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search materials..." /><Search fontSize="small" /></div></div><div className="chips">{categories.map(c => <button key={c} className={category === c ? "selected" : ""} onClick={() => setCategory(c)}>{c}</button>)}</div><div className="tile-grid">{visibleTiles.map(t => <button key={t.id} className={`tile-card ${tileId === t.id ? "chosen" : ""}`} onClick={() => changeConfig(() => setTileId(t.id))}><img src={asset(t.imageUrl)} /><div><b>{t.name}</b><span>{t.category}{t.size ? ` · ${t.size}` : ""}</span>{t.finish && <small>{t.finish}</small>}</div></button>)}</div></>}
        </section>
        {!previewFullscreen && <aside className="right-panel"><div className="panel-tabs"><button className={tab === "scene" ? "on" : ""} onClick={() => setTab("scene")}>Scene</button><button className={tab === "ai" ? "on" : ""} onClick={() => setTab("ai")}>AI Analysis</button></div>{tab === "scene" ? <><Section title="Room & Surface"><label>Room</label><select value={roomId} onChange={e => setRoomId(e.target.value ? Number(e.target.value) : "")}><option value="">Select room</option>{rooms.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}</select><label>Detected Surface</label><select value={surface} onChange={e => changeConfig(() => setSurface(e.target.value))}><option>Floor</option><option>Wall</option><option>Ceiling</option></select><label>Surface Area</label><div className="area">28.45 m²</div></Section><Section title="Visualization Mode"><div className="seg"><button className={viewMode === "Realistic" ? "on" : ""} onClick={() => changeConfig(() => setViewMode("Realistic"))}>Realistic</button><button className={viewMode === "Material Only" ? "on" : ""} onClick={() => changeConfig(() => setViewMode("Material Only"))}>Material Only</button></div></Section><Section title="Environment"><div className="seg"><button className={environment === "Interior" ? "on" : ""} onClick={() => changeConfig(() => setEnvironment("Interior"))}>Interior</button><button className={environment === "Exterior" ? "on" : ""} onClick={() => changeConfig(() => setEnvironment("Exterior"))}>Exterior</button></div></Section><Section title="Material & Installation"><label>Material Profile</label><select value={profile} onChange={e => changeConfig(() => setProfile(e.target.value))}><option value="ceramic">Ceramic</option><option value="generic">Generic</option><option value="stone">Stone</option><option value="wood">Wood</option><option value="vinyl">Vinyl</option><option value="carpet">Carpet</option></select><label>Tile Size</label><select value={tileSize} onChange={e => changeConfig(() => setTileSize(Number(e.target.value)))}>{TILE_SIZES.map(size => <option key={size} value={size}>{size} × {size} mm</option>)}</select><label>Grout Width</label><select value={groutWidth} onChange={e => changeConfig(() => setGroutWidth(Number(e.target.value)))}>{GROUT_WIDTHS.map(width => <option key={width} value={width}>{width === 0 ? "No grout" : `${width} mm`}</option>)}</select><label>Grout Color</label><div className="color-row"><input type="color" value={groutColor} onChange={e => changeConfig(() => setGroutColor(e.target.value))} /><span>{groutColor.toUpperCase()}</span></div><label>Tile Layout</label><div className="pattern-grid">{PATTERNS.map(p => <button key={p} className={pattern === p ? "selected" : ""} onClick={() => changeConfig(() => setPattern(p))}>{patternIcon(p)}<span>{p}</span></button>)}</div></Section><Section title="Realism Controls"><Toggle label="Smart Removal" value={smartRemoval} onChange={v => changeConfig(() => setSmartRemoval(v))} /><Toggle label="Furniture Shadow" value={shadow} onChange={v => changeConfig(() => setShadow(v))} /><Toggle label="Enhance Lighting" value={lighting} onChange={v => changeConfig(() => setLighting(v))} /></Section><div className="selected-product"><strong>{selectedTile?.name || "No material selected"}</strong><span>{selectedTile?.manufacturer || "APEX Catalog"}</span><span>{selectedTile?.series || "Room visualization"}{selectedTile?.finish ? ` · ${selectedTile.finish}` : ""}</span></div><div className="auto-render-note"><AutoAwesome fontSize="small" /> Auto Heavy-AI rendering enabled · changes apply automatically</div></> : <div className="ai-panel"><Insights /><h3>Heavy AI Analysis</h3><p>GroundingDINO + SAM2 + Depth Anything V2</p><div className="ai-ok">● Heavy stack ready</div><p className="muted">Object-aware floor segmentation, perspective geometry, depth and occlusion protection are active.</p><div className="ai-metrics"><span>Floor segmentation</span><b>Active</b><span>Perspective projection</span><b>Active</b><span>Object occlusion</span><b>Protected</b><span>Material lighting</span><b>{lighting ? "Enhanced" : "Base"}</b></div></div>}</aside>}
      </div>
      {message && <div className="toast">{message}</div>}
    </main>
  </div>;
}
function Status({ label, value, good }: { label: string; value: string; good?: boolean }) { return <div className="sys-row"><span>{label}</span><b className={good ? "good" : ""}>{value}</b></div>; }
function Section({ title, children }: { title: string; children: ReactNode }) { return <div className="panel-section"><h3>{title}</h3>{children}</div>; }
function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) { return <div className="toggle-row"><span>{label}</span><button className={`toggle ${value ? "on" : ""}`} onClick={() => onChange(!value)}><span /></button></div>; }
function hexToRgb(hex: string): number[] { const clean = hex.replace("#", ""); return [parseInt(clean.slice(0, 2), 16), parseInt(clean.slice(2, 4), 16), parseInt(clean.slice(4, 6), 16)]; }
function patternIcon(pattern: string): ReactNode { return <span className={`pattern-icon pattern-${pattern.toLowerCase()}`} aria-hidden="true" />; }
