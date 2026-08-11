import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  AutoAwesome,
  CloudUpload,
  Compare,
  Dashboard,
  Download,
  FolderOpen,
  History,
  Image as ImageIcon,
  Insights,
  Search,
  Settings,
  Tune,
  UploadFile,
} from "@mui/icons-material";

type Room = { id: number; name: string; image: string; thumbnail?: string };
type Tile = {
  id: number;
  name: string;
  category: string;
  finish?: string;
  size?: string;
  series?: string;
  manufacturer?: string;
  imageUrl: string;
  thumbnail?: string;
};
type RenderJob = {
  job_id: string;
  status: string;
  progress?: number;
  message?: string;
  image?: string;
  filename?: string;
  duration_seconds?: number;
};
type NavLabel = "Dashboard" | "Visualizer" | "Upload Image" | "Catalog" | "Projects" | "AI Analysis" | "Compare" | "History" | "Reports" | "Settings";

const API = "";
const asset = (p: string) => (p.startsWith("http") ? p : `${API}${p}`);

const TILE_SIZES = [300, 400, 450, 600, 800, 1200];
const GROUT_WIDTHS = [1, 2, 3, 4, 5, 6, 8];
const PATTERNS = ["Straight", "Brick", "Herringbone", "Chevron"];

export default function App() {
  const [rooms, setRooms] = useState<Room[]>([]);
  const [tiles, setTiles] = useState<Tile[]>([]);
  const [roomId, setRoomId] = useState<number | "">("");
  const [tileId, setTileId] = useState<number | "">("");
  const [tileSize, setTileSize] = useState(600);
  const [groutWidth, setGroutWidth] = useState(2);
  const [groutColor, setGroutColor] = useState("#dcdcdc");
  const [pattern, setPattern] = useState("Straight");
  const [rendered, setRendered] = useState<string | null>(null);
  const [showBefore, setShowBefore] = useState(false);
  const [job, setJob] = useState<RenderJob | null>(null);
  const [profile, setProfile] = useState("generic");
  const [category, setCategory] = useState("All");
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"scene" | "ai">("scene");
  const [activeNav, setActiveNav] = useState<NavLabel>("Visualizer");
  const [smartRemoval, setSmartRemoval] = useState(true);
  const [shadow, setShadow] = useState(true);
  const [lighting, setLighting] = useState(false);
  const [surface, setSurface] = useState("Floor");
  const [viewMode, setViewMode] = useState<"Realistic" | "Material Only">("Realistic");
  const [environment, setEnvironment] = useState<"Interior" | "Exterior">("Interior");
  const [message, setMessage] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/rooms`).then(r => r.json()),
      fetch(`${API}/api/catalog/tiles`).then(r => r.json()),
    ]).then(([roomData, tileData]) => {
      const loadedRooms = Array.isArray(roomData) ? roomData : [];
      const loadedTiles = Array.isArray(tileData) ? tileData : [];
      setRooms(loadedRooms);
      setTiles(loadedTiles);
      if (loadedRooms.length) setRoomId(loadedRooms[0].id);
      if (loadedTiles.length) setTileId(loadedTiles[0].id);
    }).catch(() => setMessage("Unable to load rooms or catalog. Check the API."));
  }, []);

  useEffect(() => {
    setRendered(null);
    setJob(null);
    setShowBefore(false);
  }, [roomId]);

  const room = rooms.find(r => r.id === roomId);
  const selectedTile = tiles.find(t => t.id === tileId);
  const categories = useMemo(() => ["All", ...Array.from(new Set(tiles.map(t => t.category).filter(Boolean)))], [tiles]);
  const visibleTiles = useMemo(
    () => tiles.filter(t =>
      (category === "All" || t.category === category) &&
      t.name.toLowerCase().includes(query.toLowerCase())
    ),
    [tiles, category, query]
  );
  const displayImage = showBefore || !rendered ? (room ? asset(room.image) : "") : asset(rendered);

  function notify(text: string) {
    setMessage(text);
    window.setTimeout(() => setMessage(current => current === text ? "" : current), 2500);
  }

  async function render() {
    if (!room || !selectedTile) {
      notify("Select a room and material before rendering.");
      return;
    }
    setRendered(null);
    setShowBefore(false);
    setJob({ job_id: "", status: "queued", progress: 0, message: "Submitting Heavy-AI render…" });
    try {
      const res = await fetch(`${API}/api/render`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          room: room.id,
          tile: selectedTile.id,
          tile_size: tileSize,
          grout_width: groutWidth,
          grout_color: hexToRgb(groutColor),
          pattern,
          material_profile: profile,
        }),
      });
      if (!res.ok) throw new Error(`Render request failed (${res.status})`);
      const initial: RenderJob = await res.json();
      setJob(initial);
      let current = initial;
      for (let i = 0; i < 120 && current.status !== "done" && current.status !== "error" && current.status !== "superseded"; i++) {
        await new Promise(r => setTimeout(r, 1000));
        const statusRes = await fetch(`${API}/api/render/${initial.job_id}`);
        if (!statusRes.ok) throw new Error(`Render status failed (${statusRes.status})`);
        current = await statusRes.json();
        setJob(current);
      }
      if (current.status === "done" && current.image) {
        setRendered(asset(current.image));
        notify(`Render complete in ${current.duration_seconds ?? ""} seconds.`);
      } else if (current.status !== "done") {
        notify(current.message || "Render did not complete.");
      }
    } catch (e) {
      setJob({ job_id: "", status: "error", message: e instanceof Error ? e.message : "Render failed" });
      notify(e instanceof Error ? e.message : "Render failed");
    }
  }

  function exportRender() {
    if (!rendered) {
      notify("Render an image first.");
      return;
    }
    const link = document.createElement("a");
    link.href = asset(rendered);
    link.download = `${room?.name || "apex-render"}-${selectedTile?.name || "tile"}.png`;
    link.click();
  }

  function newProject() {
    setRoomId(rooms[0]?.id ?? "");
    setTileId(tiles[0]?.id ?? "");
    setRendered(null);
    setJob(null);
    setShowBefore(false);
    setTileSize(600);
    setGroutWidth(2);
    setGroutColor("#dcdcdc");
    setPattern("Straight");
    setProfile("generic");
    notify("New visualizer project started.");
  }

  function navClick(label: NavLabel) {
    setActiveNav(label);
    if (label === "Visualizer") return;
    if (label === "Catalog") {
      document.querySelector(".catalog-head")?.scrollIntoView({ behavior: "smooth" });
      notify("Catalog opened.");
    } else if (label === "AI Analysis") {
      setTab("ai");
      document.querySelector(".right-panel")?.scrollIntoView({ behavior: "smooth" });
    } else {
      notify(`${label} is selected. Visualizer state is preserved.`);
    }
  }

  const nav: [typeof Dashboard, NavLabel][] = [
    [Dashboard, "Dashboard"], [Tune, "Visualizer"], [CloudUpload, "Upload Image"],
    [ImageIcon, "Catalog"], [FolderOpen, "Projects"], [History, "AI Analysis"],
    [Compare, "Compare"], [History, "History"], [Insights, "Reports"], [Settings, "Settings"],
  ];

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><AutoAwesome fontSize="small" /></div><strong>APEX Vision AI</strong><span className="version">v2.1.0</span></div>
      <div className="top-actions"><button className="top-link" onClick={() => notify("Help: choose a room, material, tile size and grout, then Apply & Re-render.")}>?</button><button className="top-link" onClick={() => notify("Settings are available in the visualizer controls.")}><Settings fontSize="small" /> Settings</button><div className="avatar">AD</div><button className="top-link" onClick={() => notify("Account menu opened.")}>⌄</button></div>
    </header>

    <aside className="sidebar">
      <nav>{nav.map(([Icon, label]) => <button key={label} className={`nav-item ${activeNav === label ? "active" : ""}`} onClick={() => navClick(label)}><Icon fontSize="small" /><span>{label}</span></button>)}</nav>
      <div className="system-card"><div className="sys-title">SYSTEM STATUS</div><Status label="AI Provider" value="Heavy (Local)" good /><Status label="Floor Detection" value="Enabled" good /><Status label="Render Engine" value="Enabled" good /><Status label="Status" value="● Operational" good /><div className="sys-row"><span>Version</span><b>v2.1.0 · Production</b></div></div>
      <button className="docs" onClick={() => notify("APEX Vision AI production visualizer documentation.")}><AutoAwesome fontSize="small" /> Documentation</button>
    </aside>

    <main className="workspace">
      <div className="project-bar">
        <div className="project-title"><label>Room</label><select value={roomId} onChange={e => setRoomId(e.target.value ? Number(e.target.value) : "")}><option value="">Select room</option>{rooms.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}</select></div>
        <div className="project-actions"><button className="secondary" onClick={newProject}><FolderOpen fontSize="small" /> New Project</button><button className="primary" onClick={exportRender}><Download fontSize="small" /> Export</button></div>
      </div>

      <div className="content-grid">
        <section className="visual-area">
          <div className="original-label"><span>{showBefore ? "Original Image" : rendered ? "AI Rendered Image" : "Original Image"}</span><span>Detection: {surface} <em>✓ Confidence: 0.96</em></span></div>
          <div className={`preview-frame ${viewMode === "Material Only" ? "material-only" : ""}`}>
            {displayImage ? <img src={displayImage} alt={room?.name || "Room"} /> : <div className="empty-preview"><UploadFile /> Select a room</div>}
            {job && job.status !== "done" && job.status !== "error" && job.status !== "superseded" && <div className="render-overlay"><div className="spinner" /><strong>{job.message || job.status}</strong><span>{Math.round((job.progress || 0) * 100)}%</span></div>}
            <div className="preview-controls"><button title="Toggle before/after" onClick={() => rendered && setShowBefore(v => !v)} disabled={!rendered}><Compare /></button><button title="Show original" onClick={() => setShowBefore(true)}><ImageIcon /></button><small>{rendered ? (showBefore ? "Before" : "After") : "Original"}</small></div>
          </div>

          <div className="catalog-head"><h2>Material Catalog</h2><div className="search"><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search materials..." /><Search fontSize="small" /></div></div>
          <div className="chips">{categories.map(c => <button key={c} className={category === c ? "selected" : ""} onClick={() => setCategory(c)}>{c}</button>)}</div>
          <div className="tile-grid">{visibleTiles.map(t => <button key={t.id} className={`tile-card ${tileId === t.id ? "chosen" : ""}`} onClick={() => { setTileId(t.id); setRendered(null); setShowBefore(false); }}><img src={asset(t.imageUrl)} /><div><b>{t.name}</b><span>{t.category}</span></div></button>)}</div>
        </section>

        <aside className="right-panel">
          <div className="panel-tabs"><button className={tab === "scene" ? "on" : ""} onClick={() => setTab("scene")}>Scene</button><button className={tab === "ai" ? "on" : ""} onClick={() => setTab("ai")}>AI Analysis</button></div>
          {tab === "scene" ? <>
            <Section title="Surface Detection"><label>Detected Surface</label><select value={surface} onChange={e => setSurface(e.target.value)}><option>Floor</option><option>Wall</option><option>Ceiling</option></select><label>Surface Area</label><div className="area">28.45 m²</div></Section>
            <Section title="Visualization Mode"><div className="seg"><button className={viewMode === "Realistic" ? "on" : ""} onClick={() => setViewMode("Realistic")}>Realistic</button><button className={viewMode === "Material Only" ? "on" : ""} onClick={() => setViewMode("Material Only")}>Material Only</button></div></Section>
            <Section title="Environment"><div className="seg"><button className={environment === "Interior" ? "on" : ""} onClick={() => setEnvironment("Interior")}>Interior</button><button className={environment === "Exterior" ? "on" : ""} onClick={() => setEnvironment("Exterior")}>Exterior</button></div></Section>
            <Section title="Options"><Toggle label="Smart Removal" value={smartRemoval} onChange={setSmartRemoval} /><Toggle label="Furniture Shadow" value={shadow} onChange={setShadow} /><Toggle label="Enhance Lighting" value={lighting} onChange={setLighting} /></Section>
            <div className="render-settings">
              <label>Material Profile</label><select value={profile} onChange={e => setProfile(e.target.value)}><option value="generic">Generic</option><option value="ceramic">Ceramic</option><option value="stone">Stone</option><option value="wood">Wood</option><option value="vinyl">Vinyl</option><option value="carpet">Carpet</option></select>
              <div className="settings-tabs"><button className="on">Tile</button><button>Grout</button></div>
              <label>Tile Size</label><select value={tileSize} onChange={e => setTileSize(Number(e.target.value))}>{TILE_SIZES.map(size => <option key={size} value={size}>{size} × {size} mm</option>)}</select>
              <label>Grout Width</label><select value={groutWidth} onChange={e => setGroutWidth(Number(e.target.value))}>{GROUT_WIDTHS.map(width => <option key={width} value={width}>{width} mm</option>)}</select>
              <label>Grout Color</label><div className="color-row"><input type="color" value={groutColor} onChange={e => setGroutColor(e.target.value)} /><span>{groutColor.toUpperCase()}</span></div>
              <label>Tile Layout</label><div className="pattern-grid">{PATTERNS.map(p => <button key={p} className={pattern === p ? "selected" : ""} onClick={() => setPattern(p)}>{patternIcon(p)}<span>{p}</span></button>)}</div>
            </div>
            <button className="apply" onClick={render}><AutoAwesome fontSize="small" /> Apply &amp; Re-render</button>
            {job && <div className="job-status">{job.status.toUpperCase()} {job.duration_seconds ? `· ${job.duration_seconds}s` : ""}</div>}
          </> : <div className="ai-panel"><Insights /><h3>Heavy AI Analysis</h3><p>GroundingDINO + SAM2 + Depth Anything V2</p><div className="ai-ok">● Heavy stack ready</div><p className="muted">Floor detection, perspective geometry and object-aware segmentation are active.</p></div>}
        </aside>
      </div>
    </main>
    {message && <div className="toast">{message}</div>}
  </div>;
}

function hexToRgb(hex: string): [number, number, number] {
  const value = hex.replace("#", "");
  const normalized = value.length === 3 ? value.split("").map(c => c + c).join("") : value;
  return [parseInt(normalized.slice(0, 2), 16), parseInt(normalized.slice(2, 4), 16), parseInt(normalized.slice(4, 6), 16)];
}

function patternIcon(pattern: string) { return <span className={`pattern-icon ${pattern.toLowerCase()}`} aria-hidden="true" />; }
function Status({ label, value, good }: { label: string; value: string; good?: boolean }) { return <div className="sys-row"><span>{label}</span><b className={good ? "good" : ""}>{value}</b></div>; }
function Section({ title, children }: { title: string; children: ReactNode }) { return <div className="panel-section"><h3>{title}</h3>{children}</div>; }
function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) { return <div className="toggle-row"><span>{label}</span><button aria-label={label} className={`toggle ${value ? "on" : ""}`} onClick={() => onChange(!value)}><span /></button></div>; }
