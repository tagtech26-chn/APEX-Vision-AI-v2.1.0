import { useEffect, useMemo, useState } from "react";
import {
  AutoAwesome,
  CameraAlt,
  CloudUpload,
  Compare,
  Dashboard,
  Download,
  Expand,
  FolderOpen,
  Fullscreen,
  History,
  Home,
  Image as ImageIcon,
  Insights,
  Search,
  Settings,
  Tune,
  UploadFile,
} from "@mui/icons-material";

type Room = { id: number; name: string; image: string; thumbnail?: string };
type Tile = {
  id: number; name: string; category: string; finish?: string; size?: string;
  series?: string; manufacturer?: string; imageUrl: string; thumbnail?: string;
};
type RenderJob = { job_id: string; status: string; progress?: number; message?: string; image?: string; filename?: string; duration_seconds?: number };

const API = "";
const asset = (p: string) => (p.startsWith("http") ? p : `${API}${p}`);

export default function App() {
  const [rooms, setRooms] = useState<Room[]>([]);
  const [tiles, setTiles] = useState<Tile[]>([]);
  const [roomId, setRoomId] = useState(7);
  const [tileId, setTileId] = useState(1);
  const [rendered, setRendered] = useState<string | null>(null);
  const [job, setJob] = useState<RenderJob | null>(null);
  const [profile, setProfile] = useState("generic");
  const [category, setCategory] = useState("All");
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"scene" | "ai">("scene");
  const [smartRemoval, setSmartRemoval] = useState(true);
  const [shadow, setShadow] = useState(true);
  const [lighting, setLighting] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/rooms`).then(r => r.json()),
      fetch(`${API}/api/catalog/tiles`).then(r => r.json()),
    ]).then(([r, t]) => {
      setRooms(Array.isArray(r) ? r : []);
      setTiles(Array.isArray(t) ? t : []);
      if (Array.isArray(r) && r.length) setRoomId((r.find((x: Room) => x.id === 7) || r[0]).id);
      if (Array.isArray(t) && t.length) setTileId(t[0].id);
    }).catch(() => undefined);
  }, []);

  const room = rooms.find(r => r.id === roomId);
  const selectedTile = tiles.find(t => t.id === tileId);
  const categories = useMemo(() => ["All", ...Array.from(new Set(tiles.map(t => t.category).filter(Boolean)))], [tiles]);
  const visibleTiles = useMemo(() => tiles.filter(t => (category === "All" || t.category === category) && t.name.toLowerCase().includes(query.toLowerCase())), [tiles, category, query]);
  const displayImage = rendered || (room ? asset(room.image) : "");

  async function render() {
    if (!room || !selectedTile) return;
    setRendered(null);
    setJob({ job_id: "", status: "queued", progress: 0, message: "Submitting render…" });
    try {
      const res = await fetch(`${API}/api/render`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ room: room.id, tile: selectedTile.id, tile_size: 600, grout_width: 2, grout_color: [220, 220, 220], pattern: "Straight", material_profile: profile }) });
      const initial = await res.json();
      setJob(initial);
      let current = initial;
      for (let i = 0; i < 90 && current.status !== "done" && current.status !== "failed"; i++) {
        await new Promise(r => setTimeout(r, 1000));
        current = await fetch(`${API}/api/render/${initial.job_id}`).then(r => r.json());
        setJob(current);
      }
      if (current.status === "done" && current.image) setRendered(asset(current.image));
    } catch (e) {
      setJob({ job_id: "", status: "failed", message: e instanceof Error ? e.message : "Render failed" });
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><AutoAwesome fontSize="small" /></div><strong>APEX Vision AI</strong><span className="version">v2.1.0</span></div>
        <div className="top-actions"><span>?</span><span>Help</span><Settings fontSize="small" /><span>Settings</span><div className="avatar">AD</div><span>⌄</span></div>
      </header>

      <aside className="sidebar">
        <nav>
          {[[Dashboard,"Dashboard"],[Tune,"Visualizer"],[CloudUpload,"Upload Image"],[ImageIcon,"Catalog"],[FolderOpen,"Projects"],[History,"AI Analysis"],[Compare,"Compare"],[History,"History"],[Insights,"Reports"],[Settings,"Settings"]].map(([Icon, label], i) => {
            const C = Icon as typeof Dashboard;
            return <div key={String(label)} className={`nav-item ${i === 1 ? "active" : ""}`}><C fontSize="small" /><span>{label as string}</span></div>;
          })}
        </nav>
        <div className="system-card">
          <div className="sys-title">SYSTEM STATUS</div>
          <Status label="AI Provider" value="Heavy (Local)" good />
          <Status label="Floor Detection" value="Enabled" good />
          <Status label="Render Engine" value="Enabled" good />
          <Status label="Status" value="● Operational" good />
          <div className="sys-row"><span>Version</span><b>v2.1.0 · Production</b></div>
        </div>
        <div className="docs"><AutoAwesome fontSize="small" /> Documentation</div>
      </aside>

      <main className="workspace">
        <div className="project-bar"><div><h1>Project: {room?.name || "Living Room"}</h1><span className="edit">⌕</span></div><div className="project-actions"><button className="secondary"><FolderOpen fontSize="small" /> New Project</button><button className="primary"><Download fontSize="small" /> Export</button></div></div>
        <div className="content-grid">
          <section className="visual-area">
            <div className="original-label"><span>Original Image</span><span>Detection: Floor <em>✓ Confidence: 0.96</em></span></div>
            <div className="preview-frame">
              {displayImage ? <img src={displayImage} alt={room?.name || "Room"} /> : <div className="empty-preview"><UploadFile /> Select a room</div>}
              {job && job.status !== "done" && <div className="render-overlay"><div className="spinner" /><strong>{job.message || job.status}</strong><span>{Math.round((job.progress || 0) * 100)}%</span></div>}
              <div className="preview-controls"><button><Compare /></button><button><ImageIcon /></button><small>Before / After</small><small>Original</small></div>
            </div>

            <div className="catalog-head"><h2>Material Catalog</h2><div className="search"><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search materials..." /><Search fontSize="small" /></div></div>
            <div className="chips">{categories.map(c => <button key={c} className={category === c ? "selected" : ""} onClick={() => setCategory(c)}>{c}</button>)}</div>
            <div className="tile-grid">
              {visibleTiles.map(t => <button key={t.id} className={`tile-card ${tileId === t.id ? "chosen" : ""}`} onClick={() => setTileId(t.id)}><img src={asset(t.imageUrl)} /><div><b>{t.name}</b><span>{t.category}</span></div></button>)}
            </div>
          </section>

          <aside className="right-panel">
            <div className="panel-tabs"><button className={tab === "scene" ? "on" : ""} onClick={() => setTab("scene")}>Scene</button><button className={tab === "ai" ? "on" : ""} onClick={() => setTab("ai")}>AI Analysis</button></div>
            {tab === "scene" ? <>
              <Section title="Surface Detection"><label>Detected Surface</label><select defaultValue="Floor"><option>Floor</option><option>Wall</option><option>Ceiling</option></select><label>Surface Area</label><div className="area">28.45 m²</div></Section>
              <Section title="Visualization Mode"><div className="seg"><button className="on">Realistic</button><button>Material Only</button></div></Section>
              <Section title="Environment"><div className="seg"><button className="on">Interior</button><button>Exterior</button></div></Section>
              <Section title="Options"><Toggle label="Smart Removal" value={smartRemoval} onChange={setSmartRemoval} /><Toggle label="Furniture Shadow" value={shadow} onChange={setShadow} /><Toggle label="Enhance Lighting" value={lighting} onChange={setLighting} /></Section>
              <div className="render-settings"><label>Material Profile</label><select value={profile} onChange={e => setProfile(e.target.value)}><option value="generic">Generic</option><option value="ceramic">Ceramic</option><option value="stone">Stone</option><option value="wood">Wood</option><option value="vinyl">Vinyl</option><option value="carpet">Carpet</option></select></div>
              <button className="apply" onClick={render}><AutoAwesome fontSize="small" /> Apply & Re-render</button>
            </> : <div className="ai-panel"><Insights /><h3>Heavy AI Analysis</h3><p>GroundingDINO + SAM2 + Depth Anything V2</p><div className="ai-ok">● Heavy stack ready</div><p className="muted">Floor detection and object-aware segmentation are active.</p></div>}
          </aside>
        </div>
      </main>
    </div>
  );
}

function Status({ label, value, good }: { label: string; value: string; good?: boolean }) { return <div className="sys-row"><span>{label}</span><b className={good ? "good" : ""}>{value}</b></div>; }
function Section({ title, children }: { title: string; children: React.ReactNode }) { return <div className="panel-section"><h3>{title}</h3>{children}</div>; }
function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) { return <div className="toggle-row"><span>{label}</span><button className={`toggle ${value ? "on" : ""}`} onClick={() => onChange(!value)}><span /></button></div>; }
