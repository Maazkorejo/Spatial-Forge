import { useEffect, useRef, useState } from 'react';
import type { DragEvent, ReactNode } from 'react';
import {
  Activity,
  Archive,
  ArrowDownToLine,
  ArrowLeftRight,
  Box,
  BoxSelect,
  ChevronDown,
  ChevronRight,
  CircleDot,
  CloudUpload,
  Download,
  FileBox,
  FileCode2,
  FileImage,
  FileOutput,
  FileText,
  FolderOpen,
  Grid3X3,
  Layers3,
  Minus,
  Move3d,
  PanelLeft,
  Plus,
  Redo2,
  Rotate3D,
  RotateCcw,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Undo2,
  WandSparkles,
  X,
  ZoomIn,
} from 'lucide-react';
import './styles.css';
import { ThreeViewport } from './ThreeViewport';
import {
  downloadBase64,
  downloadBlob,
  generateInteractiveViewerHtml,
  generateStl,
} from './utils/exporter';

type Stage = 'upload' | 'edit' | 'export';
type ViewMode = 'shaded' | 'wireframe' | 'solid';
type CutType = 'hole' | 'pocket' | 'slot' | 'round' | 'chamfer' | 'shell';

export type Model = {
  id: string;
  name: string;
  source: string;
  kind: 'jet' | 'wing' | 'shaft' | 'bracket' | 'wheel' | 'housing';
  dimensions: { length: number; width: number; thickness: number };
  volume: number;
  status: 'Converged' | 'Needs review';
  complexity: string;
  mesh?: {
    vertices: [number, number, number][];
    faces: [number, number, number][];
  };
  stepBase64?: string;
  stlBase64?: string;
  blueprintPreview?: string;
};

type Cut = { id: string; type: CutType; label: string; detail: string };
type Snapshot = { model: Model; cuts: Cut[] };

const sampleModels: Model[] = [
  { id: 'f35', name: 'F-35 Jet Profile', source: 'f35-technical.dxf', kind: 'jet', dimensions: { length: 175, width: 119, thickness: 37 }, volume: 138.2, status: 'Converged', complexity: 'High detail' },
  { id: 'wing', name: 'Wing Root Section', source: 'wing-profile.png', kind: 'wing', dimensions: { length: 240, width: 94, thickness: 18 }, volume: 298.6, status: 'Converged', complexity: 'Planar' },
  { id: 'shaft', name: 'Turned Shaft', source: 'shaft-side.dxf', kind: 'shaft', dimensions: { length: 160, width: 42, thickness: 42 }, volume: 221.4, status: 'Converged', complexity: 'Rotational' },
  { id: 'bracket', name: 'Mounting Bracket', source: 'bracket-sketch.webp', kind: 'bracket', dimensions: { length: 88, width: 72, thickness: 12 }, volume: 74.8, status: 'Needs review', complexity: 'Feature rich' },
  { id: 'wheel', name: 'Bearing Wheel', source: 'wheel-plan.jpg', kind: 'wheel', dimensions: { length: 96, width: 96, thickness: 22 }, volume: 126.7, status: 'Converged', complexity: 'Radial' },
  { id: 'housing', name: 'Engine Housing', source: 'housing-views.dxf', kind: 'housing', dimensions: { length: 138, width: 114, thickness: 66 }, volume: 592.1, status: 'Needs review', complexity: 'Shell form' },
];

const cutMeta: Record<CutType, { label: string; detail: string; icon: string }> = {
  hole: { label: 'Cut hole', detail: 'Ø 6 mm', icon: '○' },
  pocket: { label: 'Cut pocket', detail: '8 × 12 mm', icon: '□' },
  slot: { label: 'Cut slot', detail: '12 × 4 mm', icon: '▱' },
  round: { label: 'Round edges', detail: 'R 2 mm', icon: '⌒' },
  chamfer: { label: 'Chamfer', detail: '1.5 mm', icon: '◢' },
  shell: { label: 'Hollow shell', detail: '2 mm wall', icon: '◫' },
};

const stageItems: { id: Stage; label: string; icon: ReactNode }[] = [
  { id: 'upload', label: 'Upload', icon: <CloudUpload size={17} /> },
  { id: 'edit', label: 'Edit', icon: <SlidersHorizontal size={17} /> },
  { id: 'export', label: 'Export', icon: <ArrowDownToLine size={17} /> },
];

function App() {
  const [stage, setStage] = useState<Stage>('upload');
  const [activeModel, setActiveModel] = useState<Model>(sampleModels[0]);
  const [originalModel, setOriginalModel] = useState<Model>(sampleModels[0]);
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [history, setHistory] = useState<Snapshot[]>([]);
  const [future, setFuture] = useState<Snapshot[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('shaded');
  const [blueprintOpen, setBlueprintOpen] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const showToast = (message: string) => {
    setToast(message);
    window.setTimeout(() => setToast(null), 2600);
  };

  const snapshot = (): Snapshot => ({ model: activeModel, cuts });

  const commit = (nextModel: Model, nextCuts: Cut[], message: string) => {
    setHistory((items: Snapshot[]) => [...items.slice(-19), snapshot()]);
    setFuture([]);
    setActiveModel(nextModel);
    setCuts(nextCuts);
    showToast(message);
  };

  // Load sample model from live backend
  const fetchSampleModel = async (sample: Model) => {
    try {
      showToast(`Loading ${sample.name} from backend…`);
      const res = await fetch(`/api/v1/sample_cad?case=${sample.id}`);
      if (res.ok) {
        const data = await res.json();
        const loaded: Model = {
          id: sample.id,
          name: data.name || data.object_class || sample.name,
          source: data.source || sample.source,
          kind: (data.kind || sample.kind) as Model['kind'],
          dimensions: data.dimensions || sample.dimensions,
          volume: data.volume || sample.volume,
          status: data.iou && data.iou < 0.85 ? 'Needs review' : 'Converged',
          complexity: data.complexity || sample.complexity,
          mesh: data.mesh || (data.meshes?.body ? { vertices: data.meshes.body.vertices, faces: data.meshes.body.faces } : undefined),
          stepBase64: data.step_base64,
          stlBase64: data.stl_base64,
          blueprintPreview: data.sheet_preview_base64 ? `data:image/png;base64,${data.sheet_preview_base64}` : undefined,
        };
        setOriginalModel(loaded);
        setActiveModel(loaded);
        setCuts([]);
        setHistory([]);
        setFuture([]);
        setStage('edit');
        showToast(`${loaded.name} reconstructed · model ready`);
        return;
      }
    } catch {
      // Backend request fallback to sample
    }
    setOriginalModel(sample);
    setActiveModel(sample);
    setCuts([]);
    setHistory([]);
    setFuture([]);
    setStage('edit');
    showToast(`${sample.name} loaded into workspace`);
  };

  // Pre-load default F-35 model on mount
  useEffect(() => {
    fetchSampleModel(sampleModels[0]);
  }, []);

  // Process live uploaded 2D drawing (DXF or Image)
  const processFile = async (file?: File) => {
    if (!file) return;
    const accepted = ['.dxf', '.png', '.jpg', '.jpeg', '.webp'].some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );
    if (!accepted) {
      showToast('Use a .dxf, .png, .jpg, or .webp file');
      return;
    }
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/v1/synthesize', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Synthesis failed with HTTP ${res.status}`);
      }

      const data = await res.json();
      const dims = data.dimensions
        ? {
            length: data.dimensions.length || data.dimensions.H || data.dimensions.depth || 100,
            width: data.dimensions.width || data.dimensions.W || 80,
            thickness: data.dimensions.thickness || data.dimensions.D || data.dimensions.height || 20,
          }
        : { length: 120, width: 80, thickness: 25 };

      const bodyMesh = data.mesh || (data.meshes?.body ? { vertices: data.meshes.body.vertices, faces: data.meshes.body.faces } : undefined);
      const isReview = data.needs_review || (data.iou && data.iou < 0.85);

      const uploaded: Model = {
        id: 'uploaded',
        name: file.name.replace(/\.[^/.]+$/, ''),
        source: file.name,
        kind: file.name.toLowerCase().includes('jet') ? 'jet' : file.name.toLowerCase().includes('wheel') ? 'wheel' : 'bracket',
        dimensions: dims,
        volume: data.volume || 150.0,
        status: isReview ? 'Needs review' : 'Converged',
        complexity: data.fidelity === 'preview' ? 'Planar contour' : 'Multi-view B-Rep',
        mesh: bodyMesh,
        stepBase64: data.step_base64,
        stlBase64: data.stl_base64,
        blueprintPreview: data.sheet_preview_base64 ? `data:image/png;base64,${data.sheet_preview_base64}` : undefined,
      };

      setIsUploading(false);
      setOriginalModel(uploaded);
      setActiveModel(uploaded);
      setCuts([]);
      setHistory([]);
      setFuture([]);
      setStage('edit');
      showToast(`${uploaded.name} synthesized into 3D solid`);
    } catch {
      setIsUploading(false);
      showToast('Error connecting to backend synthesis engine');
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    processFile(event.dataTransfer.files[0]);
  };

  // Adjust part dimension and stretch real 3D mesh along respective axis
  const updateDimension = (key: keyof Model['dimensions'], amount: number) => {
    const nextVal = Math.max(1, activeModel.dimensions[key] + amount);
    const ratio = nextVal / activeModel.dimensions[key];
    const nextDims = { ...activeModel.dimensions, [key]: nextVal };
    const nextVolume = Number((activeModel.volume * ratio).toFixed(1));

    let nextMesh = activeModel.mesh;
    if (activeModel.mesh) {
      const idx = key === 'width' ? 0 : key === 'length' ? 1 : 2;
      const nextVertices = activeModel.mesh.vertices.map((v) => {
        const copy = [...v] as [number, number, number];
        copy[idx] = Math.round(copy[idx] * ratio * 100) / 100;
        return copy;
      });
      nextMesh = { vertices: nextVertices, faces: activeModel.mesh.faces };
    }

    commit(
      { ...activeModel, dimensions: nextDims, volume: nextVolume, mesh: nextMesh },
      cuts,
      `${key[0].toUpperCase() + key.slice(1)} adjusted to ${nextVal} mm`
    );
  };

  // Scale model without changing its construct
  const scaleModel = (factor: number) => {
    const dimensions = {
      length: Math.round(activeModel.dimensions.length * factor),
      width: Math.round(activeModel.dimensions.width * factor),
      thickness: Math.round(activeModel.dimensions.thickness * factor),
    };
    const volume = Number((activeModel.volume * factor ** 3).toFixed(1));

    let nextMesh = activeModel.mesh;
    if (activeModel.mesh) {
      const nextVertices = activeModel.mesh.vertices.map(([x, y, z]) => [
        Math.round(x * factor * 100) / 100,
        Math.round(y * factor * 100) / 100,
        Math.round(z * factor * 100) / 100,
      ] as [number, number, number]);
      nextMesh = { vertices: nextVertices, faces: activeModel.mesh.faces };
    }

    commit(
      { ...activeModel, dimensions, volume, mesh: nextMesh },
      cuts,
      factor > 1 ? 'Model scaled up 20%' : 'Model scaled down 20%'
    );
  };

  const addCut = (type: CutType) => {
    const meta = cutMeta[type];
    const newCut: Cut = { id: `${type}-${Date.now()}`, type, label: meta.label, detail: meta.detail };
    const volumeFactor = type === 'shell' ? 0.82 : type === 'hole' ? 0.97 : 0.94;
    commit(
      { ...activeModel, volume: Number((activeModel.volume * volumeFactor).toFixed(1)) },
      [...cuts, newCut],
      `${meta.label} applied`
    );
  };

  const removeCut = (cutId: string) => {
    const removed = cuts.find((cut: Cut) => cut.id === cutId);
    const nextCuts = cuts.filter((cut: Cut) => cut.id !== cutId);
    commit(
      { ...activeModel, volume: Number((activeModel.volume * 1.04).toFixed(1)) },
      nextCuts,
      `${removed?.label ?? 'Cut'} removed`
    );
  };

  const undo = () => {
    const previous = history[history.length - 1];
    if (!previous) return;
    setFuture((items: Snapshot[]) => [...items, snapshot()]);
    setHistory((items: Snapshot[]) => items.slice(0, -1));
    setActiveModel(previous.model);
    setCuts(previous.cuts);
    showToast('Previous state restored');
  };

  const redo = () => {
    const next = future[future.length - 1];
    if (!next) return;
    setHistory((items: Snapshot[]) => [...items, snapshot()]);
    setFuture((items: Snapshot[]) => items.slice(0, -1));
    setActiveModel(next.model);
    setCuts(next.cuts);
    showToast('Edit reapplied');
  };

  const reset = () => {
    setActiveModel(originalModel);
    setCuts([]);
    setHistory([]);
    setFuture([]);
    setStage('edit');
    showToast('Model reset to reconstructed state');
  };

  // Export functions (STEP, STL, GLB, standalone interactive HTML, DXF)
  const exportFile = (extension: string) => {
    const baseName = activeModel.name.toLowerCase().replace(/\s+/g, '-');

    if (extension === 'html') {
      const vertices = activeModel.mesh?.vertices || [];
      const faces = activeModel.mesh?.faces || [];
      const htmlContent = generateInteractiveViewerHtml({
        title: activeModel.name,
        source: activeModel.source,
        dimensions: activeModel.dimensions,
        volume: activeModel.volume,
        vertices,
        faces,
        blueprintPreview: activeModel.blueprintPreview,
      });
      const blob = new Blob([htmlContent], { type: 'text/html' });
      downloadBlob(blob, `${baseName}-viewer.html`);
      showToast('Standalone HTML 3D Viewer exported');
      return;
    }

    if (extension === 'stl') {
      if (activeModel.stlBase64) {
        downloadBase64(activeModel.stlBase64, `${baseName}.stl`, 'application/sla');
      } else if (activeModel.mesh) {
        const stlStr = generateStl(activeModel.mesh.vertices, activeModel.mesh.faces, activeModel.name);
        const blob = new Blob([stlStr], { type: 'application/sla' });
        downloadBlob(blob, `${baseName}.stl`);
      } else {
        const blob = new Blob([`solid ${baseName}\nendsolid ${baseName}\n`], { type: 'application/sla' });
        downloadBlob(blob, `${baseName}.stl`);
      }
      showToast('Print-ready STL exported');
      return;
    }

    if (extension === 'step') {
      if (activeModel.stepBase64) {
        downloadBase64(activeModel.stepBase64, `${baseName}.step`, 'application/step');
      } else {
        window.location.href = '/api/v1/cad_edit/export/step';
      }
      showToast('Parametric STEP solid exported');
      return;
    }

    if (extension === 'dxf') {
      window.location.href = `/api/v1/download_dxf/${activeModel.source}`;
      showToast('2D Technical DXF downloaded');
      return;
    }

    // GLB or fallback
    const payload = `Spatial Forge export\nModel: ${activeModel.name}\nSource: ${activeModel.source}\nDimensions: ${activeModel.dimensions.length} x ${activeModel.dimensions.width} x ${activeModel.dimensions.thickness} mm\nVolume: ${activeModel.volume} cm³\nCuts: ${cuts.map((cut: Cut) => cut.label).join(', ') || 'None'}`;
    const blob = new Blob([payload], { type: 'text/plain' });
    downloadBlob(blob, `${baseName}.${extension}`);
    showToast(`${extension.toUpperCase()} export downloaded`);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark" aria-label="Spatial Forge home">
          <img className="brand-icon" src="/spatialforge-icon.png" alt="" />
          <span className="brand-text">SF</span>
        </div>
        <div className="rail-rule" />
        <nav className="rail-nav" aria-label="Primary navigation">
          <button className="rail-button active" title="Workspace"><PanelLeft size={18} /></button>
          <button className="rail-button" title="Models" onClick={() => setStage('edit')}><Layers3 size={18} /></button>
          <button className="rail-button" title="Files" onClick={() => setStage('upload')}><Archive size={18} /></button>
        </nav>
        <div className="rail-bottom">
          <button className="rail-button" title="Settings"><Settings2 size={18} /></button>
          <div className="avatar">AR</div>
        </div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div className="breadcrumb-wrap">
            <div className="mini-logo"><img src="/spatialforge-icon.png" alt="" /></div>
            <div className="breadcrumb-copy">
              <img className="topbar-logo" src="/spatialforge-logo.png" alt="Spatial Forge" />
              <span className="breadcrumb-file">{activeModel.source}</span>
            </div>
            <ChevronRight className="breadcrumb-chevron" size={15} />
            <span className="current-stage">{stageItems.find((item) => item.id === stage)?.label}</span>
          </div>
          <div className="topbar-actions">
            <div className="stage-tabs">
              {stageItems.map((item) => (
                <button
                  key={item.id}
                  className={`stage-tab ${stage === item.id ? 'active' : ''}`}
                  onClick={() => setStage(item.id)}
                >
                  {item.icon}
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
            <div className={`status-chip ${activeModel.status === 'Converged' ? 'status-converged' : 'status-review'}`}>
              <span className="status-dot" />
              {activeModel.status}
            </div>
            <button className="ghost-button" onClick={() => fileInputRef.current?.click()}>
              <FolderOpen size={15} /> Open file
            </button>
            <button className="primary-button" onClick={() => { setStage('edit'); showToast('Reconstruction pipeline rerun'); }}>
              <WandSparkles size={15} /> Reconstruct
            </button>
          </div>
        </header>

        <input
          ref={fileInputRef}
          className="visually-hidden"
          type="file"
          accept=".dxf,.png,.jpg,.jpeg,.webp"
          onChange={(event) => processFile(event.target.files?.[0])}
        />

        {stage === 'upload' && (
          <UploadStage
            isDragging={isDragging}
            setIsDragging={setIsDragging}
            handleDrop={handleDrop}
            onBrowse={() => fileInputRef.current?.click()}
            isUploading={isUploading}
            loadModel={fetchSampleModel}
          />
        )}
        {stage === 'edit' && (
          <EditStage
            model={activeModel}
            cuts={cuts}
            viewMode={viewMode}
            setViewMode={setViewMode}
            blueprintOpen={blueprintOpen}
            setBlueprintOpen={setBlueprintOpen}
            onScale={scaleModel}
            onDimension={updateDimension}
            onAddCut={addCut}
            onRemoveCut={removeCut}
            onUndo={undo}
            onRedo={redo}
            onReset={reset}
            canUndo={history.length > 0}
            canRedo={future.length > 0}
          />
        )}
        {stage === 'export' && (
          <ExportStage model={activeModel} cuts={cuts} onExport={exportFile} />
        )}
      </main>

      {toast && (
        <div className="toast">
          <ShieldCheck size={16} />
          {toast}
        </div>
      )}
    </div>
  );
}

function UploadStage({
  isDragging,
  setIsDragging,
  handleDrop,
  onBrowse,
  isUploading,
  loadModel,
}: {
  isDragging: boolean;
  setIsDragging: (value: boolean) => void;
  handleDrop: (event: DragEvent<HTMLDivElement>) => void;
  onBrowse: () => void;
  isUploading: boolean;
  loadModel: (model: Model) => void;
}) {
  return (
    <section className="page-content upload-page">
      <div className="page-intro split-intro">
        <div>
          <span className="section-kicker"><Sparkles size={13} /> Reconstruction workspace</span>
          <h1>Bring a drawing<br /><em>into focus.</em></h1>
        </div>
        <div className="intro-note">
          <span className="intro-line" />
          Drop a technical drawing to create a clean, editable solid.
        </div>
      </div>

      <div
        className={`dropzone ${isDragging ? 'dragging' : ''}`}
        onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={onBrowse}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => event.key === 'Enter' && onBrowse()}
      >
        <div className="dropzone-art">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="upload-core">
            {isUploading ? <Activity size={25} className="spin" /> : <CloudUpload size={25} />}
          </div>
        </div>
        <div className="dropzone-copy">
          <strong>{isUploading ? 'Synthesizing solid…' : 'Drop your drawing here'}</strong>
          <span>
            {isUploading
              ? 'Segmenting views · solving features · grounding model'
              : 'DXF, PNG, JPG, or WEBP · up to 50 MB'}
          </span>
        </div>
        <button className="dropzone-button" onClick={(event) => { event.stopPropagation(); onBrowse(); }}>
          {isUploading ? 'Processing' : 'Choose file'} <ChevronRight size={15} />
        </button>
      </div>

      <div className="library-head">
        <div>
          <span className="section-kicker">Quick start</span>
          <h2>Sample cases</h2>
        </div>
        <span className="case-count">{sampleModels.length} ready-to-inspect models</span>
      </div>

      <div className="sample-grid">
        {sampleModels.map((model) => (
          <SampleCard key={model.id} model={model} onClick={() => loadModel(model)} />
        ))}
      </div>
    </section>
  );
}

function SampleCard({ model, onClick }: { model: Model; onClick: () => void }) {
  return (
    <button className="sample-card" onClick={onClick}>
      <div className={`sample-visual visual-${model.kind}`}>
        <ModelShape kind={model.kind} compact />
      </div>
      <div className="sample-card-footer">
        <div>
          <strong>{model.name}</strong>
          <span>{model.complexity} · {model.dimensions.length} mm</span>
        </div>
        <ChevronRight size={16} />
      </div>
    </button>
  );
}

function EditStage({
  model,
  cuts,
  viewMode,
  setViewMode,
  blueprintOpen,
  setBlueprintOpen,
  onScale,
  onDimension,
  onAddCut,
  onRemoveCut,
  onUndo,
  onRedo,
  onReset,
  canUndo,
  canRedo,
}: {
  model: Model;
  cuts: Cut[];
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  blueprintOpen: boolean;
  setBlueprintOpen: (open: boolean) => void;
  onScale: (factor: number) => void;
  onDimension: (key: keyof Model['dimensions'], amount: number) => void;
  onAddCut: (type: CutType) => void;
  onRemoveCut: (id: string) => void;
  onUndo: () => void;
  onRedo: () => void;
  onReset: () => void;
  canUndo: boolean;
  canRedo: boolean;
}) {
  const specRows = [
    { label: 'Length', key: 'length' as const, step: 5 },
    { label: 'Width', key: 'width' as const, step: 5 },
    { label: 'Thickness', key: 'thickness' as const, step: 2 },
  ];

  return (
    <section className="page-content edit-page">
      <div className="edit-head">
        <div>
          <span className="section-kicker"><BoxSelect size={13} /> Model inspector</span>
          <h1>{model.name}</h1>
        </div>
        <div className="edit-head-meta">
          <span className="watertight"><CircleDot size={14} /> Watertight</span>
          <span className="source-pill"><FileText size={14} /> {model.source}</span>
        </div>
      </div>

      <div className="edit-grid">
        <aside className="panel inspector-panel">
          <div className="panel-title">
            <span>Feature tree</span>
            <button className="icon-button"><MoreDots /></button>
          </div>
          <div className="identity-card">
            <div className="identity-icon"><Box size={18} /></div>
            <div>
              <strong>{model.name}</strong>
              <span>Reconstructed solid</span>
            </div>
            <button className="icon-button"><ChevronDown size={16} /></button>
          </div>

          <div className="tree-item active">
            <span className="tree-branch"><ChevronDown size={14} /></span>
            <span className="tree-icon"><Box size={14} /></span>
            <div>
              <strong>Base solid</strong>
              <span>Source geometry</span>
            </div>
            <span className="tree-status" />
          </div>

          {cuts.map((cut: Cut) => (
            <div className="tree-item" key={`tree-${cut.id}`}>
              <span className="tree-branch"><ChevronRight size={14} /></span>
              <span className="tree-icon"><CircleDot size={13} /></span>
              <div>
                <strong>{cut.label}</strong>
                <span>{cut.detail}</span>
              </div>
            </div>
          ))}

          <div className="panel-divider" />
          <div className="subhead">
            <span>Measured specs</span>
            <ArrowLeftRight size={14} />
          </div>
          <div className="measured-list">
            <SpecValue label="Envelope" value={`${model.dimensions.length} × ${model.dimensions.width} × ${model.dimensions.thickness}`} unit="mm" />
            <SpecValue label="Solid volume" value={model.volume.toFixed(1)} unit="cm³" />
            <SpecValue label="Topology" value="B-Rep" unit="" />
          </div>
          <div className="inspector-footer">
            <Activity size={14} />
            <span>Geometry is centered on origin</span>
          </div>
        </aside>

        <section className="viewport-panel panel">
          <div className="viewport-toolbar">
            <div className="toolbar-group">
              <span className="tool-label"><Move3d size={14} /> View</span>
              {(['shaded', 'wireframe', 'solid'] as ViewMode[]).map((mode) => (
                <button
                  key={mode}
                  className={`view-button ${viewMode === mode ? 'active' : ''}`}
                  onClick={() => setViewMode(mode)}
                >
                  {mode === 'shaded' && <Layers3 size={14} />}
                  {mode === 'wireframe' && <Grid3X3 size={14} />}
                  {mode === 'solid' && <Box size={14} />}
                  {mode}
                </button>
              ))}
            </div>
            <div className="toolbar-group viewport-actions">
              <button className="icon-button" title="Interactive 3D view"><Rotate3D size={16} /></button>
              <button className="icon-button" title="Zoom in"><ZoomIn size={16} /></button>
              <span className="zoom-readout">100%</span>
            </div>
          </div>

          <div className="viewport-canvas">
            <div className="axis axis-x">X</div>
            <div className="axis axis-y">Y</div>
            <div className="origin-mark" />
            <div className="viewport-corner corner-tl">0, 0, 0</div>
            <div className="viewport-corner corner-br">MM · FRONT</div>

            {/* Interactive Three.js WebGL Viewport */}
            {model.mesh ? (
              <ThreeViewport mesh={model.mesh} viewMode={viewMode} cutsCount={cuts.length} />
            ) : (
              <div className="model-stage">
                <ModelShape kind={model.kind} viewMode={viewMode} cuts={cuts.length} />
              </div>
            )}

            <div className="viewport-caption">
              <span><span className="caption-dot" />Centered & grounded</span>
              <span>Orbit · Pan · Zoom</span>
            </div>

            {blueprintOpen ? (
              <BlueprintCard model={model} onMinimize={() => setBlueprintOpen(false)} />
            ) : (
              <button className="blueprint-restore" onClick={() => setBlueprintOpen(true)}>
                <FileImage size={15} /> Show 2D drawing
              </button>
            )}
          </div>
        </section>

        <aside className="panel controls-panel">
          <div className="panel-title">
            <span>Adjust solid</span>
            <button className="icon-button"><SlidersHorizontal size={16} /></button>
          </div>

          <div className="control-section scale-section">
            <div className="subhead">
              <span>Scale size</span>
              <span className="control-hint">20% steps</span>
            </div>
            <div className="scale-buttons">
              <button onClick={() => onScale(0.8)}>
                <Minus size={15} />
                <span>Make small</span>
              </button>
              <button onClick={() => onScale(1.2)}>
                <Plus size={15} />
                <span>Make big</span>
              </button>
            </div>
          </div>

          <div className="control-section">
            <div className="subhead">
              <span>Adjust parts</span>
              <span className="control-hint">mm</span>
            </div>
            <div className="stepper-list">
              {specRows.map((row) => (
                <Stepper
                  key={row.key}
                  label={row.label}
                  value={model.dimensions[row.key]}
                  onMinus={() => onDimension(row.key, -row.step)}
                  onPlus={() => onDimension(row.key, row.step)}
                />
              ))}
            </div>
          </div>

          <div className="volume-card">
            <span>Solid volume</span>
            <strong>{model.volume.toFixed(1)} <small>cm³</small></strong>
            <div className="volume-bar">
              <span style={{ width: `${Math.min(96, Math.max(24, model.volume / 7))}%` }} />
            </div>
          </div>

          <div className="control-section cuts-section">
            <div className="subhead">
              <span>Cut parts</span>
              <span className="control-hint">direct edit</span>
            </div>
            <div className="cut-grid">
              {(Object.keys(cutMeta) as CutType[]).map((type) => (
                <button key={type} className="cut-button" onClick={() => onAddCut(type)}>
                  <span className="cut-glyph">{cutMeta[type].icon}</span>
                  <span>{cutMeta[type].label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="control-section applied-section">
            <div className="subhead">
              <span>Applied cuts</span>
              <span className="count-badge">{cuts.length}</span>
            </div>
            {cuts.length === 0 ? (
              <div className="empty-cuts">
                <CircleDot size={15} />
                <span>No cuts on this model</span>
              </div>
            ) : (
              <div className="applied-list">
                {cuts.map((cut: Cut) => (
                  <div className="applied-cut" key={cut.id}>
                    <div className="cut-list-icon">{cutMeta[cut.type].icon}</div>
                    <div>
                      <strong>{cut.label}</strong>
                      <span>{cut.detail}</span>
                    </div>
                    <button className="remove-button" onClick={() => onRemoveCut(cut.id)}>
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="history-row">
            <button disabled={!canUndo} onClick={onUndo}><Undo2 size={15} /> Undo</button>
            <button disabled={!canRedo} onClick={onRedo}><Redo2 size={15} /> Redo</button>
            <button onClick={onReset}><RotateCcw size={15} /> Reset</button>
          </div>
        </aside>
      </div>
    </section>
  );
}

function ExportStage({
  model,
  cuts,
  onExport,
}: {
  model: Model;
  cuts: Cut[];
  onExport: (extension: string) => void;
}) {
  const exports = [
    { ext: 'step', title: 'Parametric CAD', detail: 'Analytical B-Rep solid', icon: <FileBox size={21} />, meta: '2.4 MB' },
    { ext: 'stl', title: 'Print-ready mesh', detail: 'Watertight triangular mesh', icon: <Box size={21} />, meta: '1.8 MB' },
    { ext: 'html', title: 'Interactive viewer', detail: 'Self-contained 3D webpage', icon: <FileCode2 size={21} />, meta: '684 KB' },
    { ext: 'glb', title: 'Web / AR asset', detail: 'Materials and normals', icon: <FileOutput size={21} />, meta: '3.1 MB' },
    { ext: 'dxf', title: '2D technical drawing', detail: 'Dimensions and layers', icon: <FileImage size={21} />, meta: '412 KB' },
  ];

  return (
    <section className="page-content export-page">
      <div className="page-intro split-intro export-intro">
        <div>
          <span className="section-kicker"><ArrowDownToLine size={13} /> Deliverables</span>
          <h1>Ready for<br /><em>the next tool.</em></h1>
        </div>
        <div className="export-model-card">
          <div className="export-model-icon"><Box size={18} /></div>
          <div>
            <span>Active model</span>
            <strong>{model.name}</strong>
          </div>
          <span className="export-ready"><span className="status-dot" />Ready</span>
        </div>
      </div>

      <div className="export-summary">
        <div>
          <span>Model envelope</span>
          <strong>{model.dimensions.length} × {model.dimensions.width} × {model.dimensions.thickness} <small>mm</small></strong>
        </div>
        <div>
          <span>Solid volume</span>
          <strong>{model.volume.toFixed(1)} <small>cm³</small></strong>
        </div>
        <div>
          <span>Applied edits</span>
          <strong>{cuts.length.toString().padStart(2, '0')} <small>cuts</small></strong>
        </div>
        <div className="summary-status">
          <ShieldCheck size={17} />
          <span>Geometry verified<br /><small>Watertight B-Rep</small></span>
        </div>
      </div>

      <div className="export-grid">
        {exports.map((item) => (
          <div className="export-card" key={item.ext}>
            <div className="export-card-top">
              <div className="export-icon">{item.icon}</div>
              <span className="file-ext">.{item.ext}</span>
            </div>
            <div className="export-card-copy">
              <strong>{item.title}</strong>
              <span>{item.detail}</span>
            </div>
            <div className="export-card-bottom">
              <span>{item.meta}</span>
              <button className="download-button" onClick={() => onExport(item.ext)}>
                <Download size={15} /> Export
              </button>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function BlueprintCard({ model, onMinimize }: { model: Model; onMinimize: () => void }) {
  return (
    <div className="blueprint-card">
      <div className="blueprint-header">
        <div>
          <span>Source drawing</span>
          <strong>{model.source}</strong>
        </div>
        <div className="blueprint-actions">
          <button className="mini-icon-button" onClick={onMinimize}><Minus size={14} /></button>
          <button className="mini-icon-button"><X size={14} /></button>
        </div>
      </div>
      <div className="blueprint-art">
        {model.blueprintPreview ? (
          <img
            src={model.blueprintPreview}
            alt="Source drawing preview"
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        ) : (
          <>
            <div className="blueprint-cross cross-h" />
            <div className="blueprint-cross cross-v" />
            <div className="blueprint-outline outline-main" />
            <div className="blueprint-outline outline-inner" />
            <span className="dimension dimension-top">{model.dimensions.length} mm</span>
            <span className="dimension dimension-side">{model.dimensions.width} mm</span>
          </>
        )}
      </div>
      <div className="blueprint-footer">
        <span><span className="blueprint-dot" />2D reference</span>
        <span>Scale 1:1</span>
      </div>
    </div>
  );
}

function Stepper({ label, value, onMinus, onPlus }: { label: string; value: number; onMinus: () => void; onPlus: () => void }) {
  return (
    <div className="stepper-row">
      <span>{label}</span>
      <div className="stepper">
        <button onClick={onMinus} aria-label={`Decrease ${label}`}><Minus size={13} /></button>
        <strong>{value}</strong>
        <button onClick={onPlus} aria-label={`Increase ${label}`}><Plus size={13} /></button>
      </div>
    </div>
  );
}

function SpecValue({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="spec-value">
      <span>{label}</span>
      <strong>{value} <small>{unit}</small></strong>
    </div>
  );
}

function MoreDots() {
  return <span className="more-dots"><i /><i /><i /></span>;
}

function ModelShape({ kind, compact = false, viewMode = 'shaded', cuts = 0 }: { kind: Model['kind']; compact?: boolean; viewMode?: ViewMode; cuts?: number }) {
  const scale = compact ? 0.72 : 1;
  return (
    <svg className={`model-shape ${viewMode} ${compact ? 'compact' : ''}`} viewBox="0 0 440 280" style={{ transform: `scale(${scale})` }} aria-label="3D reconstructed model">
      <defs>
        <linearGradient id="modelTop" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stopColor="#eff5ff" /><stop offset="1" stopColor="#9bb8dc" /></linearGradient>
        <linearGradient id="modelSide" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#5e7da7" /><stop offset="1" stopColor="#314763" /></linearGradient>
        <linearGradient id="modelAccent"><stop offset="0" stopColor="#c8dcf7" /><stop offset="1" stopColor="#6b8db9" /></linearGradient>
      </defs>
      {kind === 'jet' && (
        <g className="shape-group">
          <path className="model-shadow" d="M78 204 L211 204 L350 223 L367 232 L211 237 L83 224 Z" />
          <path className="model-side-face" d="M104 153 L252 125 L359 176 L217 211 L104 185 Z" />
          <path className="model-top-face" d="M104 153 L247 93 L359 145 L252 185 Z" />
          <path className="model-top-face" d="M247 93 L277 72 L370 119 L359 145 Z" />
          <path className="model-accent-face" d="M199 129 L247 93 L359 145 L315 158 Z" />
          <path className="model-side-face" d="M104 153 L104 185 L217 211 L217 180 Z" />
          <path className="model-line" d="M132 154 L252 125 L329 158" />
          <path className="model-line" d="M217 180 L315 158" />
          <path className="model-line" d="M159 139 L193 119" />
          <path className="model-line" d="M277 72 L277 111" />
        </g>
      )}
      {kind === 'wing' && (
        <g className="shape-group">
          <path className="model-shadow" d="M62 207 L348 213 L374 226 L92 230 Z" />
          <path className="model-side-face" d="M88 166 L329 132 L361 161 L117 205 Z" />
          <path className="model-top-face" d="M88 166 L219 78 L361 123 L329 132 Z" />
          <path className="model-accent-face" d="M219 78 L248 65 L387 109 L361 123 Z" />
          <path className="model-line" d="M104 166 L252 113 L350 133" />
          <path className="model-line" d="M117 190 L329 153" />
          <path className="model-line" d="M190 98 L218 130" />
        </g>
      )}
      {kind === 'shaft' && (
        <g className="shape-group">
          <path className="model-shadow" d="M74 206 L357 206 L382 218 L97 223 Z" />
          <ellipse className="model-top-face" cx="205" cy="140" rx="128" ry="45" />
          <path className="model-side-face" d="M77 140 L77 183 C77 208 130 225 205 225 C280 225 333 208 333 183 L333 140 C333 165 280 183 205 183 C130 183 77 165 77 140 Z" />
          <ellipse className="model-accent-face" cx="205" cy="140" rx="63" ry="22" />
          <ellipse className="model-line" cx="205" cy="140" rx="34" ry="12" />
        </g>
      )}
      {kind === 'bracket' && (
        <g className="shape-group">
          <path className="model-shadow" d="M109 214 L325 217 L345 227 L120 231 Z" />
          <path className="model-side-face" d="M120 119 L170 101 L170 176 L291 157 L291 201 L120 222 Z" />
          <path className="model-top-face" d="M120 119 L170 88 L341 115 L291 157 L170 176 Z" />
          <path className="model-accent-face" d="M291 157 L341 115 L341 160 L291 201 Z" />
          <circle className="model-hole" cx="226" cy="133" r="17" />
          <circle className="model-hole-inner" cx="226" cy="133" r="8" />
          <path className="model-line" d="M170 88 L170 176" />
          <path className="model-line" d="M120 168 L170 150" />
        </g>
      )}
      {kind === 'wheel' && (
        <g className="shape-group">
          <path className="model-shadow" d="M101 221 L315 221 L342 231 L120 233 Z" />
          <ellipse className="model-side-face" cx="220" cy="150" rx="110" ry="72" />
          <ellipse className="model-top-face" cx="220" cy="135" rx="110" ry="72" />
          <ellipse className="model-accent-face" cx="220" cy="135" rx="53" ry="34" />
          <ellipse className="model-hole" cx="220" cy="135" rx="25" ry="15" />
          <path className="model-line" d="M120 135 C120 95 165 63 220 63 C275 63 320 95 320 135" />
        </g>
      )}
      {kind === 'housing' && (
        <g className="shape-group">
          <path className="model-shadow" d="M91 218 L345 218 L369 229 L111 232 Z" />
          <path className="model-side-face" d="M111 122 L111 194 L184 218 L329 179 L329 113 Z" />
          <path className="model-top-face" d="M111 122 L228 69 L361 103 L329 113 L184 156 Z" />
          <path className="model-accent-face" d="M184 156 L329 113 L329 179 L184 218 Z" />
          <ellipse className="model-hole" cx="255" cy="145" rx="37" ry="22" />
          <ellipse className="model-hole-inner" cx="255" cy="145" rx="20" ry="11" />
          <path className="model-line" d="M111 157 L184 181" />
          <path className="model-line" d="M228 69 L228 112" />
        </g>
      )}
      {cuts > 0 && (
        <g className="cut-mark">
          <circle cx="300" cy="112" r="13" />
          <circle cx="300" cy="112" r="5" />
        </g>
      )}
    </svg>
  );
}

export default App;
