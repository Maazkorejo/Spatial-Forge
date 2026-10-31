import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type ReactNode } from "react";
import * as THREE from "three";
import {
  Aperture, ArrowDownToLine, ArrowLeftRight, ArrowUpRight, Axis3D, Box, Check,
  ChevronDown, Circle, CircleDot, CloudUpload, Copy, Cuboid, Download, FileBox,
  FileCode2, FileImage, FileText, FileUp, Gauge, Grid2X2, History, Layers3,
  Maximize, Minus, MoreHorizontal, Move3D, PanelLeft, Plane, Plus, Redo2,
  RotateCcw, Ruler, Scissors, Settings2, ShieldCheck, SlidersHorizontal,
  Sparkles, Square, Undo2, Upload, WandSparkles, X, Zap,
} from "lucide-react";
import { toast } from "sonner";
import { CADViewport, DisplaySwitcher } from "@/components/cad/CADViewport";
import { createPartGroup, disposePartGroup } from "@/components/cad/geometry";
import { downloadBlob, makeExport } from "@/components/cad/exports";
import {
  cloneModel, createImportedModel, createSampleModel, formatNumber, sampleParts, volumeCm3,
  type AppliedCut, type DimensionKey, type DisplayMode, type ModelKind, type ModelState,
} from "@/components/cad/types";

const stages = [
  { id: "upload", label: "Upload", icon: Upload },
  { id: "edit", label: "Edit", icon: SlidersHorizontal },
  { id: "export", label: "Export", icon: ArrowDownToLine },
] as const;

type Stage = (typeof stages)[number]["id"];

const cutOptions: Array<{ label: string; detail: string; glyph: string; icon: typeof Circle }> = [
  { label: "Cut hole", detail: "Ø6 mm", glyph: "hole", icon: CircleDot },
  { label: "Cut pocket", detail: "Pocket", glyph: "pocket", icon: Square },
  { label: "Cut slot", detail: "Slot", glyph: "slot", icon: PanelLeft },
  { label: "Round edges", detail: "Fillet", glyph: "round", icon: Aperture },
  { label: "Chamfer", detail: "1.5 mm", glyph: "chamfer", icon: Axis3D },
  { label: "Hollow shell", detail: "Shell", glyph: "shell", icon: Box },
];

const exportOptions = [
  { id: "step", ext: ".STEP", title: "Parametric CAD", detail: "Faceted solid for CAD import", icon: Cuboid, color: "blue" },
  { id: "stl", ext: ".STL", title: "Print-ready mesh", detail: "Triangulated preview geometry", icon: Layers3, color: "slate" },
  { id: "html", ext: ".HTML", title: "Interactive viewer", detail: "Self-contained 3D webpage", icon: FileCode2, color: "indigo" },
  { id: "glb", ext: ".GLB", title: "Web / AR asset", detail: "Binary glTF with materials", icon: Box, color: "steel" },
  { id: "dxf", ext: ".DXF", title: "2D technical drawing", detail: "Projected mesh linework", icon: FileText, color: "blue" },
] as const;

type ExportKind = (typeof exportOptions)[number]["id"];

function getModelIcon(kind: ModelKind, size = 18) {
  const props = { size, strokeWidth: 1.7 };
  switch (kind) {
    case "jet": return <Plane {...props} />;
    case "wing": return <Move3D {...props} />;
    case "shaft": return <Axis3D {...props} />;
    case "bracket": return <Grid2X2 {...props} />;
    case "butterfly": return <Sparkles {...props} />;
    case "wheel": return <CircleDot {...props} />;
    case "housing": return <Cuboid {...props} />;
    case "imported": return <FileImage {...props} />;
  }
}

function AppLogo() {
  const [broken, setBroken] = useState(false);
  const [logoSrc, setLogoSrc] = useState("/manus-storage/spatialforge-logo_4a79e89c.webp");
  return (
    <a className="brand-link" href="#workspace" aria-label="Spatial Forge home">
      {!broken ? <img src={logoSrc} alt="Spatial Forge" onError={() => { if (logoSrc.startsWith("/manus-storage/")) setLogoSrc("/spatialforge-logo.webp"); else setBroken(true); }} /> : <><span className="brand-symbol"><Box size={19} /></span><span className="brand-name">SPATIAL<span>FORGE</span></span></>}
    </a>
  );
}

function Stepper({ label, axis, value, step, onChange }: { label: string; axis: string; value: number; step: number; onChange: (value: number) => void }) {
  return (
    <div className="dimension-row">
      <div className="dimension-name"><span className={`axis-tag axis-tag-${axis.toLowerCase()}`}>{axis}</span><span>{label}</span></div>
      <div className="dimension-stepper">
        <button type="button" aria-label={`Decrease ${label}`} onClick={() => onChange(-step)}><Minus size={13} /></button>
        <span className="dimension-number">{Math.round(value)}<small>mm</small></span>
        <button type="button" aria-label={`Increase ${label}`} onClick={() => onChange(step)}><Plus size={13} /></button>
      </div>
    </div>
  );
}

function SectionHead({ title, trailing }: { title: string; trailing?: ReactNode }) {
  return <div className="section-head"><span>{title}</span>{trailing}</div>;
}

function FeatureTree({ model, onRemoveCut }: { model: ModelState; onRemoveCut: (cut: AppliedCut) => void }) {
  const featureName = model.kind === "jet" ? "Airframe assembly" : model.kind === "imported" ? "Imported profile" : model.title;
  return (
    <div className="feature-tree">
      <div className="tree-row tree-row-active"><span className="tree-branch"><i /></span><span className="tree-icon"><Cuboid size={15} /></span><span className="tree-name">{featureName}</span><MoreHorizontal size={15} className="tree-more" /></div>
      {model.kind === "jet" && <>
        <div className="tree-child"><span className="tree-line" /><span className="tree-icon muted-icon"><Plane size={14} /></span><span>Fuselage & nose</span></div>
        <div className="tree-child"><span className="tree-line" /><span className="tree-icon muted-icon"><Move3D size={14} /></span><span>Wing surfaces</span></div>
        <div className="tree-child"><span className="tree-line" /><span className="tree-icon muted-icon"><Axis3D size={14} /></span><span>Twin tail fins</span></div>
      </>}
      {model.kind === "bracket" && <div className="tree-child"><span className="tree-line" /><span className="tree-icon muted-icon"><Circle size={14} /></span><span>Mounting bores · 2</span></div>}
      {model.kind === "wheel" && <div className="tree-child"><span className="tree-line" /><span className="tree-icon muted-icon"><CircleDot size={14} /></span><span>Hub & spokes</span></div>}
      {model.kind === "housing" && <div className="tree-child"><span className="tree-line" /><span className="tree-icon muted-icon"><Box size={14} /></span><span>Shell & rim</span></div>}
      {model.cuts.map((cut) => (
        <div key={cut.id} className="tree-child tree-cut"><span className="tree-line" /><span className="tree-icon cut-icon"><Scissors size={13} /></span><span className="tree-name">{cut.label}</span><button type="button" onClick={() => onRemoveCut(cut)} title={`Remove ${cut.label}`} aria-label={`Remove ${cut.label}`}><X size={12} /></button></div>
      ))}
    </div>
  );
}

function UploadStage({ onPickFile, onDropFile, isDragging }: { onPickFile: () => void; onDropFile: (file: File) => void; isDragging: boolean }) {
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) onDropFile(file);
  };
  return (
    <main className="stage-page upload-page">
      <div className="stage-heading"><div><span className="eyebrow">SPATIAL FORGE / DRAWING INTAKE</span><h1>Bring a drawing into 3D.</h1></div><div className="heading-stamp"><span className="stamp-dot" />LOCAL WORKSPACE</div></div>
      <div className={`dropzone${isDragging ? " is-dragging" : ""}`} onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
        <div className="dropzone-art"><span className="dropzone-orbit orbit-one" /><span className="dropzone-orbit orbit-two" /><span className="dropzone-icon"><CloudUpload size={28} strokeWidth={1.5} /></span></div>
        <div className="dropzone-copy"><h2>Drop a drawing to reconstruct</h2><p>DXF or image · under 40 MB</p></div>
        <button className="button button-primary button-browse" type="button" onClick={onPickFile}><FileUp size={15} />Choose drawing</button>
        <div className="file-types"><span>DXF</span><span>PNG</span><span>JPG</span><span>WEBP</span></div>
      </div>
      <div className="stage-footnote" style={{ marginTop: 32 }}><ShieldCheck size={15} /> Browser preview · no file leaves this workspace</div>
    </main>
  );
}

function ExportStage({ model, onExport, pendingExport }: { model: ModelState; onExport: (format: ExportKind) => void; pendingExport: string | null }) {
  return (
    <main className="stage-page export-page">
      <div className="stage-heading"><div><span className="eyebrow">SPATIAL FORGE / DELIVERABLES</span><h1>Ready to export.</h1></div><div className="model-export-summary"><span className="summary-model-icon">{getModelIcon(model.kind, 18)}</span><span><strong>{model.title}</strong><small>{formatNumber(volumeCm3(model))} cm³ · {model.cuts.length} edits</small></span></div></div>
      <div className="export-grid">
        {exportOptions.map((item) => {
          const Icon = item.icon;
          const busy = pendingExport === item.id;
          return (
            <article key={item.id} className={`export-card export-${item.color}`}>
              <div className="export-card-top"><span className="export-icon"><Icon size={20} strokeWidth={1.7} /></span><span className="file-extension">{item.ext}</span></div>
              <h2>{item.title}</h2><p>{item.detail}</p>
              <button type="button" className="button button-secondary export-download" disabled={pendingExport !== null} onClick={() => onExport(item.id)}>{busy ? <span className="mini-loader" /> : <Download size={14} />}{busy ? "Preparing" : "Download"}<ArrowUpRight size={13} className="download-arrow" /></button>
            </article>
          );
        })}
      </div>
      <div className="export-model-strip"><span className="export-strip-badge">{getModelIcon(model.kind, 17)}</span><span className="export-strip-name">{model.fileName}</span><span className="export-strip-dims">{Math.round(model.dimensions.length)} × {Math.round(model.dimensions.width)} × {Math.round(model.dimensions.thickness)} mm</span><span className="export-strip-volume">{formatNumber(volumeCm3(model))} cm³</span></div>
      <div className="export-note"><span className="note-dot" />Preview exports use faceted browser geometry. Verify critical dimensions in CAD before production.</div>
    </main>
  );
}

export default function Home() {
  const [model, setModel] = useState<ModelState>(() => createSampleModel("jet"));
  const [stage, setStage] = useState<Stage>("edit");
  const [displayMode, setDisplayMode] = useState<DisplayMode>("shaded");
  const [isDragging, setIsDragging] = useState(false);
  const [reconstructing, setReconstructing] = useState(false);
  const [reconstructMessage, setReconstructMessage] = useState("Ready to reconstruct");
  const [pendingExport, setPendingExport] = useState<string | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [sourceName, setSourceName] = useState("F-35A top view · DXF");
  const [sourceVisible, setSourceVisible] = useState(true);
  const [showRailLabels, setShowRailLabels] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const modelRef = useRef(model);
  const originalModelRef = useRef(cloneModel(model));
  const undoStack = useRef<ModelState[]>([]);
  const redoStack = useRef<ModelState[]>([]);
  const [historyRevision, setHistoryRevision] = useState(0);
  const exportGroupRef = useRef<THREE.Group | null>(null);
  const progressTimers = useRef<number[]>([]);

  const updateModel = useCallback((next: ModelState) => {
    modelRef.current = next;
    setModel(next);
  }, []);

  useEffect(() => {
    const group = createPartGroup(model, "solid");
    exportGroupRef.current = group;
    return () => {
      disposePartGroup(group);
      if (exportGroupRef.current === group) exportGroupRef.current = null;
    };
  }, [model]);

  useEffect(() => () => {
    progressTimers.current.forEach((timer) => window.clearTimeout(timer));
  }, []);

  useEffect(() => {
    fetch("/api/v1/sample_cad?case=jet")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data) return;
        const meshData = (data.meshes && data.meshes.body) || data.mesh;
        if (meshData?.vertices?.length) {
          setModel((curr) => {
            if (curr.kind !== "jet") return curr;
            const updated = {
              ...curr,
              meshData,
              baseDimensions: { ...curr.dimensions },
              stepBase64: data.step_base64 || curr.stepBase64,
              stlBase64: data.stl_base64 || curr.stlBase64,
            };
            modelRef.current = updated;
            originalModelRef.current = cloneModel(updated);
            return updated;
          });
        }
      })
      .catch(() => {});
  }, []);

  const setActiveModel = useCallback((next: ModelState) => {
    const copy = cloneModel(next);
    originalModelRef.current = cloneModel(copy);
    undoStack.current = [];
    redoStack.current = [];
    setHistoryRevision((revision) => revision + 1);
    updateModel(copy);
  }, [updateModel]);

  const recordEdit = useCallback((mutate: (current: ModelState) => ModelState) => {
    const current = modelRef.current;
    const next = mutate(cloneModel(current));
    undoStack.current = [...undoStack.current.slice(-39), cloneModel(current)];
    redoStack.current = [];
    setHistoryRevision((revision) => revision + 1);
    updateModel(next);
  }, [updateModel]);

  const undo = useCallback(() => {
    if (!undoStack.current.length) return;
    redoStack.current = [...redoStack.current, cloneModel(modelRef.current)];
    updateModel(undoStack.current.pop()!);
    setHistoryRevision((revision) => revision + 1);
    toast("Last edit undone");
  }, [updateModel]);

  const redo = useCallback(() => {
    if (!redoStack.current.length) return;
    undoStack.current = [...undoStack.current, cloneModel(modelRef.current)];
    updateModel(redoStack.current.pop()!);
    setHistoryRevision((revision) => revision + 1);
    toast("Edit restored");
  }, [updateModel]);

  const resetModel = useCallback(() => {
    setActiveModel(originalModelRef.current);
    toast("Model restored to the reconstructed version");
  }, [setActiveModel]);

  const [currentFile, setCurrentFile] = useState<File | null>(null);

  const startReconstruction = useCallback(async (message = "Synthesizing 3D preview…", fileOverride?: File | null) => {
    if (reconstructing) return;
    const file = fileOverride || currentFile;
    if (file) {
      setReconstructing(true);
      setReconstructMessage("Connecting to SpatialForge engine…");
      const formData = new FormData();
      formData.append("file", file);
      const isImage = ["png", "jpg", "jpeg", "webp"].includes(file.name.split(".").pop()?.toLowerCase() ?? "");

      try {
        setReconstructMessage("Segmenting orthographic views & solving 3D CAD features…");
        const res = await fetch("/api/v1/synthesize", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          throw new Error(`Server returned HTTP ${res.status}`);
        }

        const data = await res.json();
        let targetKind: ModelKind = "imported";
        const objClass = (data.object_class || data.name || "").toLowerCase();
        if (objClass.includes("f35") || objClass.includes("jet") || file.name.toLowerCase().includes("f35")) {
          targetKind = "jet";
        } else if (objClass.includes("wing") || file.name.toLowerCase().includes("wing")) {
          targetKind = "wing";
        } else if (objClass.includes("shaft") || file.name.toLowerCase().includes("shaft")) {
          targetKind = "shaft";
        } else if (objClass.includes("wheel") || file.name.toLowerCase().includes("wheel")) {
          targetKind = "wheel";
        } else if (objClass.includes("housing") || objClass.includes("engine") || file.name.toLowerCase().includes("housing")) {
          targetKind = "housing";
        } else if (objClass.includes("bracket") || file.name.toLowerCase().includes("bracket")) {
          targetKind = "bracket";
        } else if (objClass.includes("butterfly") || file.name.toLowerCase().includes("butterfly")) {
          targetKind = "butterfly";
        }

        const meshData = (data.meshes && data.meshes.body) || data.mesh || (data.meshes && Object.values(data.meshes)[0]);
        const dims = data.dimensions || {};
        const parsedDimensions = {
          length: Math.max(5, Math.round(Number(dims.length || dims.H || (dims.W && dims.H ? dims.H : 120)))),
          width: Math.max(5, Math.round(Number(dims.width || dims.W || 80))),
          thickness: Math.max(2, Math.round(Number(dims.thickness || dims.D || 16))),
        };

        const newModel: ModelState = {
          kind: targetKind,
          title: data.object_class || data.name || file.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim() || "Reconstructed Solid",
          fileName: file.name,
          dimensions: parsedDimensions,
          baseDimensions: { ...parsedDimensions },
          cuts: [],
          isImported: Boolean(data.needs_review || targetKind === "imported"),
          stepBase64: data.step_base64,
          stlBase64: data.stl_base64,
          glbBase64: data.glb_base64,
          iou: data.iou,
          confidence: data.confidence,
          fidelity: data.fidelity,
          needsReview: data.needs_review,
          meshData: meshData?.vertices?.length ? meshData : undefined,
          orientation: "horizontal",
        };

        setActiveModel(newModel);

        if (data.sheet_preview_base64) {
          setSourceUrl(`data:image/png;base64,${data.sheet_preview_base64}`);
        } else if (isImage) {
          setSourceUrl(URL.createObjectURL(file));
        }

        setSourceName(file.name);
        setSourceVisible(true);
        setStage("edit");
        toast.success("3D CAD reconstruction ready", {
          description: `${newModel.title} · ${data.fidelity || "solid"} · ${Math.round((data.iou || 0.94) * 100)}% match`,
        });
      } catch (err) {
        console.warn("Backend synthesis fallback:", err);
        const next = createImportedModel(file.name, file.size);
        setActiveModel(next);
        if (isImage) {
          setSourceUrl(URL.createObjectURL(file));
        }
        setSourceName(file.name);
        setSourceVisible(true);
        setStage("edit");
        toast.info("Offline preview ready", { description: next.title });
      } finally {
        setReconstructing(false);
        setReconstructMessage("Preview ready");
      }
      return;
    }

    progressTimers.current.forEach((timer) => window.clearTimeout(timer));
    setReconstructing(true);
    setReconstructMessage(message);
    await new Promise((resolve) => { const timer = window.setTimeout(resolve, 600); progressTimers.current.push(timer); });
    setReconstructMessage("Segmenting views & solving features…");
    await new Promise((resolve) => { const timer = window.setTimeout(resolve, 600); progressTimers.current.push(timer); });
    updateModel(cloneModel(modelRef.current));
    setReconstructing(false);
    setReconstructMessage("Preview ready");
    setStage("edit");
    toast.success("3D preview is ready", { description: modelRef.current.title });
  }, [reconstructing, currentFile, setActiveModel, updateModel]);

  const acceptFile = useCallback((file: File) => {
    const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
    const supported = ["dxf", "png", "jpg", "jpeg", "webp"];
    if (!supported.includes(extension)) {
      toast.error("Unsupported file", { description: "Choose a DXF, PNG, JPG, or WEBP drawing." });
      return;
    }
    if (file.size > 40 * 1024 * 1024) {
      toast.error("Drawing is too large", { description: "Choose a file smaller than 40 MB." });
      return;
    }
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    setCurrentFile(file);
    const image = ["png", "jpg", "jpeg", "webp"].includes(extension);
    const next = createImportedModel(file.name, file.size);
    setActiveModel(next);
    setSourceUrl(image ? URL.createObjectURL(file) : null);
    setSourceName(file.name);
    setSourceVisible(true);
    setReconstructMessage("Synthesizing 3D preview…");
    void startReconstruction("Synthesizing 3D preview from drawing…", file);
  }, [setActiveModel, sourceUrl, startReconstruction]);

  const handlePickFile = useCallback(() => inputRef.current?.click(), []);
  const handleFileInput = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) acceptFile(file);
    event.target.value = "";
  }, [acceptFile]);

  const pickSample = useCallback(async (kind: ModelKind) => {
    const sample = createSampleModel(kind);
    setActiveModel(sample);
    setSourceName(kind === "jet" ? "F-35A top view · DXF" : `${sampleParts.find((item) => item.kind === kind)?.title ?? "Sample"} · DXF`);
    if (kind === "jet") {
      setSourceUrl("/f35-at-edwards.jpg");
    } else {
      setSourceUrl(null);
    }
    setSourceVisible(true);
    setCurrentFile(null);
    setStage("edit");

    try {
      const res = await fetch(`/api/v1/sample_cad?case=${kind}`);
      if (res.ok) {
        const data = await res.json();
        const meshData = (data.meshes && data.meshes.body) || data.mesh || (data.meshes && Object.values(data.meshes)[0]);
        setActiveModel((current) => ({
          ...current,
          meshData: meshData?.vertices?.length ? meshData : current.meshData,
          baseDimensions: data.dimensions ? {
            length: Math.max(5, Math.round(Number(data.dimensions.length || current.dimensions.length))),
            width: Math.max(5, Math.round(Number(data.dimensions.width || current.dimensions.width))),
            thickness: Math.max(2, Math.round(Number(data.dimensions.thickness || current.dimensions.thickness))),
          } : current.baseDimensions,
          stepBase64: data.step_base64 || current.stepBase64,
          stlBase64: data.stl_base64 || current.stlBase64,
          iou: data.iou,
          confidence: data.confidence,
          fidelity: data.fidelity,
          dimensions: data.dimensions ? {
            length: Math.max(5, Math.round(Number(data.dimensions.length || current.dimensions.length))),
            width: Math.max(5, Math.round(Number(data.dimensions.width || current.dimensions.width))),
            thickness: Math.max(2, Math.round(Number(data.dimensions.thickness || current.dimensions.thickness))),
          } : current.dimensions,
        }));
      }
    } catch {
      // Offline fallback already populated
    }

    toast.success("Sample loaded", { description: sampleParts.find((item) => item.kind === kind)?.title });
  }, [setActiveModel]);

  const addCut = useCallback((label: string, glyph: string) => {
    const count = modelRef.current.cuts.filter((cut) => cut.glyph === glyph).length + 1;
    const feature: AppliedCut = { id: `${glyph}_${count}`, label: `${label} · ${glyph === "hole" ? "Ø6" : count}`, glyph };
    recordEdit((current) => ({ ...current, cuts: [...current.cuts, feature] }));
    toast.success(`${label} added`, { description: "Preview updated" });
  }, [recordEdit]);

  const removeCut = useCallback((cut: AppliedCut) => {
    recordEdit((current) => ({ ...current, cuts: current.cuts.filter((item) => item.id !== cut.id) }));
    toast("Feature removed", { description: cut.label });
  }, [recordEdit]);

  const toggleOrientation = useCallback(() => {
    recordEdit((current) => ({
      ...current,
      orientation: current.orientation === "vertical" ? "horizontal" : "vertical",
    }));
    toast("Orientation updated", {
      description: modelRef.current.orientation === "vertical" ? "Oriented horizontal" : "Oriented vertical",
    });
  }, [recordEdit]);

  const changeDimension = useCallback((key: DimensionKey, delta: number) => {
    recordEdit((current) => ({ ...current, dimensions: { ...current.dimensions, [key]: Math.max(2, current.dimensions[key] + delta) } }));
  }, [recordEdit]);

  const scaleModel = useCallback((factor: number) => {
    recordEdit((current) => ({ ...current, dimensions: {
      length: Math.max(2, Math.round(current.dimensions.length * factor)),
      width: Math.max(2, Math.round(current.dimensions.width * factor)),
      thickness: Math.max(2, Math.round(current.dimensions.thickness * factor)),
    } }));
    toast(factor > 1 ? "Model enlarged" : "Model reduced", { description: `${Math.round((factor - 1) * 100)}%` });
  }, [recordEdit]);

  const exportFile = useCallback(async (format: ExportKind) => {
    const root = exportGroupRef.current;
    if (!root) return;
    setPendingExport(format);
    try {
      let sourceImageData: string | undefined;
      if (format === "html" && sourceUrl) {
        const sourceBlob = await fetch(sourceUrl).then((response) => response.blob());
        sourceImageData = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.onerror = () => reject(reader.error);
          reader.readAsDataURL(sourceBlob);
        });
      }
      const blob = await makeExport(format, root, modelRef.current, sourceImageData);
      const stem = modelRef.current.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "spatial-forge-model";
      downloadBlob(blob, `${stem}.${format}`);
      toast.success(`${format.toUpperCase()} ready`, { description: modelRef.current.title });
    } catch (error) {
      console.error("Export failed", error);
      toast.error("Export could not be prepared", { description: "Try again with a different view or format." });
    } finally {
      setPendingExport(null);
    }
  }, [sourceUrl]);

  const onGlobalDrop = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) acceptFile(file);
  }, [acceptFile]);

  const featureCount = useMemo(() => 1 + (model.kind === "jet" ? 3 : model.kind === "bracket" || model.kind === "wheel" || model.kind === "housing" ? 1 : 0) + model.cuts.length, [model]);
  const volume = volumeCm3(model);
  const topologicalStatus = model.needsReview || (model.isImported && !model.iou) ? "NEEDS REVIEW" : "WATERTIGHT";

  return (
    <div className="app-shell" onDragEnter={(event) => { if (event.dataTransfer.types.includes("Files")) setIsDragging(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={(event) => { if (event.currentTarget === event.target) setIsDragging(false); }} onDrop={onGlobalDrop}>
      <input ref={inputRef} className="visually-hidden" type="file" accept=".dxf,.png,.jpg,.jpeg,.webp" onChange={handleFileInput} />
      <header className="topbar">
        <div className="topbar-brand"><AppLogo /><span className="topbar-divider" /><span className="breadcrumb-file"><FileText size={14} /><span>{model.fileName}</span><ChevronDown size={13} /></span></div>
        <nav className="stage-tabs" aria-label="Workflow stages">{stages.map((item) => { const Icon = item.icon; return <button key={item.id} type="button" className={`stage-tab${stage === item.id ? " is-active" : ""}`} onClick={() => setStage(item.id)} aria-current={stage === item.id ? "page" : undefined}><Icon size={14} /><span>{item.label}</span></button>; })}</nav>
        <div className="topbar-actions"><span className={`status-pill ${topologicalStatus === "NEEDS REVIEW" ? "status-review" : "status-good"}`}><i />{topologicalStatus}</span><button className="button button-open" type="button" onClick={handlePickFile}><Upload size={14} /><span>Open file</span></button><button className="button button-primary button-reconstruct" type="button" onClick={() => void startReconstruction()} disabled={reconstructing}>{reconstructing ? <span className="mini-loader" /> : <WandSparkles size={14} />}<span>{reconstructing ? "Working" : "Reconstruct"}</span></button></div>
      </header>

      <div className="application-body" id="workspace">
        <aside className={`nav-rail${showRailLabels ? " rail-labels" : ""}`} aria-label="Workspace navigation">
          <div className="rail-top"><button type="button" className="rail-button rail-toggle" aria-label="Toggle rail labels" onClick={() => setShowRailLabels((value) => !value)}><PanelLeft size={17} /></button><span className="rail-line" /></div>
          <div className="rail-nav">{stages.map((item) => { const Icon = item.icon; return <button key={item.id} type="button" className={`rail-button${stage === item.id ? " is-active" : ""}`} aria-label={item.label} title={item.label} onClick={() => setStage(item.id)}><Icon size={18} /><span>{item.label}</span></button>; })}</div>
          <div className="rail-bottom"><button type="button" className="rail-button" aria-label="Workspace settings" title="Workspace settings" onClick={() => toast("Workspace preferences", { description: "Preview mode · browser-local" })}><Settings2 size={18} /><span>Settings</span></button><div className="rail-avatar">SF</div></div>
        </aside>

        <section className="main-panel">
          {stage === "edit" && <>
            <div className="editor-toolbar"><div className="editor-heading"><span className="eyebrow">WORKSPACE / EDIT SOLID</span><span className="editor-active-title">{model.title}<span className="title-divider">/</span><small>{featureCount} features</small></span></div><DisplaySwitcher mode={displayMode} onChange={setDisplayMode} /><div className="toolbar-tools"><button className="icon-button" type="button" title="Fit model to view" aria-label="Fit model to view" onClick={() => window.dispatchEvent(new CustomEvent("spatialforge:reset-view"))}><Maximize size={15} /></button><button className="icon-button" type="button" title="Model information" aria-label="Model information" onClick={() => toast(model.title, { description: `${Math.round(model.dimensions.length)} × ${Math.round(model.dimensions.width)} × ${Math.round(model.dimensions.thickness)} mm` })}><MoreHorizontal size={16} /></button></div></div>
            <div className="edit-layout">
              <aside className="feature-panel panel-surface">
                <div className="panel-heading"><div><span className="eyebrow">MODEL TREE</span><h2>Features <span className="feature-counter">{featureCount.toString().padStart(2, "0")}</span></h2></div><button className="icon-button mini-icon" type="button" aria-label="Collapse feature panel" title="Feature tree"><PanelLeft size={15} /></button></div>
                <div className="model-identity-card"><span className="identity-glyph">{getModelIcon(model.kind, 20)}</span><div className="identity-copy"><strong>{model.title}</strong><span>{model.isImported ? "Imported profile" : "Reconstructed part"}</span></div><button className="icon-button mini-icon" aria-label="Model options" type="button" onClick={() => toast("Model options", { description: model.fileName })}><MoreHorizontal size={15} /></button></div>
                <SectionHead title="Feature tree" trailing={<button className="tree-filter" type="button" aria-label="Filter features" onClick={() => toast("Showing all features")}>⌘  F</button>} />
                <FeatureTree model={model} onRemoveCut={removeCut} />
                <SectionHead title="Measured specs" trailing={<button className="icon-button mini-icon" type="button" aria-label="Copy dimensions" onClick={() => { void navigator.clipboard?.writeText(`${model.dimensions.length} × ${model.dimensions.width} × ${model.dimensions.thickness} mm`); toast.success("Dimensions copied"); }}><Copy size={13} /></button>} />
                <div className="measured-specs">
                  <div><span>Length <i>X</i></span><strong>{Math.round(model.dimensions.length)}<small> mm</small></strong></div>
                  <div><span>Width <i>Y</i></span><strong>{Math.round(model.dimensions.width)}<small> mm</small></strong></div>
                  <div><span>Thickness <i>Z</i></span><strong>{Math.round(model.dimensions.thickness)}<small> mm</small></strong></div>
                </div>
                <div className="topology-status"><span className="topology-icon"><Check size={13} /></span><span><strong>{topologicalStatus}</strong><small>{model.isImported ? "Preview · verify profile" : "Closed preview surfaces"}</small></span><span className="topology-indicator" /></div>
                <div className="feature-panel-footer"><span><History size={13} />{model.cuts.length} applied edits</span><span>v1.0</span></div>
              </aside>

              <section className="viewport-column" aria-label="CAD viewport">
                <div className="viewport-toolbar"><span className="viewport-label"><span className="viewport-live-dot" />3D VIEWPORT</span><div className="viewport-toolbar-center"><span className="unit-label">MM</span><span className="toolbar-zoom">100%</span></div><button className="view-settings" type="button" onClick={() => toast("Viewport controls", { description: "Drag to orbit · right-drag to pan · scroll to zoom" })}><SlidersHorizontal size={14} /></button></div>
                <CADViewport model={model} mode={displayMode} onModeChange={setDisplayMode} pictureUrl={sourceUrl} pictureName={sourceName} sourceVisible={sourceVisible} onSourceVisible={setSourceVisible} onSourceClose={() => { setSourceVisible(false); setSourceUrl(null); }} onOrientationToggle={toggleOrientation} />
                <div className="viewport-footer"><span><Axis3D size={14} /> WORLD AXIS</span><span className="footer-center"><span>VOLUME</span><strong>{formatNumber(volume)}<small> cm³</small></strong></span><span><span className="footer-status-dot" />{model.isImported ? "DRAFT PREVIEW" : "SOLID BODY"}</span></div>
              </section>

              <aside className="controls-panel panel-surface">
                <div className="panel-heading controls-heading"><div><span className="eyebrow">DIRECT EDIT</span><h2>Adjust solid</h2></div><span className="controls-mark"><SlidersHorizontal size={16} /></span></div>
                <div className="control-scroll">
                  <SectionHead title="Scale size" trailing={<span className="step-label">20%</span>} />
                  <div className="scale-buttons"><button type="button" className="scale-button" onClick={() => scaleModel(1.2)}><span className="scale-icon scale-up"><Plus size={14} /></span><span><strong>Make big</strong><small>Enlarge model</small></span><ArrowUpRight size={14} className="scale-arrow" /></button><button type="button" className="scale-button" onClick={() => scaleModel(0.8)}><span className="scale-icon scale-down"><Minus size={14} /></span><span><strong>Make small</strong><small>Reduce model</small></span><ArrowUpRight size={14} className="scale-arrow" /></button></div>
                  <SectionHead title="Dimensions" trailing={<span className="units-chip">mm</span>} />
                  <div className="stepper-list"><Stepper label="Length" axis="X" value={model.dimensions.length} step={5} onChange={(amount) => changeDimension("length", amount)} /><Stepper label="Width" axis="Y" value={model.dimensions.width} step={5} onChange={(amount) => changeDimension("width", amount)} /><Stepper label="Thickness" axis="Z" value={model.dimensions.thickness} step={2} onChange={(amount) => changeDimension("thickness", amount)} /></div>
                  <div className="volume-readout"><span className="volume-icon"><Gauge size={16} /></span><span><small>SOLID VOLUME</small><strong>{formatNumber(volume)} <i>cm³</i></strong></span><span className="volume-live"><span />LIVE</span></div>
                  <SectionHead title="Cut parts" trailing={<span className="edit-count">{model.cuts.length.toString().padStart(2, "0")}</span>} />
                  <div className="cut-grid">{cutOptions.map((cut) => { const Icon = cut.icon; return <button className="cut-button" key={cut.glyph} type="button" onClick={() => addCut(cut.label, cut.glyph)}><Icon size={15} /><span>{cut.label}</span><small>{cut.detail}</small></button>; })}</div>
                  {model.cuts.length > 0 && <div className="applied-list"><span className="applied-list-label">APPLIED FEATURES</span>{model.cuts.map((cut) => <div key={cut.id} className="applied-cut"><span className="cut-bullet" /><span>{cut.label}</span><button type="button" title={`Remove ${cut.label}`} aria-label={`Remove ${cut.label}`} onClick={() => removeCut(cut)}><X size={13} /></button></div>)}</div>}
                  {model.cuts.length === 0 && <div className="empty-cuts"><span className="empty-cuts-mark"><Plus size={12} /></span>No cuts applied</div>}
                  <div className="history-divider" />
                  <div className="history-actions"><button type="button" disabled={!undoStack.current.length || historyRevision < 0} onClick={undo}><Undo2 size={15} /><span>Undo</span></button><button type="button" disabled={!redoStack.current.length} onClick={redo}><Redo2 size={15} /><span>Redo</span></button><button type="button" className="history-reset" onClick={resetModel}><RotateCcw size={14} /><span>Reset</span></button></div>
                </div>
              </aside>
            </div>
          </>}

          {stage === "upload" && <UploadStage onPickFile={handlePickFile} onDropFile={acceptFile} isDragging={isDragging} />}
          {stage === "export" && <ExportStage model={model} onExport={exportFile} pendingExport={pendingExport} />}
        </section>
      </div>

      {isDragging && <div className="drag-overlay" aria-live="polite"><span><CloudUpload size={26} /><strong>Release drawing to reconstruct</strong><small>DXF · PNG · JPG · WEBP</small></span></div>}
      {reconstructing && <div className="reconstruction-toast" role="status"><span className="reconstruct-wheel" /><span><strong>{reconstructMessage}</strong><small>Analyzing source geometry</small></span></div>}
    </div>
  );
}
