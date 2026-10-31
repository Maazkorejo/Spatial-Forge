import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { useEffect, useRef, useState } from "react";
import { Download, Eye, EyeOff, FileText, Maximize2, Minus, RotateCcw, X, ZoomIn, ZoomOut } from "lucide-react";
import { createPartGroup, disposePartGroup } from "./geometry";
import type { DisplayMode, ModelState } from "./types";

const viewModes: Array<{ label: string; value: DisplayMode }> = [
  { label: "Shaded", value: "shaded" },
  { label: "Wireframe", value: "wireframe" },
  { label: "Solid", value: "solid" },
];

type CADViewportProps = {
  model: ModelState;
  mode: DisplayMode;
  onModeChange: (mode: DisplayMode) => void;
  pictureUrl: string | null;
  pictureName: string;
  sourceVisible: boolean;
  onSourceVisible: (visible: boolean) => void;
  onSourceClose: () => void;
  onOrientationToggle?: () => void;
};

type SceneHandle = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  partGroup: THREE.Group | null;
  resetView: () => void;
};

function BlueprintSketch({ model }: { model: ModelState }) {
  return (
    <svg className="blueprint-sketch" viewBox="0 0 240 132" aria-label={`${model.title} top-view drawing`}>
      <defs>
        <pattern id="blueprint-grid" width="12" height="12" patternUnits="userSpaceOnUse">
          <path d="M 12 0 L 0 0 0 12" fill="none" stroke="currentColor" strokeWidth=".5" opacity=".34" />
        </pattern>
      </defs>
      <rect x="0" y="0" width="240" height="132" fill="url(#blueprint-grid)" />
      <g fill="none" stroke="currentColor" strokeWidth="1.3" opacity=".92" strokeLinejoin="round">
        <path d="M120 14 L133 48 L191 70 L196 80 L142 73 L132 108 L120 103 L108 108 L98 73 L44 80 L49 70 L107 48 Z" />
        <path d="M120 18 L120 102 M64 77 L176 77" strokeDasharray="3 3" opacity=".55" />
        <path d="M29 21 L29 111 M25 21 L33 21 M25 111 L33 111 M29 18 L29 24 M29 108 L29 114" strokeWidth=".8" />
        <path d="M204 93 L204 112 M200 93 L208 93 M200 112 L208 112" strokeWidth=".8" />
      </g>
      <g fill="currentColor" opacity=".72" fontFamily="monospace" fontSize="7">
        <text x="14" y="68" transform="rotate(-90 14 68)">L {Math.round(model.dimensions.length)} mm</text>
        <text x="170" y="123">W {Math.round(model.dimensions.width)} mm</text>
        <text x="198" y="107">TOP</text>
      </g>
    </svg>
  );
}

function disposeScene(handle: SceneHandle) {
  if (handle.partGroup) disposePartGroup(handle.partGroup);
  handle.controls.dispose();
  handle.renderer.dispose();
  handle.renderer.domElement.remove();
}

export function CADViewport({ model, mode, onModeChange, pictureUrl, pictureName, sourceVisible, onSourceVisible, onSourceClose, onOrientationToggle }: CADViewportProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneHandleRef = useRef<SceneHandle | null>(null);
  const resetRef = useRef<() => void>(() => undefined);
  const modelTitleRef = useRef<string | null>(null);
  const [full2DOpen, setFull2DOpen] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && full2DOpen) {
        setFull2DOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [full2DOpen]);

  useEffect(() => {
    const resetFromToolbar = () => resetRef.current();
    window.addEventListener("spatialforge:reset-view", resetFromToolbar);
    return () => window.removeEventListener("spatialforge:reset-view", resetFromToolbar);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#14181e");
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 3000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.16;
    renderer.domElement.setAttribute("aria-label", "Interactive three-dimensional CAD model; drag to orbit, right-drag to pan, and scroll to zoom");
    container.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight("#f9fbff", "#6e7f98", 2.0));
    const keyLight = new THREE.DirectionalLight("#ffffff", 3.2);
    keyLight.position.set(100, 180, 85);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight("#c0d5ff", 1.2);
    fillLight.position.set(-130, 80, -70);
    scene.add(fillLight);

    const grid = new THREE.GridHelper(240, 48, "#465268", "#2a313c");
    grid.position.y = -0.02;
    grid.material.transparent = true;
    grid.material.opacity = 0.65;
    scene.add(grid);
    const axisX = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-120, 0.015, 0), new THREE.Vector3(120, 0.015, 0)]),
      new THREE.LineBasicMaterial({ color: "#889bb5", transparent: true, opacity: 0.32 }),
    );
    const axisZ = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.015, -120), new THREE.Vector3(0, 0.015, 120)]),
      new THREE.LineBasicMaterial({ color: "#889bb5", transparent: true, opacity: 0.32 }),
    );
    scene.add(axisX, axisZ);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.rotateSpeed = 0.62;
    controls.zoomSpeed = 0.85;
    controls.panSpeed = 0.72;
    controls.minDistance = 30;
    controls.maxDistance = 1100;
    controls.maxPolarAngle = Math.PI * 0.49;
    const resetView = () => {
      const state = sceneHandleRef.current;
      if (!state) return;
      const dims = state.partGroup?.userData.dimensions ?? { length: 120, width: 80, thickness: 24 };
      const span = Math.max(dims.length, dims.width, dims.thickness);
      state.camera.position.set(span * 1.25, span * 1.45, span * 1.5);
      state.controls.target.set(0, dims.thickness * 0.55, 0);
      state.controls.update();
    };
    const handle: SceneHandle = { scene, camera, renderer, controls, partGroup: null, resetView };
    sceneHandleRef.current = handle;
    resetRef.current = resetView;

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (!width || !height) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();

    let frameId = 0;
    const renderFrame = () => {
      controls.update();
      renderer.render(scene, camera);
      frameId = window.requestAnimationFrame(renderFrame);
    };
    renderFrame();

    return () => {
      window.cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      scene.traverse((object) => {
        if (object instanceof THREE.Line && !(object instanceof THREE.LineSegments)) {
          object.geometry.dispose();
          if (Array.isArray(object.material)) object.material.forEach((item) => item.dispose());
          else object.material.dispose();
        }
      });
      disposeScene(handle);
      sceneHandleRef.current = null;
    };
  }, []);

  useEffect(() => {
    const handle = sceneHandleRef.current;
    if (!handle) return;
    if (handle.partGroup) {
      handle.scene.remove(handle.partGroup);
      disposePartGroup(handle.partGroup);
    }
    const nextGroup = createPartGroup(model, mode);
    handle.partGroup = nextGroup;
    handle.scene.add(nextGroup);
    if (modelTitleRef.current !== model.title) handle.resetView();
    modelTitleRef.current = model.title;
  }, [model, mode]);

  return (
    <div className="viewport-stage">
      <div className="viewport-canvas" ref={containerRef} />
      <div className="viewport-corner viewport-corner-tl"><span className="corner-label">PERSPECTIVE</span><span className="corner-dot" /> <span>1 : 1</span></div>
      <div className="viewport-axis" aria-hidden="true"><span className="axis-y">Y</span><span className="axis-x">X</span><span className="axis-z">Z</span><i /></div>
      <div className="viewport-tools-hint"><span className="hint-key">LMB</span> orbit <span className="hint-divider" /><span className="hint-key">RMB</span> pan <span className="hint-divider" /><span className="hint-key">⌘</span> zoom</div>
      <button className="viewport-reset icon-button" type="button" aria-label="Reset view" title="Reset view" onClick={() => resetRef.current()}><RotateCcw size={15} /></button>

      {pictureUrl && (
        <button
          className="full-2d-pill"
          type="button"
          title="Open full 2D CAD technical drawing sheet"
          aria-label="Open full 2D CAD technical drawing sheet"
          onClick={() => setFull2DOpen(true)}
          style={{ position: "absolute", top: 10, left: 140, zIndex: 3 }}
        >
          <FileText size={12} />
          <span>Full 2D CAD</span>
        </button>
      )}

      <button
        className="orientation-pill"
        type="button"
        title="Toggle Horizontal / Vertical orientation (AI automatic default: Horizontal)"
        aria-label="Toggle orientation"
        onClick={onOrientationToggle}
        style={{ position: "absolute", top: 10, left: pictureUrl ? 245 : 140, zIndex: 3 }}
      >
        <span className="orientation-dot" />
        <span>{model.orientation === "vertical" ? "Vertical" : "Horizontal"}</span>
      </button>

      {sourceVisible ? (
        <section className="blueprint-pip" aria-label="Source drawing picture in picture">
          <header className="blueprint-pip-header">
            <span><span className="pip-status" />2D DRAWING</span>
            <div>
              <button
                className="pip-expand-btn"
                type="button"
                title="View full 2D CAD drawing"
                aria-label="View full 2D CAD drawing"
                onClick={() => setFull2DOpen(true)}
              >
                <Maximize2 size={13} />
              </button>
              <button className="pip-tool" type="button" title="Minimize drawing" aria-label="Minimize drawing" onClick={() => onSourceVisible(false)}><Minus size={13} /></button>
              <button className="pip-tool" type="button" title="Close drawing" aria-label="Close drawing" onClick={onSourceClose}><X size={13} /></button>
            </div>
          </header>
          <div
            className="pip-source-clickable"
            title="Click to view full-resolution 2D CAD technical drawing"
            onClick={() => setFull2DOpen(true)}
          >
            {pictureUrl ? <img className="pip-source-image" src={pictureUrl} alt={`Uploaded source drawing ${pictureName}`} /> : <BlueprintSketch model={model} />}
          </div>
          <footer className="blueprint-pip-footer">
            <span>{pictureName}</span>
            <span style={{ cursor: "pointer", textDecoration: "underline" }} onClick={() => setFull2DOpen(true)}>View full</span>
          </footer>
        </section>
      ) : (
        <button className="show-drawing-pill" type="button" onClick={() => onSourceVisible(true)}><Eye size={14} /> Show 2D drawing</button>
      )}

      {full2DOpen && (
        <div className="blueprint-modal-backdrop" onClick={() => setFull2DOpen(false)}>
          <div className="blueprint-modal" onClick={(e) => e.stopPropagation()}>
            <header className="blueprint-modal-header">
              <div className="blueprint-modal-title">
                <FileText size={16} />
                <span>2D CAD TECHNICAL DRAWING</span>
                <span className="badge">{pictureName}</span>
                <span className="badge">{Math.round(model.dimensions.length)} × {Math.round(model.dimensions.width)} mm</span>
              </div>
              <div className="blueprint-modal-tools">
                <button
                  type="button"
                  className="blueprint-modal-tool"
                  title="Zoom in (+)"
                  aria-label="Zoom in"
                  onClick={() => setZoomLevel((z) => Math.min(3.5, Number((z + 0.25).toFixed(2))))}
                >
                  <ZoomIn size={15} />
                </button>
                <button
                  type="button"
                  className="blueprint-modal-tool"
                  title="Zoom out (-)"
                  aria-label="Zoom out"
                  onClick={() => setZoomLevel((z) => Math.max(0.5, Number((z - 0.25).toFixed(2))))}
                >
                  <ZoomOut size={15} />
                </button>
                <button
                  type="button"
                  className="blueprint-modal-tool"
                  title="Reset zoom (1:1)"
                  aria-label="Reset zoom"
                  onClick={() => setZoomLevel(1)}
                >
                  <RotateCcw size={14} />
                </button>
                {pictureUrl && (
                  <a
                    className="blueprint-modal-tool"
                    title="Download 2D CAD image"
                    aria-label="Download 2D CAD image"
                    href={pictureUrl}
                    download={pictureName || "technical-drawing.png"}
                  >
                    <Download size={14} />
                  </a>
                )}
                <button
                  type="button"
                  className="blueprint-modal-tool"
                  title="Close (Esc)"
                  aria-label="Close modal"
                  onClick={() => setFull2DOpen(false)}
                >
                  <X size={16} />
                </button>
              </div>
            </header>
            <div className="blueprint-modal-body">
              {pictureUrl ? (
                <img
                  className="blueprint-modal-image"
                  src={pictureUrl}
                  alt={`Full technical drawing ${pictureName}`}
                  style={{ transform: `scale(${zoomLevel})` }}
                />
              ) : (
                <div style={{ width: "90%", maxWidth: 880, transform: `scale(${zoomLevel})` }}>
                  <BlueprintSketch model={model} />
                </div>
              )}
            </div>
            <footer className="blueprint-modal-footer">
              <span>Zoom: {Math.round(zoomLevel * 100)}% · Use scrollbars or trackpad to pan</span>
              <span>SpatialForge Blueprint Inspector · 1st / 3rd Angle Orthographic</span>
            </footer>
          </div>
        </div>
      )}

      <button className="viewport-fullscreen icon-button" type="button" title="Focus viewport" aria-label="Focus viewport" onClick={() => containerRef.current?.requestFullscreen?.()}><Maximize2 size={15} /></button>
      {pictureUrl && <span className="viewport-source-indicator"><EyeOff size={12} /> Local source</span>}
    </div>
  );
}

export function DisplaySwitcher({ mode, onChange }: { mode: DisplayMode; onChange: (mode: DisplayMode) => void }) {
  return <div className="display-switcher" role="group" aria-label="Viewport rendering mode">{viewModes.map((item) => <button key={item.value} className={mode === item.value ? "display-mode is-active" : "display-mode"} type="button" onClick={() => onChange(item.value)}>{item.label}</button>)}</div>;
}
