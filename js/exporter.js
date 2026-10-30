/**
 * SpatialForge - Exporter Module
 * Handles file downloads (STEP, STL, GLB, DXF) and generation of standalone interactive 3D HTML models.
 */

export function downloadBase64(base64Data, filename, mimeType) {
  const byteChars = atob(base64Data);
  const byteNums = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) {
    byteNums[i] = byteChars.charCodeAt(i);
  }
  const blob = new Blob([new Uint8Array(byteNums)], { type: mimeType });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

export function extractSceneMeshPayload() {
  const meshList = [];
  const cadRootGroup = window.cadRootGroup;
  if (!cadRootGroup) return meshList;

  cadRootGroup.updateMatrixWorld(true);
  cadRootGroup.traverse(child => {
    if (child.isMesh && child.geometry) {
      const geom = child.geometry;
      const posAttr = geom.getAttribute('position');
      if (!posAttr) return;

      const positions = [];
      const worldMatrix = child.matrixWorld;
      const v = new THREE.Vector3();

      for (let i = 0; i < posAttr.count; i++) {
        v.fromBufferAttribute(posAttr, i);
        v.applyMatrix4(worldMatrix);
        positions.push(Math.round(v.x * 1000) / 1000, Math.round(v.y * 1000) / 1000, Math.round(v.z * 1000) / 1000);
      }

      meshList.push({
        positions: positions,
        color: child.userData && child.userData.isCutMesh ? 0x181a20 : 0x94a3b8,
        roughness: child.userData && child.userData.isCutMesh ? 0.9 : 0.35,
        metalness: child.userData && child.userData.isCutMesh ? 0.1 : 0.15,
        isCut: !!(child.userData && child.userData.isCutMesh)
      });
    }
  });

  return meshList;
}

export function generateStandaloneViewerHtml(data) {
  const meshesJson = JSON.stringify(data.meshes);
  const hasPreview = data.previewSrc && data.previewSrc.length > 20;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${data.title} - Spatial Forge 3D</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><` + `/script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"><` + `/script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #090a0d;
      color: #edeef0;
      font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    #canvas3d { width: 100%; height: 100%; display: block; }
    .badge-bar {
      position: absolute;
      top: 18px;
      left: 20px;
      z-index: 10;
      display: flex;
      flex-direction: column;
      gap: 6px;
      pointer-events: none;
    }
    .brand-tag {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      background: rgba(18, 20, 26, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #94a3b8;
    }
    .brand-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #6f8dff;
      box-shadow: 0 0 8px rgba(111, 141, 255, 0.8);
    }
    .title-box {
      background: rgba(18, 20, 26, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 10px 14px;
      border-radius: 10px;
      max-width: 380px;
    }
    .model-title {
      font-size: 15px;
      font-weight: 700;
      color: #f8fafc;
      letter-spacing: -0.01em;
      margin-bottom: 3px;
    }
    .model-meta {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: #94a3b8;
    }
    .controls-bar {
      position: absolute;
      bottom: 22px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(18, 20, 26, 0.9);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 6px 12px;
      border-radius: 30px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    .btn {
      background: transparent;
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #cbd5e1;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .btn:hover {
      background: rgba(255, 255, 255, 0.08);
      color: #ffffff;
      border-color: rgba(255, 255, 255, 0.2);
    }
    .btn.active {
      background: rgba(111, 141, 255, 0.2);
      border-color: #6f8dff;
      color: #c7cdff;
    }
    .pip-container {
      position: absolute;
      top: 18px;
      right: 20px;
      z-index: 10;
      width: 260px;
      background: rgba(18, 20, 26, 0.92);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
      display: ${hasPreview ? 'flex' : 'none'};
      flex-direction: column;
      transition: width 0.2s ease, opacity 0.2s ease;
    }
    .pip-header {
      padding: 8px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      background: rgba(25, 28, 36, 0.6);
    }
    .pip-title-text {
      font-size: 11px;
      font-weight: 600;
      color: #94a3b8;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .pip-btn {
      background: transparent;
      border: none;
      color: #64748b;
      cursor: pointer;
      padding: 3px;
      display: flex;
      border-radius: 4px;
    }
    .pip-btn:hover { color: #f1f5f9; background: rgba(255, 255, 255, 0.08); }
    .pip-body {
      width: 100%;
      height: 180px;
      background: #0d0f14;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .pip-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .pip-pill {
      position: absolute;
      top: 18px;
      right: 20px;
      z-index: 10;
      display: none;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      background: rgba(18, 20, 26, 0.9);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      font-size: 11px;
      font-weight: 600;
      color: #94a3b8;
      cursor: pointer;
    }
    .pip-pill:hover { color: #ffffff; border-color: #6f8dff; }
  </style>
</head>
<body>
  <div class="badge-bar">
    <div class="brand-tag">
      <span class="brand-dot"></span>
      Constructed by Spatial Forge
    </div>
    <div class="title-box">
      <div class="model-title">${data.title}</div>
      <div class="model-meta">${data.info}</div>
    </div>
  </div>

  <div class="pip-container" id="drawingPip">
    <div class="pip-header">
      <div class="pip-title-text">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
        2D Input Drawing
      </div>
      <div style="display:flex;gap:4px;">
        <button class="pip-btn" onclick="minimizePip()" title="Minimize">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </button>
        <button class="pip-btn" onclick="closePip()" title="Close">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
    </div>
    <div class="pip-body">
      <img src="${data.previewSrc}" class="pip-img" alt="2D Drawing">
    </div>
  </div>

  <div class="pip-pill" id="drawingPill" onclick="restorePip()">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
    Blueprint
  </div>

  <div class="controls-bar">
    <button class="btn active" id="btnRotate" onclick="toggleAutoRotate()">Auto Rotate</button>
    <button class="btn" onclick="resetView()">Reset View</button>
  </div>

  <canvas id="canvas3d"></canvas>

  <script>
    const meshData = ${meshesJson};
    let scene, camera, renderer, controls;
    let sceneGroup = new THREE.Group();
    let isAutoRotating = true;
    let defaultCameraPos = new THREE.Vector3();
    let defaultTarget = new THREE.Vector3();

    function init() {
      const canvas = document.getElementById('canvas3d');
      renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.1;

      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 2000);

      const ambient = new THREE.AmbientLight(0xffffff, 0.65);
      scene.add(ambient);

      const hemi = new THREE.HemisphereLight(0xffffff, 0x1e2029, 0.7);
      scene.add(hemi);

      const dir1 = new THREE.DirectionalLight(0xffffff, 0.85);
      dir1.position.set(120, 180, 100);
      scene.add(dir1);

      const dir2 = new THREE.DirectionalLight(0x6f8dff, 0.35);
      dir2.position.set(-100, -50, -80);
      scene.add(dir2);

      const grid = new THREE.GridHelper(240, 48, 0x242838, 0x141620);
      grid.position.y = -0.01;
      scene.add(grid);

      scene.add(sceneGroup);

      meshData.forEach(part => {
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.Float32BufferAttribute(part.positions, 3));
        geom.computeVertexNormals();

        const mat = new THREE.MeshStandardMaterial({
          color: part.color,
          roughness: part.roughness,
          metalness: part.metalness,
          polygonOffset: true,
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1
        });

        const mesh = new THREE.Mesh(geom, mat);
        sceneGroup.add(mesh);

        if (!part.isCut) {
          const edgesGeom = new THREE.EdgesGeometry(geom, 22);
          const edgesLine = new THREE.LineSegments(edgesGeom, new THREE.LineBasicMaterial({ color: 0x0f172a }));
          sceneGroup.add(edgesLine);
        }
      });

      const box = new THREE.Box3().setFromObject(sceneGroup);
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);

      sceneGroup.position.x = -center.x;
      sceneGroup.position.z = -center.z;
      sceneGroup.position.y = -box.min.y;

      const maxDim = Math.max(size.x, size.y, size.z, 20);
      camera.position.set(maxDim * 1.5, maxDim * 1.3, maxDim * 1.8);
      defaultCameraPos.copy(camera.position);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 2.0;
      controls.target.set(0, size.y / 2, 0);
      defaultTarget.copy(controls.target);

      window.addEventListener('resize', onResize);
      animate();
    }

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }

    function onResize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }

    function toggleAutoRotate() {
      isAutoRotating = !isAutoRotating;
      controls.autoRotate = isAutoRotating;
      document.getElementById('btnRotate').classList.toggle('active', isAutoRotating);
    }

    function resetView() {
      camera.position.copy(defaultCameraPos);
      controls.target.copy(defaultTarget);
      controls.update();
    }

    function minimizePip() {
      document.getElementById('drawingPip').style.display = 'none';
      document.getElementById('drawingPill').style.display = 'flex';
    }
    function restorePip() {
      document.getElementById('drawingPip').style.display = 'flex';
      document.getElementById('drawingPill').style.display = 'none';
    }
    function closePip() {
      document.getElementById('drawingPip').style.display = 'none';
      document.getElementById('drawingPill').style.display = 'none';
    }

    window.toggleAutoRotate = toggleAutoRotate;
    window.resetView = resetView;
    window.minimizePip = minimizePip;
    window.restorePip = restorePip;
    window.closePip = closePip;

    window.addEventListener('DOMContentLoaded', init);
  <` + `/script>
</body>
</html>`;
}

export function downloadStandaloneHtml() {
  const filename = window.currentDxfFilename || "model.dxf";
  const cleanName = filename.replace(/\.[^/.]+$/, "");
  const modelTitle = cleanName.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

  const meshPayload = extractSceneMeshPayload();
  if (meshPayload.length === 0) {
    alert("No 3D model currently loaded to export.");
    return;
  }

  let previewSrc = "";
  if (window.currentPreviewBase64) {
    previewSrc = window.currentPreviewBase64.startsWith("data:") ? window.currentPreviewBase64 : `data:image/png;base64,${window.currentPreviewBase64}`;
  } else {
    const pipImg = document.getElementById('pipImage');
    if (pipImg && pipImg.src) previewSrc = pipImg.src;
  }

  const curDims = window.currentDimensions || { W: 40, D: 20, H: 10 };
  const curVol = window.currentVolume || 8000;
  const dimText = `${curDims.W} × ${curDims.D} × ${curDims.H} mm`;
  const volText = `${curVol.toLocaleString()} mm³`;
  const infoText = `Dimensions: ${dimText} · Solid Volume: ${volText}`;

  const htmlContent = generateStandaloneViewerHtml({
    title: modelTitle,
    filename: filename,
    info: infoText,
    meshes: meshPayload,
    previewSrc: previewSrc
  });

  const blob = new Blob([htmlContent], { type: "text/html;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${cleanName}_spatialforge_3d.html`;
  link.click();
  URL.revokeObjectURL(link.href);
}

export function downloadExport(fmt) {
  if (fmt === 'html') {
    downloadStandaloneHtml();
    return;
  }
  const curFn = window.currentDxfFilename || "model.dxf";
  const curSessId = window.cadEditSessionId || "default";

  if (fmt === 'step') {
    if (window.currentStepBase64) {
      downloadBase64(window.currentStepBase64, `${curFn.replace(/\.[^/.]+$/, "")}.step`, 'application/step');
    } else {
      window.location.href = `http://127.0.0.1:8085/api/v1/cad_edit/export/step?session_id=${curSessId}`;
    }
  } else if (fmt === 'stl') {
    if (window.currentStlBase64) {
      downloadBase64(window.currentStlBase64, `${curFn.replace(/\.[^/.]+$/, "")}.stl`, 'application/sla');
    } else {
      window.location.href = `http://127.0.0.1:8085/api/v1/cad_edit/export/stl?session_id=${curSessId}`;
    }
  } else if (fmt === 'dxf') {
    window.location.href = `http://127.0.0.1:8085/api/v1/download_dxf/${curFn}`;
  } else if (fmt === 'glb') {
    if (window.currentGlbBase64) {
      downloadBase64(window.currentGlbBase64, `${curFn.replace(/\.[^/.]+$/, "")}.glb`, 'model/gltf-binary');
    } else {
      alert("GLB asset available on DXF reconstruction.");
    }
  }
}

// Bind to window for HTML event handlers
window.downloadExport = downloadExport;
window.downloadStandaloneHtml = downloadStandaloneHtml;
window.downloadBase64 = downloadBase64;
window.extractSceneMeshPayload = extractSceneMeshPayload;
window.generateStandaloneViewerHtml = generateStandaloneViewerHtml;
