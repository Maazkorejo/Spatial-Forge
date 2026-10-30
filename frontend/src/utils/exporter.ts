export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function downloadBase64(base64: string, filename: string, mimeType: string) {
  const byteChars = atob(base64);
  const byteNums = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) {
    byteNums[i] = byteChars.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNums);
  const blob = new Blob([byteArray], { type: mimeType });
  downloadBlob(blob, filename);
}

export function generateStl(
  vertices: [number, number, number][],
  faces: [number, number, number][],
  name: string = 'model'
): string {
  let stl = `solid ${name.replace(/\s+/g, '_')}\n`;
  for (const [a, b, c] of faces) {
    const v1 = vertices[a] || [0, 0, 0];
    const v2 = vertices[b] || [0, 0, 0];
    const v3 = vertices[c] || [0, 0, 0];

    // Compute surface normal via cross product
    const ax = v2[0] - v1[0];
    const ay = v2[1] - v1[1];
    const az = v2[2] - v1[2];
    const bx = v3[0] - v1[0];
    const by = v3[1] - v1[1];
    const bz = v3[2] - v1[2];

    let nx = ay * bz - az * by;
    let ny = az * bx - ax * bz;
    let nz = ax * by - ay * bx;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len;
    ny /= len;
    nz /= len;

    stl += `  facet normal ${nx.toFixed(4)} ${ny.toFixed(4)} ${nz.toFixed(4)}\n`;
    stl += `    outer loop\n`;
    stl += `      vertex ${v1[0].toFixed(3)} ${v1[1].toFixed(3)} ${v1[2].toFixed(3)}\n`;
    stl += `      vertex ${v2[0].toFixed(3)} ${v2[1].toFixed(3)} ${v2[2].toFixed(3)}\n`;
    stl += `      vertex ${v3[0].toFixed(3)} ${v3[1].toFixed(3)} ${v3[2].toFixed(3)}\n`;
    stl += `    endloop\n`;
    stl += `  endfacet\n`;
  }
  stl += `endsolid ${name.replace(/\s+/g, '_')}\n`;
  return stl;
}

export function generateInteractiveViewerHtml(params: {
  title: string;
  source: string;
  dimensions: { length: number; width: number; thickness: number };
  volume: number;
  vertices: [number, number, number][];
  faces: [number, number, number][];
  blueprintPreview?: string;
}): string {
  const meshJson = JSON.stringify({
    vertices: params.vertices,
    faces: params.faces,
  });

  const hasBlueprint = Boolean(params.blueprintPreview);
  const blueprintSrc = params.blueprintPreview || '';

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${params.title} · Spatial Forge 3D</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #0f141e;
      color: #edf2f7;
      font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }
    #viewport { width: 100%; height: 100%; display: block; }
    .brand-header {
      position: absolute;
      top: 22px;
      left: 24px;
      z-index: 10;
      display: flex;
      flex-direction: column;
      gap: 6px;
      pointer-events: none;
    }
    .brand-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(23, 34, 53, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(174, 189, 208, 0.25);
      padding: 6px 12px;
      border-radius: 9px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #9cb3ce;
    }
    .brand-badge .dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #5fa8d4;
      box-shadow: 0 0 8px rgba(95, 168, 212, 0.8);
    }
    .meta-box {
      background: rgba(23, 34, 53, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(174, 189, 208, 0.2);
      padding: 10px 14px;
      border-radius: 11px;
      max-width: 380px;
    }
    .meta-box h1 {
      font-size: 15px;
      font-weight: 800;
      color: #ffffff;
      letter-spacing: -0.02em;
    }
    .meta-box p {
      font-family: 'DM Mono', monospace;
      font-size: 11px;
      color: #8da4be;
      margin-top: 3px;
    }
    .toolbar {
      position: absolute;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(23, 34, 53, 0.9);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(174, 189, 208, 0.22);
      padding: 6px 14px;
      border-radius: 30px;
      box-shadow: 0 10px 30px rgba(10, 16, 26, 0.5);
    }
    .tool-btn {
      background: transparent;
      border: 1px solid rgba(174, 189, 208, 0.2);
      color: #cbd7e6;
      padding: 7px 14px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .tool-btn:hover {
      background: rgba(255, 255, 255, 0.08);
      color: #ffffff;
      border-color: rgba(174, 189, 208, 0.4);
    }
    .tool-btn.active {
      background: rgba(95, 168, 212, 0.25);
      border-color: #5fa8d4;
      color: #e5f3fc;
    }
    .blueprint-card {
      position: absolute;
      top: 22px;
      right: 24px;
      z-index: 10;
      width: 250px;
      background: rgba(23, 34, 53, 0.92);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(174, 189, 208, 0.25);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 12px 32px rgba(10, 16, 26, 0.5);
      display: ${hasBlueprint ? 'flex' : 'none'};
      flex-direction: column;
    }
    .blueprint-bar {
      padding: 8px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid rgba(174, 189, 208, 0.15);
      background: rgba(15, 23, 38, 0.6);
      font-size: 11px;
      font-weight: 700;
      color: #8da4be;
    }
    .blueprint-actions button {
      background: transparent;
      border: none;
      color: #8da4be;
      cursor: pointer;
      padding: 3px 6px;
      border-radius: 4px;
    }
    .blueprint-actions button:hover {
      color: #ffffff;
      background: rgba(255, 255, 255, 0.1);
    }
    .blueprint-body {
      width: 100%;
      height: 160px;
      background: #090e17;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .blueprint-body img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .blueprint-pill {
      position: absolute;
      top: 22px;
      right: 24px;
      z-index: 10;
      display: none;
      align-items: center;
      gap: 6px;
      padding: 7px 14px;
      background: rgba(23, 34, 53, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(174, 189, 208, 0.25);
      border-radius: 9px;
      font-size: 11px;
      font-weight: 700;
      color: #8da4be;
      cursor: pointer;
    }
    .blueprint-pill:hover {
      color: #ffffff;
      border-color: #5fa8d4;
    }
  </style>
</head>
<body>
  <div class="brand-header">
    <div class="brand-badge">
      <span class="dot"></span>
      Constructed by Spatial Forge
    </div>
    <div class="meta-box">
      <h1>${params.title}</h1>
      <p>${params.dimensions.length} × ${params.dimensions.width} × ${params.dimensions.thickness} mm · ${params.volume} cm³ · Watertight B-Rep</p>
    </div>
  </div>

  <div class="blueprint-card" id="blueprintCard">
    <div class="blueprint-bar">
      <span>2D Source Drawing</span>
      <div class="blueprint-actions">
        <button onclick="minimizeBlueprint()" title="Minimize">_</button>
        <button onclick="closeBlueprint()" title="Close">✕</button>
      </div>
    </div>
    <div class="blueprint-body">
      <img src="${blueprintSrc}" alt="2D Drawing" />
    </div>
  </div>

  <div class="blueprint-pill" id="blueprintPill" onclick="restoreBlueprint()">
    Show 2D Drawing
  </div>

  <div class="toolbar">
    <button class="tool-btn active" id="btnRotate" onclick="toggleRotate()">Auto Rotate</button>
    <button class="tool-btn" onclick="resetCamera()">Reset View</button>
  </div>

  <canvas id="viewport"></canvas>

  <script>
    const meshData = ${meshJson};
    let scene, camera, renderer, controls;
    let meshGroup = new THREE.Group();
    let isAutoRotating = true;
    let defaultCamPos = new THREE.Vector3();
    let defaultTarget = new THREE.Vector3();

    function init() {
      const canvas = document.getElementById('viewport');
      renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(window.innerWidth, window.innerHeight);

      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0e141f);

      camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 2500);

      const ambient = new THREE.AmbientLight(0xffffff, 0.7);
      scene.add(ambient);

      const hemi = new THREE.HemisphereLight(0xffffff, 0x1b2434, 0.6);
      scene.add(hemi);

      const dir1 = new THREE.DirectionalLight(0xffffff, 0.9);
      dir1.position.set(120, 180, 140);
      scene.add(dir1);

      const dir2 = new THREE.DirectionalLight(0x7ea2cc, 0.45);
      dir2.position.set(-120, -60, -100);
      scene.add(dir2);

      const grid = new THREE.GridHelper(300, 30, 0x223247, 0x141f2e);
      grid.position.y = -0.1;
      scene.add(grid);

      scene.add(meshGroup);

      if (meshData.vertices && meshData.vertices.length > 0) {
        const geom = new THREE.BufferGeometry();
        const positions = new Float32Array(meshData.vertices.flat());
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const indices = new Uint32Array(meshData.faces.flat());
        geom.setIndex(new THREE.BufferAttribute(indices, 1));
        geom.computeVertexNormals();
        geom.center();

        const mat = new THREE.MeshStandardMaterial({
          color: 0x5a7ca8,
          roughness: 0.35,
          metalness: 0.3,
          polygonOffset: true,
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1
        });

        const threeMesh = new THREE.Mesh(geom, mat);
        meshGroup.add(threeMesh);

        const edges = new THREE.EdgesGeometry(geom, 24);
        const edgeLine = new THREE.LineSegments(
          edges,
          new THREE.LineBasicMaterial({ color: 0x1d3048, linewidth: 1, transparent: true, opacity: 0.6 })
        );
        meshGroup.add(edgeLine);

        const box = new THREE.Box3().setFromObject(meshGroup);
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z, 20);
        camera.position.set(maxDim * 1.5, maxDim * 1.3, maxDim * 1.8);
        defaultCamPos.copy(camera.position);
      } else {
        camera.position.set(120, 90, 120);
      }

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 1.2;
      defaultTarget.copy(controls.target);

      window.addEventListener('resize', onWindowResize);
      animate();
    }

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }

    function onWindowResize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }

    function toggleRotate() {
      isAutoRotating = !isAutoRotating;
      controls.autoRotate = isAutoRotating;
      document.getElementById('btnRotate').classList.toggle('active', isAutoRotating);
    }

    function resetCamera() {
      camera.position.copy(defaultCamPos);
      controls.target.copy(defaultTarget);
      controls.update();
    }

    function minimizeBlueprint() {
      document.getElementById('blueprintCard').style.display = 'none';
      document.getElementById('blueprintPill').style.display = 'flex';
    }

    function closeBlueprint() {
      document.getElementById('blueprintCard').style.display = 'none';
      document.getElementById('blueprintPill').style.display = 'none';
    }

    function restoreBlueprint() {
      document.getElementById('blueprintCard').style.display = 'flex';
      document.getElementById('blueprintPill').style.display = 'none';
    }

    window.onload = init;
  </script>
</body>
</html>`;
}
