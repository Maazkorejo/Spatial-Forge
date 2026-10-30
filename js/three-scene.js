/**
 * SpatialForge - Three.js 3D CAD Engine Module
 * Studio scene lighting, materials, camera positioning, middle-centering,
 * smart AI orientation, wireframe/shaded controls, and mesh rendering.
 */

export let scene, camera, renderer, controls;
export const threeWrapper = document.getElementById('threeCanvasWrapper');
export const cadRootGroup = new THREE.Group();
export let currentShadingMode = 'wire_on_solid';

export const cadSolidMaterial = new THREE.MeshStandardMaterial({
  color: 0x94a3b8,
  roughness: 0.35,
  metalness: 0.15,
  polygonOffset: true,
  polygonOffsetFactor: 1,
  polygonOffsetUnits: 1
});

export const cadEdgeMaterial = new THREE.LineBasicMaterial({
  color: 0x0f172a
});

export const pureWireMaterial = new THREE.MeshBasicMaterial({
  color: 0x6f8dff,
  wireframe: true
});

export function initThree() {
  const container = document.getElementById('threeCanvasWrapper');
  if (!container) return;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e0f11);

  const aspect = container.clientWidth / container.clientHeight || 1;
  camera = new THREE.PerspectiveCamera(40, aspect, 0.1, 2000);
  camera.position.set(120, 100, 140);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0, 0);

  // Studio Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
  scene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
  dirLight1.position.set(150, 200, 150);
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0x6f8dff, 0.35);
  dirLight2.position.set(-150, 100, -150);
  scene.add(dirLight2);

  // Grid helper
  const grid = new THREE.GridHelper(300, 30, 0x2a2c31, 0x1d1f23);
  grid.position.y = -0.1;
  scene.add(grid);

  scene.add(cadRootGroup);

  window.addEventListener('resize', onWindowResize);
  animate();

  window.scene = scene;
  window.camera = camera;
  window.renderer = renderer;
  window.controls = controls;
}

export function onWindowResize() {
  const container = document.getElementById('threeCanvasWrapper');
  if (!renderer || !camera || !container) return;
  const width = container.clientWidth || 300;
  const height = container.clientHeight || 300;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

export function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

export function clearSceneMeshes() {
  cadRootGroup.rotation.set(0, 0, 0);
  cadRootGroup.position.set(0, 0, 0);
  while (cadRootGroup.children.length > 0) {
    const obj = cadRootGroup.children[0];
    if (obj.geometry) obj.geometry.dispose();
    cadRootGroup.remove(obj);
  }
}

export function renderCadMeshes(meshesDict) {
  clearSceneMeshes();
  if (!meshesDict || Object.keys(meshesDict).length === 0) return;

  for (const [partId, partData] of Object.entries(meshesDict)) {
    if (!partData.vertices || !partData.faces) continue;

    const geom = new THREE.BufferGeometry();
    const positions = [];
    for (const f of partData.faces) {
      for (const idx of f) {
        const pt = partData.vertices[idx];
        if (pt) positions.push(pt[0], pt[2], -pt[1]); // Convert Z-up to Three.js Y-up
      }
    }

    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.computeVertexNormals();

    const mesh = new THREE.Mesh(geom, cadSolidMaterial);
    cadRootGroup.add(mesh);

    // Add crisp edges
    const edgesGeom = new THREE.EdgesGeometry(geom, 24);
    const edgesLine = new THREE.LineSegments(edgesGeom, cadEdgeMaterial);
    edgesLine.visible = (currentShadingMode === 'wire_on_solid');
    cadRootGroup.add(edgesLine);
  }

  // AI Orientation & Viewport Middle-Centering
  cadRootGroup.updateMatrixWorld(true);
  let box = new THREE.Box3().setFromObject(cadRootGroup);
  if (!box.isEmpty()) {
    let size = new THREE.Vector3();
    box.getSize(size);

    // Smart AI Orientation (Horizontal vs. Vertical)
    const isStandingVertically = (size.y > size.z * 1.35 && size.y > size.x * 1.05) || (size.y > size.z * 2.2);
    if (isStandingVertically) {
      cadRootGroup.rotation.x = -Math.PI / 2;
      cadRootGroup.updateMatrixWorld(true);
      box.setFromObject(cadRootGroup);
      box.getSize(size);
    }

    const center = new THREE.Vector3();
    box.getCenter(center);

    // Center model in the exact middle of the grid, flush on y = 0
    cadRootGroup.position.x = -center.x;
    cadRootGroup.position.z = -center.z;
    cadRootGroup.position.y = -box.min.y;
    cadRootGroup.updateMatrixWorld(true);

    // Focus camera on the exact center of the model in the middle of the screen
    const modelH = size.y;
    if (controls) {
      controls.target.set(0, modelH * 0.45, 0);
      const maxDim = Math.max(size.x, size.y, size.z);
      const dist = Math.max(maxDim * 1.85, 80);
      camera.position.set(dist * 0.72, modelH * 0.5 + dist * 0.55, dist * 0.82);
      camera.lookAt(0, modelH * 0.45, 0);
      controls.update();
    }
  }
}

export function setShadingMode(mode) {
  currentShadingMode = mode;
  document.querySelectorAll('.vt-btn').forEach(b => b.classList.remove('active'));
  if (mode === 'wire_on_solid') {
    const el = document.getElementById('btnShaded');
    if (el) el.classList.add('active');
  } else if (mode === 'pure_wire') {
    const el = document.getElementById('btnWire');
    if (el) el.classList.add('active');
  } else if (mode === 'solid') {
    const el = document.getElementById('btnSolid');
    if (el) el.classList.add('active');
  }

  cadRootGroup.traverse(child => {
    if (child.isMesh) {
      child.material = (mode === 'pure_wire') ? pureWireMaterial : cadSolidMaterial;
    }
    if (child.isLineSegments) {
      child.visible = (mode === 'wire_on_solid');
    }
  });
}

// Bind to window for HTML event handlers and global accessibility
window.initThree = initThree;
window.onWindowResize = onWindowResize;
window.clearSceneMeshes = clearSceneMeshes;
window.renderCadMeshes = renderCadMeshes;
window.setShadingMode = setShadingMode;
window.cadRootGroup = cadRootGroup;
