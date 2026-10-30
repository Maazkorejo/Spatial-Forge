/**
 * SpatialForge - CAD Edit UI & Parametric Adjust Engine Module
 * Construct-preserving scaling, stepper adjustments, part cuts/removals, and version history.
 */

export let cadEditSessionId = "default";
export let baseDimensions = { W: 40.0, D: 20.0, H: 10.0 };
export let currentDimensions = { W: 40.0, D: 20.0, H: 10.0 };
export let modelScale = { x: 1.0, y: 1.0, z: 1.0 };
export let baseVolume = 8000;
export let currentVolume = 8000;
export let appliedCuts = [];
export let undoStack = [];
export let redoStack = [];
export let editVersion = 0;
export let cutCounter = 0;

export const cutMaterial = new THREE.MeshStandardMaterial({
  color: 0x181a20,
  roughness: 0.9,
  metalness: 0.1
});

export const cutBorderMaterial = new THREE.LineBasicMaterial({
  color: 0x6f8dff,
  linewidth: 2
});

export function getModelLocalTop() {
  let maxY = 1.0;
  const cadRootGroup = window.cadRootGroup;
  if (cadRootGroup) {
    cadRootGroup.traverse(c => {
      if (c.isMesh && c.geometry && !c.userData.isCutMesh) {
        if (!c.geometry.boundingBox) c.geometry.computeBoundingBox();
        if (c.geometry.boundingBox && c.geometry.boundingBox.max.y > maxY) {
          maxY = c.geometry.boundingBox.max.y;
        }
      }
    });
  }
  return maxY;
}

export function applyTransform() {
  const cadRootGroup = window.cadRootGroup;
  if (cadRootGroup) {
    cadRootGroup.scale.set(modelScale.x, modelScale.y, modelScale.z);
  }

  currentDimensions.W = Math.round(baseDimensions.W * modelScale.x * 10) / 10;
  currentDimensions.D = Math.round(baseDimensions.D * modelScale.y * 10) / 10;
  currentDimensions.H = Math.round(baseDimensions.H * modelScale.z * 10) / 10;
  currentVolume = Math.round(baseVolume * (modelScale.x * modelScale.y * modelScale.z));

  window.currentDimensions = currentDimensions;
  window.currentVolume = currentVolume;

  const dx = document.getElementById('dimX');
  const dy = document.getElementById('dimY');
  const dz = document.getElementById('dimZ');
  const vol = document.getElementById('confVol');
  if (dx) dx.innerText = `${currentDimensions.W.toFixed(1)} mm`;
  if (dy) dy.innerText = `${currentDimensions.D.toFixed(1)} mm`;
  if (dz) dz.innerText = `${currentDimensions.H.toFixed(1)} mm`;
  if (vol) vol.innerText = `${currentVolume.toLocaleString()} mm³`;

  const vTag = document.getElementById('cadVersionTag');
  if (vTag) vTag.innerText = `v${editVersion}`;

  updateFeatureLists();
}

export function scaleModel(factor) {
  const cadRootGroup = window.cadRootGroup;
  if (!cadRootGroup) return;

  modelScale.x = Math.round(modelScale.x * factor * 1000) / 1000;
  modelScale.y = Math.round(modelScale.y * factor * 1000) / 1000;
  modelScale.z = Math.round(modelScale.z * factor * 1000) / 1000;

  editVersion++;
  applyTransform();
  pushHistory({ type: 'scale', factor });

  const pct = Math.round((factor - 1) * 100);
  const sign = pct > 0 ? '+' : '';
  displayGateReport({ ok: true, version: editVersion, messages: [`Scaled construct by ${sign}${pct}%`] });
}

export function adjustDim(axis, delta) {
  const cadRootGroup = window.cadRootGroup;
  if (!cadRootGroup) return;

  const currentVal = currentDimensions[axis];
  const targetVal = Math.max(1, Math.round((currentVal + delta) * 10) / 10);
  const axisFactor = targetVal / currentVal;

  if (axis === 'W') modelScale.x = Math.round(modelScale.x * axisFactor * 1000) / 1000;
  else if (axis === 'D') modelScale.y = Math.round(modelScale.y * axisFactor * 1000) / 1000;
  else if (axis === 'H') modelScale.z = Math.round(modelScale.z * axisFactor * 1000) / 1000;

  editVersion++;
  applyTransform();
  pushHistory({ type: 'adjust', axis, delta });

  const axisName = axis === 'W' ? 'Length' : (axis === 'D' ? 'Width' : 'Thickness');
  displayGateReport({ ok: true, version: editVersion, messages: [`Adjusted ${axisName} to ${targetVal} mm`] });
}

export function quickCutHole(d = 6) {
  const cadRootGroup = window.cadRootGroup;
  if (!cadRootGroup) return;
  cutCounter++;
  const id = `hole_${cutCounter}`;
  const r = d / 2;
  const depth = 8;
  const geom = new THREE.CylinderGeometry(r, r, depth, 32);
  const mesh = new THREE.Mesh(geom, cutMaterial);
  mesh.userData.isCutMesh = true;

  const topY = getModelLocalTop();
  mesh.position.set(0, topY - depth / 2 + 0.1, 0);

  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geom, 30), cutBorderMaterial);
  mesh.add(edges);

  cadRootGroup.add(mesh);

  appliedCuts.push({
    id,
    kind: 'hole',
    label: `Cut Hole Ø${d}`,
    icon: '⚪',
    mesh
  });

  editVersion++;
  displayGateReport({ ok: true, version: editVersion, messages: [`Added cut hole Ø${d} mm on top face`] });
  updateFeatureLists();
  pushHistory({ type: 'add_cut', id });
}

export function quickCutPocket(w = 20, d = 10, depth = 5) {
  const cadRootGroup = window.cadRootGroup;
  if (!cadRootGroup) return;
  cutCounter++;
  const id = `pocket_${cutCounter}`;
  const geom = new THREE.BoxGeometry(w, depth, d);
  const mesh = new THREE.Mesh(geom, cutMaterial);
  mesh.userData.isCutMesh = true;

  const topY = getModelLocalTop();
  mesh.position.set(0, topY - depth / 2 + 0.1, 0);

  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geom, 30), cutBorderMaterial);
  mesh.add(edges);

  cadRootGroup.add(mesh);

  appliedCuts.push({
    id,
    kind: 'pocket_rect',
    label: `Cut Pocket ${w}×${d}`,
    icon: '⏹',
    mesh
  });

  editVersion++;
  displayGateReport({ ok: true, version: editVersion, messages: [`Added pocket cut ${w}×${d} mm`] });
  updateFeatureLists();
  pushHistory({ type: 'add_cut', id });
}

export function quickCutSlot(length = 24, width = 6) {
  const cadRootGroup = window.cadRootGroup;
  if (!cadRootGroup) return;
  cutCounter++;
  const id = `slot_${cutCounter}`;
  const depth = 6;
  const geom = new THREE.BoxGeometry(length, depth, width);
  const mesh = new THREE.Mesh(geom, cutMaterial);
  mesh.userData.isCutMesh = true;

  const topY = getModelLocalTop();
  mesh.position.set(0, topY - depth / 2 + 0.1, 0);

  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geom, 30), cutBorderMaterial);
  mesh.add(edges);

  cadRootGroup.add(mesh);

  appliedCuts.push({
    id,
    kind: 'slot',
    label: `Cut Slot ${length}×${width}`,
    icon: '⬭',
    mesh
  });

  editVersion++;
  displayGateReport({ ok: true, version: editVersion, messages: [`Added slotted cut ${length}×${width} mm`] });
  updateFeatureLists();
  pushHistory({ type: 'add_cut', id });
}

export function quickFilletEdges(radius = 2) {
  cutCounter++;
  const id = `fillet_${cutCounter}`;
  appliedCuts.push({
    id,
    kind: 'fillet',
    label: `Round Edges R${radius}`,
    icon: '⌒'
  });
  editVersion++;
  displayGateReport({ ok: true, version: editVersion, messages: [`Applied R${radius} edge rounds`] });
  updateFeatureLists();
  pushHistory({ type: 'add_cut', id });
}

export function quickChamfer(dist = 1.5) {
  cutCounter++;
  const id = `chamfer_${cutCounter}`;
  appliedCuts.push({
    id,
    kind: 'chamfer',
    label: `Chamfer ${dist}mm`,
    icon: '⏢'
  });
  editVersion++;
  displayGateReport({ ok: true, version: editVersion, messages: [`Applied ${dist}mm edge chamfer`] });
  updateFeatureLists();
  pushHistory({ type: 'add_cut', id });
}

export function quickShell(thickness = 2) {
  cutCounter++;
  const id = `shell_${cutCounter}`;
  appliedCuts.push({
    id,
    kind: 'shell',
    label: `Hollow Shell ${thickness}mm`,
    icon: '◰'
  });
  editVersion++;
  displayGateReport({ ok: true, version: editVersion, messages: [`Applied ${thickness}mm hollow shell`] });
  updateFeatureLists();
  pushHistory({ type: 'add_cut', id });
}

export function removeCut(cutId) {
  const cadRootGroup = window.cadRootGroup;
  const idx = appliedCuts.findIndex(c => c.id === cutId);
  if (idx !== -1) {
    const cut = appliedCuts[idx];
    if (cut.mesh && cadRootGroup) {
      cadRootGroup.remove(cut.mesh);
      if (cut.mesh.geometry) cut.mesh.geometry.dispose();
    }
    appliedCuts.splice(idx, 1);
    editVersion++;
    displayGateReport({ ok: true, version: editVersion, messages: [`Removed part #${cutId}`] });
    updateFeatureLists();
    pushHistory({ type: 'remove_cut', cutId });
  }
}

export function pushHistory(entry) {
  undoStack.push({
    ...entry,
    scale: { ...modelScale },
    dims: { ...currentDimensions },
    vol: currentVolume,
    cutIds: appliedCuts.map(c => c.id)
  });
  redoStack = [];
}

export function cadEditUndo() {
  if (undoStack.length === 0) return;
  const current = undoStack.pop();
  redoStack.push(current);
  if (undoStack.length > 0) {
    const prev = undoStack[undoStack.length - 1];
    modelScale = { ...prev.scale };
    currentDimensions = { ...prev.dims };
    currentVolume = prev.vol;
    applyTransform();
  } else {
    cadEditReset();
  }
}

export function cadEditRedo() {
  if (redoStack.length === 0) return;
  const next = redoStack.pop();
  undoStack.push(next);
  modelScale = { ...next.scale };
  currentDimensions = { ...next.dims };
  currentVolume = next.vol;
  applyTransform();
}

export function cadEditReset() {
  const cadRootGroup = window.cadRootGroup;
  modelScale = { x: 1.0, y: 1.0, z: 1.0 };
  if (cadRootGroup) {
    cadRootGroup.scale.set(1, 1, 1);
  }
  appliedCuts.forEach(c => {
    if (c.mesh && cadRootGroup) {
      cadRootGroup.remove(c.mesh);
      if (c.mesh.geometry) c.mesh.geometry.dispose();
    }
  });
  appliedCuts = [];
  currentDimensions = { ...baseDimensions };
  currentVolume = baseVolume;
  undoStack = [];
  redoStack = [];
  editVersion = 0;
  applyTransform();
  const rep = document.getElementById('cadGateReport');
  if (rep) rep.style.display = 'none';
}

export function updateFeatureLists() {
  const featCount = document.getElementById('featuresCount');
  if (featCount) featCount.innerText = `${appliedCuts.length + 1}`;

  const list = document.getElementById('featuresList');
  if (list) {
    const curFn = window.currentDxfFilename;
    const modelLabel = curFn ? curFn.replace(/\.[^/.]+$/, "") : "Constructed Solid";
    let html = `<div class="part-row sel"><span>${modelLabel}</span><span class="src measured">${currentDimensions.W}×${currentDimensions.D}×${currentDimensions.H}</span></div>`;
    appliedCuts.forEach(f => {
      html += `<div class="part-row"><span>${f.label}</span><span class="src inferred">#${f.id}</span></div>`;
    });
    list.innerHTML = html;
  }

  const cutsEl = document.getElementById('appliedCutsList');
  if (cutsEl) {
    if (appliedCuts.length === 0) {
      cutsEl.innerHTML = '<div class="empty-cuts">No cut parts applied yet.</div>';
    } else {
      cutsEl.innerHTML = appliedCuts.map(f => {
        return `<div class="cut-item">
          <span class="cut-item-title">
            <span style="color:var(--accent);">${f.icon || '•'}</span> ${f.label} <small style="color:var(--muted);font-size:10px;">(${f.id})</small>
          </span>
          <button class="cut-remove-btn" onclick="removeCut('${f.id}')" title="Remove part ${f.id}">✕ Remove</button>
        </div>`;
      }).join('');
    }
  }
}

export function displayGateReport(data) {
  const rep = document.getElementById('cadGateReport');
  if (!rep) return;
  if (data.ok) {
    rep.style.display = 'block';
    rep.style.background = 'rgba(52, 211, 153, 0.08)';
    rep.style.border = '1px solid rgba(52, 211, 153, 0.25)';
    const msg = (data.messages && data.messages.length) ? data.messages[0] : 'Applied successfully';
    rep.innerHTML = `<div style="color:var(--emerald);font-weight:600;display:flex;align-items:center;gap:6px;">
      <span>✓</span> ${msg}
    </div>`;
    setTimeout(() => { if (rep) rep.style.display = 'none'; }, 2600);
  } else {
    rep.style.display = 'block';
    rep.style.background = 'rgba(248, 113, 113, 0.08)';
    rep.style.border = '1px solid rgba(248, 113, 113, 0.25)';
    const errMsg = (data.errors && data.errors.length) ? data.errors.join('; ') : 'Invalid operation';
    rep.innerHTML = `<div style="color:#f87171;font-weight:600;display:flex;align-items:center;gap:6px;">
      <span>✕</span> ${errMsg}
    </div>`;
  }
}

// Bind to window for HTML event handlers and interoperability
window.scaleModel = scaleModel;
window.adjustDim = adjustDim;
window.quickCutHole = quickCutHole;
window.quickCutPocket = quickCutPocket;
window.quickCutSlot = quickCutSlot;
window.quickFilletEdges = quickFilletEdges;
window.quickChamfer = quickChamfer;
window.quickShell = quickShell;
window.removeCut = removeCut;
window.cadEditUndo = cadEditUndo;
window.cadEditRedo = cadEditRedo;
window.cadEditReset = cadEditReset;
window.updateFeatureLists = updateFeatureLists;
window.displayGateReport = displayGateReport;
window.applyTransform = applyTransform;
window.baseDimensions = baseDimensions;
window.currentDimensions = currentDimensions;
window.modelScale = modelScale;
window.baseVolume = baseVolume;
window.currentVolume = currentVolume;
window.appliedCuts = appliedCuts;
window.undoStack = undoStack;
window.redoStack = redoStack;
window.editVersion = editVersion;
