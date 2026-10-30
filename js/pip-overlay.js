/**
 * SpatialForge - Picture-in-Picture & Overlay UI Module
 * Handles floating 2D drawing PiP screen and rolling wheel loader overlay.
 */

export let isPipMinimized = false;

export function togglePip(minimize) {
  isPipMinimized = (minimize !== undefined) ? minimize : !isPipMinimized;
  const screen = document.getElementById('pipScreen');
  const pill = document.getElementById('pipPill');
  if (screen && pill) {
    if (isPipMinimized) {
      screen.style.display = 'none';
      pill.style.display = 'flex';
    } else {
      screen.style.display = 'flex';
      pill.style.display = 'none';
    }
  }
}

export function togglePipSize() {
  const screen = document.getElementById('pipScreen');
  if (screen) {
    screen.classList.toggle('enlarged');
  }
}

export function showReconLoader(title = "Constructing 3D CAD B-Rep...", sub = "Vectorizing contours & extruding solid...") {
  const el = document.getElementById('reconstructSpinner');
  const t = document.getElementById('reconStatusTitle');
  const s = document.getElementById('reconStatusSub');
  if (t) t.innerText = title;
  if (s) s.innerText = sub;
  if (el) el.style.display = 'flex';
}

export function hideReconLoader() {
  const el = document.getElementById('reconstructSpinner');
  if (el) el.style.display = 'none';
}

// Bind to window for HTML inline event handlers
window.togglePip = togglePip;
window.togglePipSize = togglePipSize;
window.showReconLoader = showReconLoader;
window.hideReconLoader = hideReconLoader;
