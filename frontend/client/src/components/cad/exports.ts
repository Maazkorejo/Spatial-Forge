import * as THREE from "three";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import { STLExporter } from "three/addons/exporters/STLExporter.js";
import type { ModelState } from "./types";
import { volumeCm3, formatNumber } from "./types";

type Triangle = [[number, number, number], [number, number, number], [number, number, number]];

function collectTriangles(root: THREE.Object3D): Triangle[] {
  root.updateMatrixWorld(true);
  const triangles: Triangle[] = [];
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh) || object.userData.cutMarker) return;
    const geometry = object.geometry.index ? object.geometry.toNonIndexed() : object.geometry;
    const position = geometry.getAttribute("position");
    const points: THREE.Vector3[] = [];
    for (let index = 0; index < position.count; index += 1) points.push(new THREE.Vector3().fromBufferAttribute(position, index).applyMatrix4(object.matrixWorld));
    for (let index = 0; index + 2 < points.length; index += 3) {
      const a = points[index];
      const b = points[index + 1];
      const c = points[index + 2];
      if (new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a)).lengthSq() < 1e-10) continue;
      triangles.push([[a.x, a.y, a.z], [b.x, b.y, b.z], [c.x, c.y, c.z]]);
    }
    if (geometry !== object.geometry) geometry.dispose();
  });
  return triangles;
}

export function exportSTL(root: THREE.Object3D) {
  root.updateMatrixWorld(true);
  const result = new STLExporter().parse(root, { binary: true });
  return new Blob([result as BlobPart], { type: "application/sla" });
}

export function exportGLB(root: THREE.Object3D) {
  root.updateMatrixWorld(true);
  return new Promise<Blob>((resolve, reject) => {
    new GLTFExporter().parse(root, (result) => {
      if (result instanceof ArrayBuffer) resolve(new Blob([result], { type: "model/gltf-binary" }));
      else reject(new Error("A binary GLB could not be created."));
    }, (error) => reject(error), { binary: true });
  });
}

function dxfFromTriangles(triangles: Triangle[]) {
  const lines = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"];
  const precision = (value: number) => Number(value.toFixed(3));
  const edges = new Set<string>();
  triangles.forEach((triangle) => {
    for (let index = 0; index < 3; index += 1) {
      const a = triangle[index];
      const b = triangle[(index + 1) % 3];
      const start = [precision(a[0]), precision(a[2])];
      const end = [precision(b[0]), precision(b[2])];
      const first = `${start[0]},${start[1]}`;
      const second = `${end[0]},${end[1]}`;
      if (first === second) continue;
      const key = first < second ? `${first}|${second}` : `${second}|${first}`;
      if (edges.has(key)) continue;
      edges.add(key);
      lines.push("0", "LINE", "8", "PROFILE", "10", String(start[0]), "20", String(start[1]), "30", "0", "11", String(end[0]), "21", String(end[1]), "31", "0");
    }
  });
  lines.push("0", "ENDSEC", "0", "EOF");
  return new Blob([lines.join("\n")], { type: "application/dxf" });
}

function stepFromTriangles(triangles: Triangle[], title: string) {
  let nextId = 1;
  const entities: string[] = [];
  const add = (line: string) => { const id = nextId++; entities.push(`#${id}=${line};`); return id; };
  const pointCache = new Map<string, number>();
  const pointId = (point: number[]) => {
    const rounded = point.map((value) => Number(value.toFixed(5)));
    const key = rounded.join(",");
    const found = pointCache.get(key);
    if (found) return found;
    const id = add(`CARTESIAN_POINT('',(${rounded.map((value) => value.toFixed(5)).join(",")}))`);
    pointCache.set(key, id);
    return id;
  };

  const grouped: number[] = [];
  for (let triangleIndex = 0; triangleIndex < triangles.length; triangleIndex += 1) {
    const triangle = triangles[triangleIndex];
    const [a, b, c] = triangle;
    const av = new THREE.Vector3(...a);
    const bv = new THREE.Vector3(...b);
    const cv = new THREE.Vector3(...c);
    const normal = bv.clone().sub(av).cross(cv.clone().sub(av)).normalize();
    if (normal.lengthSq() < 0.5) continue;
    const pa = pointId(a); const pb = pointId(b); const pc = pointId(c);
    const direction = add(`DIRECTION('',(${normal.toArray().map((value) => value.toFixed(6)).join(",")}))`);
    const reference = add("DIRECTION('',(1.,0.,0.))");
    const placement = add(`AXIS2_PLACEMENT_3D('',#${pa},#${direction},#${reference})`);
    const plane = add(`PLANE('',#${placement})`);
    const loop = add(`POLY_LOOP('',(#${pa},#${pb},#${pc}))`);
    const bound = add(`FACE_OUTER_BOUND('',#${loop},.T.)`);
    const face = add(`ADVANCED_FACE('',(#${bound}),#${plane},.T.)`);
    grouped.push(face);
  }
  const shell = add(`CLOSED_SHELL('',(${grouped.map((id) => `#${id}`).join(",")}))`);
  const brep = add(`FACETED_BREP('${title.replace(/[^A-Za-z0-9 _-]/g, "").slice(0, 40)}',#${shell})`);
  const appContext = add("APPLICATION_CONTEXT('automotive_design')");
  const protocol = add(`APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#${appContext})`);
  const productContext = add(`PRODUCT_CONTEXT('',#${appContext},'mechanical')`);
  const product = add(`PRODUCT('Spatial Forge preview','${title.replace(/[^A-Za-z0-9 _-]/g, "").slice(0, 48)}','',(#${productContext}))`);
  const formation = add(`PRODUCT_DEFINITION_FORMATION('1','',#${product})`);
  const definitionContext = add(`PRODUCT_DEFINITION_CONTEXT('part definition',#${appContext},'design')`);
  const definition = add(`PRODUCT_DEFINITION('design','',#${formation},#${definitionContext})`);
  const shape = add(`PRODUCT_DEFINITION_SHAPE('','',#${definition})`);
  const origin = add("CARTESIAN_POINT('',(0.,0.,0.))");
  const directionZ = add("DIRECTION('',(0.,0.,1.))");
  const directionX = add("DIRECTION('',(1.,0.,0.))");
  const placement = add(`AXIS2_PLACEMENT_3D('',#${origin},#${directionZ},#${directionX})`);
  const units = add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))");
  const angle = add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))");
  const solidAngle = add("(NAMED_UNIT(*) SOLID_ANGLE_UNIT() SI_UNIT($,.STERADIAN.))");
  const uncertainty = add(`UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-6),#${units},'distance_accuracy_value','')`);
  const context = add(`(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#${units},#${angle},#${solidAngle})) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#${uncertainty})) REPRESENTATION_CONTEXT('',''))`);
  const representation = add(`ADVANCED_BREP_SHAPE_REPRESENTATION('Spatial Forge faceted preview',(#${placement},#${brep}),#${context})`);
  add(`SHAPE_DEFINITION_REPRESENTATION(#${shape},#${representation})`);
  const safeTitle = title.replace(/['\\]/g, "_").slice(0, 64);
  return new Blob([`ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('Spatial Forge faceted preview'),'2;1');\nFILE_NAME('${safeTitle}.step','${new Date().toISOString()}',('Spatial Forge'),('Spatial Forge'),'','Spatial Forge','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));\nENDSEC;\nDATA;\n${entities.join("\n")}\nENDSEC;\nEND-ISO-10303-21;`], { type: "model/step" });
}

function escapeHtml(value: string) {
  return value.replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character] ?? character);
}

function standaloneHtml(triangles: Triangle[], model: ModelState) {
  const geometry = JSON.stringify(triangles);
  const title = escapeHtml(model.title);
  const data = `${model.dimensions.length} × ${model.dimensions.width} × ${model.dimensions.thickness} mm`;
  const volume = `${formatNumber(volumeCm3(model))} cm³`;
  const script = `const T=${geometry};let yaw=-.72,pitch=.58,zoom=1.1,drag=null,wire=false,spin=true;const c=document.querySelector('canvas'),x=c.getContext('2d');function draw(){const w=c.width=c.clientWidth*devicePixelRatio,h=c.height=c.clientHeight*devicePixelRatio; x.clearRect(0,0,w,h);if(spin)yaw+=.003;const all=T.flat(),span=Math.max(1,...all.map(p=>Math.max(Math.abs(p[0]),Math.abs(p[1]),Math.abs(p[2]))));const faces=T.map(t=>{const v=t.map(p=>{let a=p[0]/span,b=p[1]/span,d=p[2]/span;let X=a*Math.cos(yaw)-d*Math.sin(yaw),Z=a*Math.sin(yaw)+d*Math.cos(yaw);let Y=b*Math.cos(pitch)-Z*Math.sin(pitch),Q=b*Math.sin(pitch)+Z*Math.cos(pitch);return [X,Y,Q]});let u=v[1].map((n,i)=>n-v[0][i]),r=v[2].map((n,i)=>n-v[0][i]);let light=Math.max(.24,Math.min(1,.48+Math.abs((u[1]*r[2]-u[2]*r[1])*.6)));return {v,z:v.reduce((a,p)=>a+p[2],0)/3,l:light}}).sort((a,b)=>a.z-b.z);for(const f of faces){x.beginPath();f.v.forEach((p,i)=>{const px=w/2+p[0]*Math.min(w,h)*.66*zoom,py=h*.53-p[1]*Math.min(w,h)*.66*zoom;i?x.lineTo(px,py):x.moveTo(px,py)});x.closePath();x.fillStyle='rgb('+Math.round(95+90*f.l)+','+Math.round(112+93*f.l)+','+Math.round(137+100*f.l)+')';x.fill();x.strokeStyle=wire?'#536682':'rgba(64,83,109,.28)';x.lineWidth=devicePixelRatio*(wire?1.1:.45);x.stroke()}requestAnimationFrame(draw)}draw();c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1,Math.min(1.2,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY]}};c.onwheel=e=>{zoom=Math.max(.55,Math.min(2.5,zoom-e.deltaY*.001));e.preventDefault()};document.querySelector('#wire').onclick=()=>{wire=!wire;document.querySelector('#wire').classList.toggle('active',wire)};document.querySelector('#spin').onclick=()=>{spin=!spin;document.querySelector('#spin').classList.toggle('active',spin)};document.querySelector('#reset').onclick=()=>{yaw=-.72;pitch=.58;zoom=1.1};document.querySelector('#pipclose').onclick=()=>document.querySelector('#pip').remove();document.querySelector('#pipmin').onclick=()=>document.querySelector('#pip').classList.toggle('min')`;
  return new Blob([`<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${title} · Spatial Forge</title><style>*{box-sizing:border-box}body{margin:0;background:#dce1e8;color:#29364a;font:14px Manrope,system-ui,sans-serif;height:100vh;overflow:hidden}.card{position:absolute;z-index:2;top:24px;left:24px;padding:16px 19px;background:#cfd6df;border:1px solid #b9c2cd;border-radius:14px;box-shadow:0 12px 30px #53627a16}.brand{font-size:10px;letter-spacing:.16em;font-weight:700;color:#4769ae}.title{font-size:18px;font-weight:650;margin:9px 0 5px}.meta{font:11px ui-monospace,monospace;color:#667489}.toolbar{position:absolute;z-index:3;top:24px;right:24px;display:flex;gap:6px;padding:6px;background:#cfd6df;border:1px solid #b9c2cd;border-radius:12px}.toolbar button{background:transparent;border:0;border-radius:8px;color:#52627a;padding:9px 12px;font:inherit;cursor:pointer}.toolbar button:hover,.toolbar .active{background:#b8c7df;color:#365b9e}canvas{width:100vw;height:100vh;display:block}.pip{position:absolute;right:24px;bottom:24px;width:250px;height:158px;border:1px solid #acb9cb;border-radius:12px;background:#d0d8e3;overflow:hidden;box-shadow:0 8px 30px #53627a20}.pip header{height:33px;padding:0 10px;display:flex;align-items:center;justify-content:space-between;background:#c6d0dc;font-size:9px;letter-spacing:.1em}.pip button{background:none;border:0;color:#52627a;cursor:pointer}.sketch{width:100%;height:calc(100% - 33px);padding:8px;color:#60779a}.min{height:34px;width:160px}.min .sketch,.min header span{display:none}@media(max-width:650px){.card{top:12px;left:12px;padding:12px}.toolbar{top:auto;right:12px;bottom:12px}.pip{bottom:70px;right:12px;width:205px}.title{font-size:15px}}</style><section class="card"><div class="brand">SPATIAL FORGE · INTERACTIVE EXPORT</div><div class="title">${title}</div><div class="meta">${escapeHtml(data)} &nbsp; / &nbsp; ${escapeHtml(volume)}</div></section><nav class="toolbar"><button id="wire">Wireframe</button><button id="spin" class="active">Auto rotate</button><button id="reset">Reset view</button></nav><canvas aria-label="Interactive exported 3D model"></canvas><aside class="pip" id="pip"><header><span>2D BLUEPRINT</span><span><button id="pipmin" aria-label="Minimize drawing">−</button><button id="pipclose" aria-label="Close drawing">×</button></span></header><svg class="sketch" viewBox="0 0 240 120"><path d="M120 10 132 48 195 67 145 74 130 108 120 101 110 108 95 74 45 81 108 48Z" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M120 13v88M55 75h130" stroke="currentColor" stroke-dasharray="3 3" fill="none"/></svg></aside><script>${script}</script></html>`], { type: "text/html" });
}

export async function makeExport(format: "step" | "stl" | "html" | "glb" | "dxf", root: THREE.Object3D, model: ModelState, sourceImageData?: string) {
  root.updateMatrixWorld(true);
  const triangles = collectTriangles(root);
  switch (format) {
    case "stl": {
      if (model.stlBase64 && model.cuts.length === 0) {
        try {
          const binaryString = atob(model.stlBase64);
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
          return new Blob([bytes], { type: "application/sla" });
        } catch {
          // fallback to client tessellation
        }
      }
      return exportSTL(root);
    }
    case "glb": return exportGLB(root);
    case "step": {
      if (model.stepBase64 && model.cuts.length === 0) {
        try {
          const binaryString = atob(model.stepBase64);
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
          return new Blob([bytes], { type: "model/step" });
        } catch {
          // fallback to client faceted step
        }
      }
      return stepFromTriangles(triangles, model.title);
    }
    case "dxf": return dxfFromTriangles(triangles);
    case "html": {
      const html = await standaloneHtml(triangles, model).text();
      if (!sourceImageData) return new Blob([html], { type: "text/html" });
      const sourceImage = `<img class="sketch" src="${escapeHtml(sourceImageData)}" alt="Embedded source drawing">`;
      return new Blob([html.replace(/<svg class="sketch"[\s\S]*?<\/svg>/, sourceImage)], { type: "text/html" });
    }
  }
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1500);
}
