export type ModelKind = "jet" | "wing" | "shaft" | "bracket" | "butterfly" | "wheel" | "housing" | "imported";
export type DisplayMode = "shaded" | "wireframe" | "solid";
export type DimensionKey = "length" | "width" | "thickness";

export type Dimensions = {
  length: number;
  width: number;
  thickness: number;
};

export type AppliedCut = {
  id: string;
  label: string;
  glyph: string;
};

export type BackendMeshData = {
  vertices: number[][]; // [x, y, z]
  faces: number[][];    // [a, b, c]
};

export type ModelState = {
  kind: ModelKind;
  title: string;
  fileName: string;
  dimensions: Dimensions;
  cuts: AppliedCut[];
  isImported?: boolean;
  stepBase64?: string;
  stlBase64?: string;
  glbBase64?: string;
  iou?: number;
  confidence?: number;
  fidelity?: string;
  needsReview?: boolean;
  meshData?: BackendMeshData;
  baseDimensions?: Dimensions;
};

export type SamplePart = {
  kind: ModelKind;
  title: string;
  detail: string;
  dimensions: Dimensions;
  accent: string;
};

export const sampleParts: SamplePart[] = [
  { kind: "jet", title: "F-35 Lightning II", detail: "Aircraft · top profile", dimensions: { length: 158, width: 108, thickness: 24 }, accent: "#4268c7" },
  { kind: "wing", title: "Wing profile", detail: "Airfoil · swept planform", dimensions: { length: 132, width: 82, thickness: 12 }, accent: "#617baa" },
  { kind: "shaft", title: "Turned shaft", detail: "Rotational · stepped", dimensions: { length: 112, width: 34, thickness: 34 }, accent: "#748ba4" },
  { kind: "bracket", title: "Mount bracket", detail: "Plate · two bores", dimensions: { length: 88, width: 64, thickness: 12 }, accent: "#64788e" },
  { kind: "butterfly", title: "Butterfly plate", detail: "Profile · mirrored", dimensions: { length: 76, width: 62, thickness: 8 }, accent: "#7b7fa2" },
  { kind: "wheel", title: "Turbine wheel", detail: "Rotational · radial", dimensions: { length: 76, width: 76, thickness: 20 }, accent: "#566f98" },
  { kind: "housing", title: "Engine housing", detail: "Enclosure · open face", dimensions: { length: 82, width: 72, thickness: 48 }, accent: "#6d8296" },
];

const volumeFactors: Record<ModelKind, number> = {
  jet: 0.078,
  wing: 0.42,
  shaft: 0.48,
  bracket: 0.52,
  butterfly: 0.35,
  wheel: 0.32,
  housing: 0.45,
  imported: 0.39,
};

export function volumeCm3(model: ModelState) {
  const { length, width, thickness } = model.dimensions;
  const nominal = (length * width * thickness * volumeFactors[model.kind]) / 1000;
  const cutFactor = Math.max(0.55, 1 - model.cuts.length * 0.025);
  return nominal * cutFactor;
}

export function formatNumber(value: number, digits = 1) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value);
}

export function createSampleModel(kind: ModelKind): ModelState {
  const sample = sampleParts.find((item) => item.kind === kind) ?? sampleParts[0];
  return {
    kind: sample.kind,
    title: sample.title,
    fileName: sample.kind === "jet" ? "F-35A_top-view.dxf" : `${sample.title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.dxf`,
    dimensions: { ...sample.dimensions },
    cuts: [],
  };
}

export function createImportedModel(fileName: string, byteLength: number): ModelState {
  const baseName = fileName.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
  const scale = Math.max(0.86, Math.min(1.2, 0.9 + (byteLength % 301) / 1000));
  return {
    kind: "imported",
    title: baseName || "Imported profile",
    fileName,
    dimensions: { length: Math.round(96 * scale), width: Math.round(67 * scale), thickness: Math.round(10 * scale) },
    cuts: [],
    isImported: true,
  };
}

export function cloneModel(model: ModelState): ModelState {
  return {
    ...model,
    dimensions: { ...model.dimensions },
    baseDimensions: model.baseDimensions ? { ...model.baseDimensions } : undefined,
    cuts: model.cuts.map((cut) => ({ ...cut })),
  };
}
