# SpatialForge ⚡

> **Autonomous Physical Blueprint-to-CAD Agent**  
> *Built for the Meta Global AI Developer Hackathon 2026 (Indie Developer Track)*  
> **Problem Statement:** Push the Frontier (Manufacturing, Robotics & Physical Systems)

[![Meta Open Weight Stack](https://img.shields.io/badge/Meta%20AI-SAM%203.1%20%7C%20Muse%20Spark%20Open-blue.svg)](https://dev.meta.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview
**SpatialForge** is an autonomous physical engineering copilot that transforms 2D technical drawings, engineering blueprints (P&ID, mechanical assemblies), and hand-drawn paper sketches into verified, parametric 3D CAD models (`.STL`, `.STEP`, OpenSCAD) in seconds.

### 🚫 The Problem
Converting physical blueprints and napkin sketches into digital 3D CAD currently takes hours of tedious manual modeling in AutoCAD or SolidWorks. Naive multimodal LLMs hallucinate critical engineering tolerances, confuse dimension callouts, and produce non-functional geometry.

### 💡 The Solution: Segment-Crop-Associate-Verify (SCAV)
SpatialForge eliminates hallucination through a 4-tier autonomous architecture:
1. **Perception (Meta SAM 3.1 Open-Weight):** Zero-shot segmentation of outer hull contours, internal bores, slots, and dimension bounding patches.
2. **Deterministic Micro-OCR:** High-speed localized OCR on isolated text patches to extract exact numerical tolerances (`120.0mm`, `Ø44.0`, `4x M6`) in < 50ms without LLM guessing.
3. **Reasoning & DRC Auditor (Muse Spark Open):** Verifies ISO/ASME Y14.5 design rules, ensures manufacturing wall thickness and hole clearances, and generates parametric OpenSCAD code.
4. **Autonomous Self-Correction Loop:** Renders a 2D synthetic projection of the compiled 3D mesh, computes IoU overlap against the input sketch, and autonomously self-heals any dimensional variance before export.

---

## 🏛️ Architecture

```
[ Blueprint / Sketch ] 
          │
          ▼
 [ Meta SAM 3.1 ] ──────► Contour & Cavity Extraction (Hull, Bores, Slots)
          │
          ▼
 [ Fast Local OCR ] ────► Zero-Hallucination Dimension Value Extraction
          │
          ▼
 [ Muse Spark Open ] ───► ASME Y14.5 DRC Audit + Parametric OpenSCAD Synthesis
          │
          ▼
 [ Self-Healing Loop ] ─► 2D Projection IoU Diff & Auto-Calibration
          │
          ▼
 [ Interactive 3D WebGL ] ──► Real-Time Parametric Controls & 1-Click STL Export
```

---

## 🛠️ Tech Stack
* **Perception:** Meta SAM 3.1 / MobileSAM (Open-Weight)
* **Agent & Reasoning:** Muse Spark Open (via Baseten / Groq / Fireworks)
* **Backend:** Python 3.12, FastAPI, OpenCV, OpenSCAD CLI, `trimesh`
* **Frontend:** Next.js, Tailwind CSS, Three.js / WebGL

---

## 🚀 Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 👥 Authors
* **Muhammad Maaz Korejo** — [GitHub](https://github.com/Maazkorejo) | [Portfolio](https://maaz-korejo.vercel.app/)

---

## 📜 License
MIT License. Built for the Meta Global AI Developer Hackathon.
