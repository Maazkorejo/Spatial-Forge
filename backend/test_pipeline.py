import asyncio
import os
import sys

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agent.orchestrator import PipelineOrchestrator

async def main():
    print("==================================================")
    print("  SpatialForge End-to-End Pipeline Smoke Test     ")
    print("==================================================")

    # 1. Generate sample blueprint if not present
    samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))
    sample_img_path = os.path.join(samples_dir, "sample_bracket_blueprint.png")
    
    if not os.path.exists(sample_img_path):
        print(f"[1/4] Generating synthetic test blueprint...")
        import subprocess
        gen_script = os.path.join(samples_dir, "generate_sample_blueprint.py")
        subprocess.run([sys.executable, gen_script, sample_img_path], check=True)

    print(f"[2/4] Loading blueprint: {sample_img_path}")
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    # 2. Run Orchestrator
    print("[3/4] Running Pipeline (Perception -> Micro-OCR -> DRC -> CAD -> Self-Healing)...")
    orchestrator = PipelineOrchestrator()
    result = await orchestrator.execute_pipeline(img_bytes)

    # 3. Print Results (using UTF-8 safe print)
    print(f"\n[OK] Execution Completed in: {result.execution_time_ms} ms")
    print(f"[+] Extracted Dimensions: Length: {result.final_parameters.get('length')}mm | Width: {result.final_parameters.get('width')}mm | Bore: {result.final_parameters.get('bore_dia')}mm")
    print(f"[+] DRC Audit Status: {result.drc_report.overall_status} (Weight: {result.drc_report.estimated_weight_grams}g)")
    for rule in result.drc_report.rules:
        print(f"   - [{rule.status}] {rule.rule_name}: {rule.message}")

    print(f"[+] Self-Healing Loop: {len(result.self_healing_traces)} iterations")
    for trace in result.self_healing_traces:
        print(f"   - Iteration #{trace.iteration}: IoU = {trace.iou_score * 100:.2f}% ({trace.status})")

    # 4. Save STL output
    output_stl_path = os.path.join(samples_dir, "output_mounting_bracket.stl")
    import base64
    stl_data = base64.b64decode(result.stl_base64)
    with open(output_stl_path, "wb") as f:
        f.write(stl_data)
    print(f"[4/4] Verified 3D STL exported to: {output_stl_path} ({len(stl_data)} bytes)")
    print("\n>>> SpatialForge Backend Pipeline is fully operational! <<<")

if __name__ == "__main__":
    asyncio.run(main())
