import os
import numpy as np
from PIL import Image
import app2
import gan_module

def test_full_pipeline():
    print("============================================================")
    print("NEPHROSCAN AI(TM) — AUTOMATED SYSTEM DIAGNOSTIC TEST SUITE")
    print("============================================================")
    
    # 1. Test Model & Metadata Loading
    model, meta = app2.load_model_and_meta()
    assert model is not None, "Model failed to load!"
    assert len(meta) == 3, f"Expected 3 classes in meta, got {len(meta)}"
    print("[PASS] Model and Clinical Metadata loaded successfully.")
    
    # 2. Test Clinical Benchmark Cases
    cases = [
        ("Sample_Cyst.jpg", 0, "Cystine Kidney Stone"),
        ("Sample_Normal.jpg", 1, "Normal (Healthy)"),
        ("Sample_Tumor.jpg", 2, "Renal Tumor (Solid Mass)")
    ]
    
    for sample_name, expected_idx, expected_label in cases:
        path = os.path.join(app2.SAMPLES_DIR, sample_name)
        assert os.path.exists(path), f"Sample image missing: {path}"
        img = Image.open(path).convert("RGB")
        
        # Test Preprocessing
        tensor, resized = app2.preprocess_for_cnn(img)
        assert tensor.shape == (1, 150, 150, 3), f"Unexpected tensor shape: {tensor.shape}"
        
        # Test CNN Inference
        preds = model.predict(tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        conf = float(preds[pred_idx]) * 100.0
        disp_name = meta[str(pred_idx)]["display_name"]
        assert pred_idx == expected_idx, f"Inference failure for {sample_name}: Expected {expected_idx}, got {pred_idx}"
        
        # Test Grad-CAM
        h_pil, h_norm = app2.compute_gradcam(tensor, model)
        assert h_pil is not None, "Grad-CAM PIL heatmap generation failed!"
        assert h_norm is not None and h_norm.max() > 0, "Grad-CAM normalized gradient array empty!"
        
        # Test Lesion Caliper Morphometry
        annotated_img, metrics = gan_module.analyze_lesion_morphometry(img, h_norm, pred_idx)
        assert annotated_img is not None, "Lesion morphometry image generation failed!"
        assert "max_diameter_mm" in metrics, "Missing max_diameter_mm in metrics!"
        
        # Test GAN Super-Resolution
        restored = gan_module.apply_gan_super_resolution(img, denoise_strength=0.7)
        assert restored is not None and restored.size == img.size, "GAN Super-Resolution failed!"
        
        print(f"[PASS] {sample_name:<17} | Pred: {disp_name:<20} | Conf: {conf:6.2f}% | Diam: {metrics['max_diameter_mm']:4.1f} mm | Decision: {metrics['clinical_decision'][:32]}...")

    # 3. Test GAN Synthetic Pathology Generator
    print("\n[TEST] Verifying GAN Synthetic Pathology Generator...")
    for c_idx in [0, 1, 2]:
        synth_img = gan_module.generate_synthetic_kidney_scan(class_idx=c_idx, seed=42)
        assert synth_img is not None and synth_img.size == (150, 150), f"Synthetic generation failed for class {c_idx}"
    print("[PASS] GAN Synthetic Pathology generator generated all classes successfully.")

    print("\n============================================================")
    print(">>> ALL TESTS PASSED SUCCESSFULLY! ZERO BUGS OR FAILURES! <<<")
    print("============================================================")

if __name__ == "__main__":
    test_full_pipeline()
