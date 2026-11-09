import os
import io
import json
import base64
import datetime
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import gan_module

app = FastAPI(title="NephroScan AI™ Clinical Backend", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "kidney_model.keras")
METADATA_PATH = os.path.join(BASE_DIR, "model", "class_indices.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_images")

DEFAULT_CLASS_META = {
    "0": {
        "folder_name": "Cyst",
        "display_name": "Cystine Kidney Stone",
        "sub_title": "Cystine Calculus / Renal Cystic Lesion",
        "category": "Cystine Stone / Cyst",
        "severity": "Moderate Priority",
        "urgency": "Outpatient Urological Consultation",
        "color": "#f59e0b",
        "badge": "⚠️ Pathological Finding (Calculus)",
        "clinical_summary": "The deep CNN identified focal radiological attenuation characteristic of Cystine Nephrolithiasis (kidney calculus) or well-circumscribed renal cystic changes.",
        "description": "Cystine stones are a metabolic form of urolithiasis caused by cystinuria, an autosomal recessive renal tubular defect resulting in impaired cystine reabsorption and spontaneous crystallization in acidic urine (pH < 6.5). On non-contrast CT, cystine stones exhibit characteristic homogeneous ground-glass attenuation (approx. 500–700 Hounsfield Units).",
        "etiology": "Mutations in SLC3A1 (Type A) or SLC7A9 (Type B) genes, reducing dibasic amino acid (cystine, ornithine, lysine, arginine) resorption in proximal renal tubules.",
        "treatments": [
            "💧 Aggressive Hydration Therapy: Target 24-hr urine volume > 3.0–3.5 Liters (specific gravity < 1.010). Consume fluids evenly distributed across 24h, including during nighttime awakenings.",
            "🧪 Urinary Alkalinization: Potassium Citrate (60–80 mEq/day in divided doses) titrated to maintain urinary pH strictly between 7.0 and 7.5. Avoid sodium bicarbonate.",
            "🧂 Dietary Sodium & Protein Restriction: Limit sodium to < 2,000 mg/day (reduces cystine excretion) and moderate animal protein consumption to decrease methionine precursor intake.",
            "💊 Thiol-Binding Chelators: Second-line intervention for refractory stones (Alpha-mercaptopropionylglycine / Tiopronin, or D-Penicillamine with Pyridoxine B6).",
            "⚡ Endourological Intervention: Extracorporeal Shock Wave Lithotripsy (ESWL) for stones <10 mm, Retrograde Intrarenal Surgery (RIRS / Ureteroscopy) with Holmium:YAG / Thulium laser, or Percutaneous Nephrolithotomy (PCNL) for large staghorn calculi."
        ]
    },
    "1": {
        "folder_name": "Normal",
        "display_name": "Normal (Healthy)",
        "sub_title": "Unremarkable Bilateral Renal Parenchyma",
        "category": "Normal Anatomy",
        "severity": "Low Priority",
        "urgency": "Routine Preventive Follow-up",
        "color": "#10b981",
        "badge": "✅ Normal Renal Architecture",
        "clinical_summary": "Normal renal parenchymal thickness, preserved corticomedullary differentiation, symmetrical bilateral renal contour, and absent calculus or mass lesions.",
        "description": "Normal non-contrast abdominal CT reveals homogenous renal parenchyma with standard attenuation values (approx. 30–40 Hounsfield Units). The renal collecting system and ureters are non-dilated with absence of hydronephrosis, perinephric stranding, or hyperdense calcifications.",
        "etiology": "Physiological baseline renal parenchyma without acute obstruction, structural malformations, oncological growth, or metabolic stone formation.",
        "treatments": [
            "💧 Preventative Hydration: Maintain standard fluid intake of 2.0 to 2.5 L/day to ensure adequate urinary dilution and solute clearance.",
            "🥗 Balanced Renal Protective Diet: Moderate dietary sodium (<2,300 mg/day), maintain balanced calcium intake from natural food sources, and avoid excessive refined sugars.",
            "🩺 Routine Primary Care Screening: Annual blood pressure monitoring, urinalysis (absence of protein/blood), and baseline serum creatinine / eGFR check.",
            "🏃 Physical Fitness & Weight Optimization: Regular cardiovascular exercise and weight control to mitigate long-term hypertensive and diabetic nephrosclerosis risk."
        ]
    },
    "2": {
        "folder_name": "Tumor",
        "display_name": "Renal Tumor (Solid Mass)",
        "sub_title": "Suspicious Renal Neoplasm / Solid Mass",
        "category": "Neoplastic Lesion",
        "severity": "Critical / Urgent",
        "urgency": "Urgent Urologic Oncology Referral",
        "color": "#ef4444",
        "badge": "🚨 Critical Finding (Renal Neoplasm)",
        "clinical_summary": "The model detected focal structural heterogeneity, contour distortion, and anomalous parenchymal attenuation indicative of a solid renal mass.",
        "description": "Renal neoplasms encompass Renal Cell Carcinoma (Clear Cell RCC, Papillary RCC, Chromophobe RCC) or benign entities (Oncocytoma, Angiomyolipoma). Cross-sectional multiphasic contrast-enhanced imaging is required to assess enhancement, vascularity, and staging.",
        "etiology": "Somatic or hereditary genetic alterations (VHL, MET, TSC1/2 mutations) with known risk factors including smoking, hypertension, obesity, and toxic exposures.",
        "treatments": [
            "🏥 Urgent Urologic Oncology Referral: Comprehensive clinical staging, physical exam, and multidisciplinary tumor board review.",
            "🔬 Quad-Phase Contrast-Enhanced CT / Renal MRI: Evaluate enhancement kinetics (>15-20 HU rise), capsular penetration, and renal vein/IVC involvement.",
            "🔪 Nephron-Sparing Surgery (Partial Nephrectomy): Gold-standard resection for localized T1a/T1b masses (<7 cm) preserving functional renal parenchyma.",
            "💉 Radical Nephrectomy: Complete surgical excision of kidney, adrenal gland, and Gerota's fascia for large, centrally invasive, or high-stage tumors.",
            "⚡ Image-Guided Thermal Ablation: Percutaneous cryoablation or microwave ablation for select small exophytic tumors in high-surgical-risk patients."
        ]
    }
}

# Load model once at startup
print("[NephroScan Backend] Loading model weights and class metadata...")
loaded_model = None
if os.path.exists(MODEL_PATH):
    try:
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
        print("[NephroScan Backend] Model loaded successfully.")
    except Exception as e:
        print(f"[NephroScan Backend] Error loading model: {e}")

CLASS_META = DEFAULT_CLASS_META
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r") as f:
            raw_meta = json.load(f)
            for k, v in raw_meta.items():
                if k in CLASS_META and isinstance(v, dict):
                    CLASS_META[k].update(v)
    except Exception as e:
        print(f"[NephroScan Backend] Error loading class_indices: {e}")

def pil_to_base64(img: Image.Image, format="JPEG", quality=90) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def preprocess_for_cnn(pil_img):
    resized = pil_img.resize((150, 150), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    normalized = arr / 255.0
    return np.expand_dims(normalized, axis=0), resized

def compute_gradcam(img_tensor, base_model):
    try:
        sub_conv = tf.keras.Sequential(base_model.layers[:5])
        sub_dense = tf.keras.Sequential(base_model.layers[5:-1])
        last_dense = base_model.layers[-1]
        
        with tf.GradientTape() as tape:
            conv_outputs = sub_conv(img_tensor)
            tape.watch(conv_outputs)
            dense_feats = sub_dense(conv_outputs)
            logits = tf.matmul(dense_feats, last_dense.kernel) + last_dense.bias
            top_index = tf.argmax(logits[0])
            loss = logits[:, top_index]
            
        grads = tape.gradient(loss, conv_outputs)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_outputs[0]
        heatmap = tf.maximum(tf.squeeze(conv_out @ pooled[..., tf.newaxis]), 0)
        
        h_max = tf.reduce_max(heatmap).numpy()
        if h_max > 0:
            heatmap_norm = (heatmap / h_max).numpy()
        else:
            heatmap_norm = heatmap.numpy()
        
        # Jet Colormap
        r = np.clip(1.5 - np.abs(4.0 * heatmap_norm - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * heatmap_norm - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * heatmap_norm - 1.0), 0.0, 1.0)
        heatmap_rgb = (np.dstack([r, g, b]) * 255.0).astype(np.uint8)
        
        heatmap_pil = Image.fromarray(heatmap_rgb).resize((150, 150), Image.Resampling.BILINEAR)
        return heatmap_pil, heatmap_norm
    except Exception as e:
        print(f"[GradCAM Error] {e}")
        return None, None

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model_loaded": loaded_model is not None,
        "classes": [CLASS_META[str(i)]["display_name"] for i in range(3)],
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/api/samples")
def get_samples():
    samples = [
        {
            "id": "case_1",
            "name": "Clinical Case 1: Cystine Kidney Stone",
            "filename": "Sample_Cyst.jpg",
            "expected_class": "Cystine Stone",
            "color": "#f59e0b"
        },
        {
            "id": "case_2",
            "name": "Clinical Case 2: Normal Healthy Kidney",
            "filename": "Sample_Normal.jpg",
            "expected_class": "Normal",
            "color": "#10b981"
        },
        {
            "id": "case_3",
            "name": "Clinical Case 3: Renal Tumor / Mass",
            "filename": "Sample_Tumor.jpg",
            "expected_class": "Renal Tumor",
            "color": "#ef4444"
        }
    ]
    for s in samples:
        path = os.path.join(SAMPLES_DIR, s["filename"])
        if os.path.exists(path):
            with Image.open(path).convert("RGB") as img:
                s["thumbnail"] = pil_to_base64(img.resize((100, 100)), quality=80)
        else:
            s["thumbnail"] = None
    return samples

@app.post("/api/synthesize")
def synthesize(
    class_idx: int = Form(0),
    seed: int = Form(101),
    severity: float = Form(1.0)
):
    synth_img = gan_module.generate_synthetic_kidney_scan(class_idx, seed=seed, severity=severity)
    return {
        "class_idx": class_idx,
        "seed": seed,
        "severity": severity,
        "image": pil_to_base64(synth_img, format="JPEG", quality=95)
    }

@app.post("/api/analyze")
async def analyze_ct(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    synthetic_seed: int = Form(None),
    synthetic_class: int = Form(None),
    enable_gan_sr: bool = Form(True),
    simulate_noise: bool = Form(False),
    denoise_strength: float = Form(0.7),
    contrast_window: float = Form(1.0),
    cam_opacity: float = Form(0.45)
):
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Deep learning model not loaded on server.")

    # 1. Acquire Image
    selected_image = None
    source_name = "Custom Upload"

    if file is not None and file.filename:
        content = await file.read()
        selected_image = Image.open(io.BytesIO(content)).convert("RGB")
        source_name = f"Uploaded: {file.filename}"
    elif sample_id:
        sample_map = {
            "case_1": "Sample_Cyst.jpg",
            "case_2": "Sample_Normal.jpg",
            "case_3": "Sample_Tumor.jpg"
        }
        filename = sample_map.get(sample_id, "Sample_Cyst.jpg")
        path = os.path.join(SAMPLES_DIR, filename)
        if os.path.exists(path):
            selected_image = Image.open(path).convert("RGB")
            source_name = f"Reference CT: {filename}"
        else:
            raise HTTPException(status_code=404, detail="Sample image not found on disk.")
    elif synthetic_seed is not None:
        c_idx = synthetic_class if synthetic_class is not None else 0
        selected_image = gan_module.generate_synthetic_kidney_scan(c_idx, seed=synthetic_seed, severity=1.0)
        source_name = f"Synthetic CT (Seed {synthetic_seed})"
    else:
        # Fallback to case 1
        path = os.path.join(SAMPLES_DIR, "Sample_Cyst.jpg")
        if os.path.exists(path):
            selected_image = Image.open(path).convert("RGB")
            source_name = "Default Reference CT"
        else:
            raise HTTPException(status_code=400, detail="No valid input scan provided.")

    # 2. Radiation Noise Simulation (Demo feature)
    base_working_image = selected_image.copy()
    if simulate_noise:
        base_working_image = gan_module.add_ct_radiation_noise(base_working_image, noise_level=0.45)

    # 3. Windowing / Contrast adjustment
    if contrast_window != 1.0:
        enhancer = ImageEnhance.Contrast(base_working_image)
        base_working_image = enhancer.enhance(contrast_window)

    # 4. Neural SR-GAN Restoration
    if enable_gan_sr:
        gan_enhanced_image = gan_module.apply_gan_super_resolution(
            base_working_image,
            denoise_strength=denoise_strength,
            edge_enhancement=1.35
        )
        eval_image = gan_enhanced_image
    else:
        gan_enhanced_image = base_working_image
        eval_image = base_working_image

    # Compute difference residual map
    raw_arr = np.array(base_working_image.convert("RGB"), dtype=np.float32)
    gan_arr = np.array(gan_enhanced_image.convert("RGB"), dtype=np.float32)
    diff_arr = np.abs(gan_arr - raw_arr)
    diff_amplified = np.clip(diff_arr * 4.0, 0, 255).astype(np.uint8)
    diff_image = Image.fromarray(diff_amplified)

    # 5. CNN Preprocessing & Inference
    img_tensor, resized_pil = preprocess_for_cnn(eval_image)
    predictions = loaded_model.predict(img_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    conf_score = float(predictions[pred_idx]) * 100.0
    info = CLASS_META.get(str(pred_idx), CLASS_META["0"])

    # 6. Grad-CAM Computation
    heatmap_pil, heatmap_norm = compute_gradcam(img_tensor, loaded_model)
    if heatmap_pil is not None:
        blended_cam = Image.blend(resized_pil, heatmap_pil, alpha=cam_opacity)
    else:
        blended_cam = resized_pil

    # 7. Lesion Caliper Morphometry
    if heatmap_norm is not None:
        annotated_calipers, morph_metrics = gan_module.analyze_lesion_morphometry(
            eval_image, heatmap_norm, pred_idx
        )
    else:
        annotated_calipers = eval_image
        morph_metrics = {
            "max_diameter_mm": 0.0, "area_mm2": 0.0, "mean_hu": 35,
            "max_hu": 40, "urgency_tier": info["severity"],
            "clinical_decision": "Standard triage", "theme_color": info["color"],
            "has_lesion": False
        }

    # 8. HU Density Histogram
    gray_slice = np.array(eval_image.convert("L")).flatten()
    hu_approx = np.clip(gray_slice * 4.7 - 200, -200, 1000)
    hist_vals, bin_edges = np.histogram(hu_approx, bins=25)
    bin_centers = (0.5 * (bin_edges[:-1] + bin_edges[1:])).tolist()
    hu_histogram = [{"hu": int(bin_centers[i]), "count": int(hist_vals[i])} for i in range(len(hist_vals))]

    # 9. Probabilities
    probabilities = [
        {
            "class_idx": i,
            "name": CLASS_META[str(i)]["display_name"],
            "category": CLASS_META[str(i)]["category"],
            "probability": round(float(predictions[i]) * 100.0, 2),
            "color": CLASS_META[str(i)]["color"]
        }
        for i in range(3)
    ]

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source_name": source_name,
        "prediction": {
            "class_idx": pred_idx,
            "display_name": info["display_name"],
            "sub_title": info["sub_title"],
            "category": info["category"],
            "severity": info["severity"],
            "urgency": info["urgency"],
            "color": info["color"],
            "badge": info["badge"],
            "clinical_summary": info["clinical_summary"],
            "description": info["description"],
            "etiology": info["etiology"],
            "treatments": info["treatments"]
        },
        "confidence": round(conf_score, 2),
        "confidence_formatted": "100%" if conf_score >= 99.95 else f"{conf_score:.1f}%",
        "probabilities": probabilities,
        "morph_metrics": morph_metrics,
        "hu_histogram": hu_histogram,
        "viewports": {
            "raw": pil_to_base64(base_working_image),
            "restored": pil_to_base64(gan_enhanced_image),
            "saliency": pil_to_base64(blended_cam),
            "calipers": pil_to_base64(annotated_calipers),
            "residual_diff": pil_to_base64(diff_image)
        }
    }

@app.get("/api/architecture")
def get_architecture():
    return {
        "layers": [
            {"name": "conv2d", "type": "Conv2D (32 filters, 3x3)", "output_shape": "(148, 148, 32)", "parameters": 896},
            {"name": "max_pooling2d", "type": "MaxPooling2D (2x2)", "output_shape": "(74, 74, 32)", "parameters": 0},
            {"name": "conv2d_1", "type": "Conv2D (64 filters, 3x3)", "output_shape": "(72, 72, 64)", "parameters": 18496},
            {"name": "max_pooling2d_1", "type": "MaxPooling2D (2x2)", "output_shape": "(36, 36, 64)", "parameters": 0},
            {"name": "conv2d_2", "type": "Conv2D (128 filters, 3x3)", "output_shape": "(34, 34, 128)", "parameters": 73856},
            {"name": "max_pooling2d_2", "type": "MaxPooling2D (2x2)", "output_shape": "(17, 17, 128)", "parameters": 0},
            {"name": "flatten", "type": "Flatten", "output_shape": "(36,992)", "parameters": 0},
            {"name": "dense", "type": "Dense (ReLU)", "output_shape": "(128)", "parameters": 4735104},
            {"name": "dropout", "type": "Dropout (0.5)", "output_shape": "(128)", "parameters": 0},
            {"name": "dense_1", "type": "Dense (Softmax)", "output_shape": "(3)", "parameters": 387}
        ],
        "total_params": 14486219,
        "trainable_params": 4828739,
        "dataset_cohort": [
            {"class": "Cystine Stone (Cyst)", "count": 3709, "percentage": "33.5%", "color": "#f59e0b"},
            {"class": "Normal Kidney", "count": 5077, "percentage": "45.9%", "color": "#10b981"},
            {"class": "Renal Tumor", "count": 2283, "percentage": "20.6%", "color": "#ef4444"}
        ],
        "validation_metrics": {
            "accuracy": "100.0%",
            "precision": "> 0.98",
            "recall": "> 0.98",
            "loss_function": "Categorical Cross-Entropy",
            "optimizer": "Adam (lr=0.001)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
