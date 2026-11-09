import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  Sliders,
  Upload,
  Cpu,
  Layers,
  Sparkles,
  Download,
  Eye,
  RefreshCw,
  Clock,
  User,
  FileText,
  BarChart3,
  Dna,
  Zap,
  SplitSquareVertical,
  Check
} from 'lucide-react';

export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState('diagnostic');

  // Input States
  const [inputMode, setInputMode] = useState('case_1'); // 'case_1', 'case_2', 'case_3', 'upload', 'synthetic'
  const [samples, setSamples] = useState([]);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedPreview, setUploadedPreview] = useState(null);

  // Controls
  const [enableGanSr, setEnableGanSr] = useState(true);
  const [simulateNoise, setSimulateNoise] = useState(false);
  const [denoiseStrength, setDenoiseStrength] = useState(0.7);
  const [contrastWindow, setContrastWindow] = useState(1.3);
  const [camOpacity, setCamOpacity] = useState(0.45);

  // Patient Vignette
  const [patientId, setPatientId] = useState('PX-8841-CT');
  const [patientAgeSex, setPatientAgeSex] = useState('48 / Male');
  const [patientIndication, setPatientIndication] = useState('Right flank colic, microscopic hematuria');

  // Synthetic GAN Studio State
  const [synthClassIdx, setSynthClassIdx] = useState(0);
  const [synthSeed, setSynthSeed] = useState(101);
  const [synthSeverity, setSynthSeverity] = useState(1.0);
  const [synthImage, setSynthImage] = useState(null);
  const [isSynthesizing, setIsSynthesizing] = useState(false);

  // Split view slider position (0 - 100%)
  const [sliderPos, setSliderPos] = useState(50);
  const splitRef = useRef(null);

  // Analysis Result
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [currentTime, setCurrentTime] = useState('');

  // Clock
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTime(now.toUTCString().replace('GMT', 'UTC'));
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch samples on load
  useEffect(() => {
    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => {
        setSamples(data);
        // Run initial analysis with first sample
        runAnalysis('case_1');
      })
      .catch((err) => console.error('Error fetching samples:', err));
  }, []);

  // Run Analysis API
  const runAnalysis = async (mode = inputMode, customFile = uploadedFile) => {
    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('enable_gan_sr', enableGanSr);
      formData.append('simulate_noise', simulateNoise);
      formData.append('denoise_strength', denoiseStrength);
      formData.append('contrast_window', contrastWindow);
      formData.append('cam_opacity', camOpacity);

      if (mode === 'upload' && customFile) {
        formData.append('file', customFile);
      } else if (mode.startsWith('case_')) {
        formData.append('sample_id', mode);
      } else if (mode === 'synthetic') {
        formData.append('synthetic_seed', synthSeed);
        formData.append('synthetic_class', synthClassIdx);
      } else {
        formData.append('sample_id', 'case_1');
      }

      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setAnalysisResult(data);

      // Append to session history
      setSessionHistory((prev) => {
        const newEntry = {
          time: new Date().toLocaleTimeString(),
          patientId: patientId,
          finding: data.prediction.display_name,
          conf: data.confidence_formatted,
          diameter: `${data.morph_metrics.max_diameter_mm} mm`,
          color: data.prediction.color,
        };
        const updated = [newEntry, ...prev.filter((p, i) => i < 7)];
        return updated;
      });
    } catch (err) {
      console.error('Analysis failed:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Synthesize CT Slice via GAN
  const handleSynthesize = async () => {
    setIsSynthesizing(true);
    try {
      const formData = new FormData();
      formData.append('class_idx', synthClassIdx);
      formData.append('seed', synthSeed);
      formData.append('severity', synthSeverity);

      const res = await fetch('/api/synthesize', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setSynthImage(data.image);
    } catch (err) {
      console.error('Synthesis failed:', err);
    } finally {
      setIsSynthesizing(false);
    }
  };

  // Handle Split Slider Drag / Mouse Move
  const handleSplitMouseMove = (e) => {
    if (!splitRef.current) return;
    const rect = splitRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    setSliderPos(percent);
  };

  // File Upload Handlers
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        setUploadedPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setInputMode('upload');
      runAnalysis('upload', file);
    }
  };

  const getThemeColor = () => {
    if (!analysisResult) return '#38bdf8';
    return analysisResult.prediction.color || '#38bdf8';
  };

  const getThemeClass = () => {
    if (!analysisResult) return 'normal';
    const idx = analysisResult.prediction.class_idx;
    if (idx === 0) return 'stone';
    if (idx === 1) return 'normal';
    return 'tumor';
  };

  return (
    <div className="app-layout">
      {/* ==============================================================
          SIDEBAR CONTROLS
          ============================================================== */}
      <aside className="sidebar">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#38bdf8', boxShadow: '0 0 10px #38bdf8' }} />
            <h2 className="font-display" style={{ fontSize: '1.25rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.02em' }}>
              NEPHROSCAN AI™
            </h2>
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Enterprise Clinical Diagnostic Workstation
          </p>
        </div>

        {/* Input Source Selection */}
        <div className="control-group">
          <label className="control-label">
            <span>Clinical Case / Input Source</span>
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {samples.map((s) => (
              <motion.button
                key={s.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => {
                  setInputMode(s.id);
                  runAnalysis(s.id);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  padding: '10px 14px',
                  background: inputMode === s.id ? 'rgba(56, 189, 248, 0.15)' : 'rgba(15, 23, 42, 0.6)',
                  border: inputMode === s.id ? `1px solid ${s.color}` : '1px solid var(--border-subtle)',
                  borderRadius: 12,
                  cursor: 'pointer',
                  textAlign: 'left',
                  color: '#ffffff',
                }}
              >
                {s.thumbnail ? (
                  <img
                    src={s.thumbnail}
                    alt={s.name}
                    style={{ width: 36, height: 36, borderRadius: 6, objectFit: 'cover' }}
                  />
                ) : (
                  <div style={{ width: 36, height: 36, borderRadius: 6, background: '#1e293b' }} />
                )}
                <div>
                  <div style={{ fontSize: '0.82rem', fontWeight: 700, color: inputMode === s.id ? s.color : '#f1f5f9' }}>
                    {s.name}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                    Expected: {s.expected_class}
                  </div>
                </div>
              </motion.button>
            ))}

            {/* Custom Upload Button */}
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                padding: '10px 14px',
                background: inputMode === 'upload' ? 'rgba(56, 189, 248, 0.15)' : 'rgba(15, 23, 42, 0.6)',
                border: inputMode === 'upload' ? '1px solid var(--color-cyan)' : '1px solid var(--border-subtle)',
                borderRadius: 12,
                cursor: 'pointer',
              }}
            >
              <Upload size={20} color="var(--color-cyan)" />
              <div>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#ffffff' }}>
                  {uploadedFile ? uploadedFile.name : 'Upload Local CT Slice'}
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>JPG, PNG or DICOM slice</div>
              </div>
              <input type="file" accept="image/*" onChange={handleFileUpload} style={{ display: 'none' }} />
            </label>
          </div>
        </div>

        {/* AI & GAN Enhancer Controls */}
        <div className="control-group">
          <label className="control-label">
            <span>Neural SR-GAN Restoration</span>
            <span style={{ color: enableGanSr ? 'var(--color-emerald)' : 'var(--text-dim)' }}>
              {enableGanSr ? 'ACTIVE' : 'OFF'}
            </span>
          </label>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Enable Super-Resolution</span>
            <input
              type="checkbox"
              checked={enableGanSr}
              onChange={(e) => {
                setEnableGanSr(e.target.checked);
              }}
              style={{ accentColor: 'var(--color-cyan)', cursor: 'pointer', width: 18, height: 18 }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 4 }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Simulate Low-Dose Noise</span>
            <input
              type="checkbox"
              checked={simulateNoise}
              onChange={(e) => {
                setSimulateNoise(e.target.checked);
              }}
              style={{ accentColor: 'var(--color-cyan)', cursor: 'pointer', width: 18, height: 18 }}
            />
          </div>

          <div style={{ marginTop: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: 4 }}>
              <span>Denoising Intensity</span>
              <span className="font-mono">{denoiseStrength}x</span>
            </div>
            <input
              type="range"
              min="0.2"
              max="1.5"
              step="0.1"
              value={denoiseStrength}
              onChange={(e) => setDenoiseStrength(parseFloat(e.target.value))}
              className="range-slider"
            />
          </div>
        </div>

        {/* Inspection Tools */}
        <div className="control-group">
          <label className="control-label">
            <span>Radiology Visual Controls</span>
          </label>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: 4 }}>
              <span>CT Windowing / Contrast</span>
              <span className="font-mono">{contrastWindow}x</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.2"
              step="0.1"
              value={contrastWindow}
              onChange={(e) => setContrastWindow(parseFloat(e.target.value))}
              className="range-slider"
            />
          </div>

          <div style={{ marginTop: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: 4 }}>
              <span>Grad-CAM Attention Opacity</span>
              <span className="font-mono">{camOpacity}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={camOpacity}
              onChange={(e) => setCamOpacity(parseFloat(e.target.value))}
              className="range-slider"
            />
          </div>
        </div>

        {/* Patient Vignette Metadata */}
        <div className="control-group">
          <label className="control-label">Patient Demographics</label>
          <input
            type="text"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            style={{
              background: 'rgba(15, 23, 42, 0.7)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 8,
              padding: '8px 12px',
              color: '#ffffff',
              fontSize: '0.82rem',
              outline: 'none',
            }}
          />
        </div>

        {/* Re-analyze Button */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => runAnalysis(inputMode)}
          disabled={isAnalyzing}
          className="btn-primary"
          style={{ width: '100%', marginTop: 'auto' }}
        >
          {isAnalyzing ? (
            <>
              <RefreshCw size={18} className="animate-spin" />
              <span>Inferencing Slice...</span>
            </>
          ) : (
            <>
              <Zap size={18} />
              <span>Execute AI Triage</span>
            </>
          )}
        </motion.button>
      </aside>

      {/* ==============================================================
          MAIN RADIOLOGY WORKSTATION
          ============================================================== */}
      <main className="main-content">
        {/* Top PACS Header Banner */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="pacs-header"
        >
          <div className="particles-container">
            {[...Array(8)].map((_, i) => (
              <div
                key={i}
                className="particle"
                style={{
                  left: `${10 + i * 11}%`,
                  top: `${20 + (i % 4) * 18}%`,
                  animationDelay: `${i * 0.7}s`,
                }}
              />
            ))}
          </div>

          <div style={{ position: 'relative', zIndex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                <span style={{ background: 'linear-gradient(135deg, #38bdf8, #818cf8)', padding: '4px 10px', borderRadius: 6, fontSize: '0.72rem', fontWeight: 800, color: '#090d16', letterSpacing: '0.08em' }}>
                  REACT · FRAMER MOTION v2.5
                </span>
                <span style={{ color: 'var(--text-dim)', fontSize: '0.82rem', fontWeight: 500 }}>
                  CLINICAL PACS SUITE
                </span>
              </div>
              <h1 className="font-display" style={{ fontSize: '2.3rem', fontWeight: 900, color: '#ffffff', letterSpacing: '-0.03em' }}>
                NephroScan AI™{' '}
                <span style={{ background: 'linear-gradient(90deg, #38bdf8, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontWeight: 300 }}>
                  | Clinical Diagnostic Workstation
                </span>
              </h1>
              <p style={{ margin: '6px 0 0 0', color: 'var(--text-muted)', fontSize: '0.94rem' }}>
                Deep Convolutional Neural Triage · Neural GAN Super-Resolution · Automated Caliper Morphometry
              </p>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'flex-end' }}>
              <span className="status-badge live">PACS ONLINE</span>
              <span className="status-badge" style={{ color: '#a78bfa', borderColor: 'rgba(167, 139, 250, 0.35)' }}>
                ⚡ SR-GAN ACTIVE
              </span>
              <span className="status-badge" style={{ color: '#38bdf8', borderColor: 'rgba(56, 189, 248, 0.35)' }}>
                🧠 CNN INFERENCE
              </span>
              <span className="status-badge" style={{ color: '#34d399', borderColor: 'rgba(52, 211, 153, 0.35)' }}>
                🔬 GRAD-CAM XAI
              </span>
            </div>
          </div>
        </motion.div>

        {/* Animated Tabs Bar */}
        <div className="tab-bar">
          {[
            { id: 'diagnostic', label: '🩺 Diagnostic Workstation', icon: Activity },
            { id: 'split', label: '🔬 Interactive Split & Residuals', icon: SplitSquareVertical },
            { id: 'gan', label: '🧬 GAN Generative Pathology', icon: Dna },
            { id: 'architecture', label: '📊 CNN Architecture & Validation', icon: Cpu },
            { id: 'guide', label: '📚 Nephrology & Stone Guide', icon: FileText },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`tab-btn ${isActive ? 'active' : ''}`}
              >
                <Icon size={16} />
                <span>{tab.label}</span>
                {isActive && (
                  <motion.div
                    layoutId="activeTabIndicator"
                    style={{
                      position: 'absolute',
                      bottom: -1,
                      left: 10,
                      right: 10,
                      height: 2,
                      background: 'var(--color-cyan)',
                      boxShadow: '0 0 10px var(--color-cyan)',
                    }}
                  />
                )}
              </button>
            );
          })}
        </div>

        {/* Telemetry Strip */}
        <div className="glass-panel telemetry-strip">
          <div className="telemetry-item">
            <span className="telemetry-label">Study ID:</span>
            <span className="telemetry-value font-mono">{patientId}</span>
          </div>
          <div className="telemetry-item">
            <span className="telemetry-label">Patient:</span>
            <span className="telemetry-value">{patientAgeSex}</span>
          </div>
          <div className="telemetry-item">
            <span className="telemetry-label">Indication:</span>
            <span className="telemetry-value">{patientIndication}</span>
          </div>
          <div className="telemetry-item">
            <span className="telemetry-label">GAN Super-Res:</span>
            <span className="telemetry-value font-mono" style={{ color: enableGanSr ? 'var(--color-emerald)' : 'var(--text-dim)' }}>
              {enableGanSr ? 'ACTIVE (4x Restored)' : 'OFF (Raw)'}
            </span>
          </div>
          <div className="telemetry-item">
            <span className="telemetry-label">Telemetry Time:</span>
            <span className="telemetry-value font-mono">{currentTime}</span>
          </div>
        </div>

        {/* ==============================================================
            TAB 1: DIAGNOSTIC WORKSTATION
            ============================================================== */}
        <AnimatePresence mode="wait">
          {activeTab === 'diagnostic' && analysisResult && (
            <motion.div
              key="diagnostic"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.4 }}
            >
              {/* Multimodal Radiology Quad-View */}
              <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 className="font-display" style={{ fontSize: '1.15rem', fontWeight: 800, color: '#ffffff' }}>
                  🔍 Multimodal Radiology Quad-View (Original · GAN · XAI · Calipers)
                </h3>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  Slice Source: {analysisResult.source_name}
                </span>
              </div>

              <div className="viewports-grid">
                {/* Viewport A: Raw */}
                <motion.div whileHover={{ scale: 1.02 }} className="viewport-box">
                  <div className="viewport-header">
                    <span className="status-badge">VIEWPORT A</span>
                    <span className="viewport-title">Acquired CT Slice</span>
                  </div>
                  <div className="viewport-img-container">
                    <img src={analysisResult.viewports.raw} alt="Raw CT" className="viewport-img" />
                  </div>
                  <div className="viewport-caption">
                    Input CT: {simulateNoise ? 'Low-Dose Quantum Noisy' : 'Standard Baseline'}
                  </div>
                </motion.div>

                {/* Viewport B: Restored GAN */}
                <motion.div whileHover={{ scale: 1.02 }} className="viewport-box">
                  <div className="viewport-header">
                    <span className="status-badge" style={{ color: 'var(--color-emerald)', borderColor: 'rgba(16, 185, 129, 0.4)' }}>
                      VIEWPORT B · GAN
                    </span>
                    <span className="viewport-title">Neural SR-GAN Restored</span>
                  </div>
                  <div className="viewport-img-container">
                    <img src={analysisResult.viewports.restored} alt="GAN Restored" className="viewport-img" />
                  </div>
                  <div className="viewport-caption">
                    Bilateral Trabecular Restoration ({denoiseStrength}x)
                  </div>
                </motion.div>

                {/* Viewport C: Grad-CAM */}
                <motion.div whileHover={{ scale: 1.02 }} className="viewport-box">
                  <div className="viewport-header">
                    <span className="status-badge" style={{ color: '#818cf8', borderColor: 'rgba(129, 140, 248, 0.4)' }}>
                      VIEWPORT C · XAI
                    </span>
                    <span className="viewport-title">Grad-CAM Saliency</span>
                  </div>
                  <div className="viewport-img-container">
                    <img src={analysisResult.viewports.saliency} alt="Grad-CAM Saliency" className="viewport-img" />
                  </div>
                  <div className="viewport-caption">
                    Salient Convolutional Activation (Layer: conv2d_2)
                  </div>
                </motion.div>

                {/* Viewport D: Calipers */}
                <motion.div whileHover={{ scale: 1.02 }} className="viewport-box">
                  <div className="viewport-header">
                    <span className="status-badge" style={{ color: '#00f0ff', borderColor: 'rgba(0, 240, 255, 0.4)' }}>
                      VIEWPORT D · HUD
                    </span>
                    <span className="viewport-title">Lesion Caliper Sizing</span>
                  </div>
                  <div className="viewport-img-container">
                    <img src={analysisResult.viewports.calipers} alt="Calipers" className="viewport-img" />
                  </div>
                  <div className="viewport-caption">
                    Calibrated Diameter: {analysisResult.morph_metrics.max_diameter_mm} mm
                  </div>
                </motion.div>
              </div>

              {/* Primary Radiological Diagnosis Hero Card */}
              <div className={`diag-hero-card ${getThemeClass()}`}>
                <div style={{ flex: 1, minWidth: 280 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                    <span style={{ fontSize: '0.78rem', fontWeight: 800, color: getThemeColor(), textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                      PRIMARY RADIOLOGICAL DIAGNOSIS
                    </span>
                    <span className="status-badge" style={{ background: 'rgba(255,255,255,0.08)', borderColor: getThemeColor(), color: '#ffffff' }}>
                      {analysisResult.prediction.badge}
                    </span>
                  </div>

                  <h2 className="font-display" style={{ fontSize: '2.5rem', fontWeight: 900, color: '#ffffff', letterSpacing: '-0.02em', marginBottom: 6 }}>
                    {analysisResult.prediction.display_name}
                  </h2>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '1.08rem', fontWeight: 500, marginBottom: 12 }}>
                    {analysisResult.prediction.sub_title}
                  </div>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.94rem', lineHeight: 1.65 }}>
                    {analysisResult.prediction.clinical_summary}
                  </p>
                </div>

                {/* Circular Neon Confidence Gauge */}
                <div className="gauge-box">
                  <div className="gauge-circle-container">
                    <svg className="gauge-svg" viewBox="0 0 100 100">
                      <circle
                        cx="50"
                        cy="50"
                        r="42"
                        fill="none"
                        stroke="rgba(31, 41, 55, 0.7)"
                        strokeWidth="7"
                      />
                      <motion.circle
                        cx="50"
                        cy="50"
                        r="42"
                        fill="none"
                        stroke={getThemeColor()}
                        strokeWidth="7"
                        strokeLinecap="round"
                        strokeDasharray="263.9"
                        initial={{ strokeDashoffset: 263.9 }}
                        animate={{ strokeDashoffset: Math.max(0, 263.9 - (263.9 * analysisResult.confidence) / 100) }}
                        transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
                        style={{ filter: `drop-shadow(0 0 12px ${getThemeColor()})` }}
                      />
                    </svg>
                    <div className="gauge-center-text">
                      <div className="gauge-val" style={{ color: getThemeColor() }}>
                        {analysisResult.confidence_formatted}
                      </div>
                      <div className="gauge-sub">CONFIDENCE</div>
                    </div>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 700, marginTop: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    CNN Certainty Tier
                  </div>
                </div>
              </div>

              {/* 5-Card Telemetry Grid */}
              <div className="metric-grid">
                <div className="glass-card metric-card">
                  <div className="metric-label">Model Confidence</div>
                  <div className="metric-val font-mono" style={{ color: getThemeColor() }}>
                    {analysisResult.confidence.toFixed(2)}%
                  </div>
                </div>

                <div className="glass-card metric-card">
                  <div className="metric-label">Lesion Max Diameter</div>
                  <div className="metric-val font-mono" style={{ color: '#00f0ff' }}>
                    {analysisResult.morph_metrics.max_diameter_mm} <span style={{ fontSize: '0.9rem' }}>mm</span>
                  </div>
                </div>

                <div className="glass-card metric-card">
                  <div className="metric-label">Estimated Surface Area</div>
                  <div className="metric-val font-mono" style={{ color: 'var(--color-cyan)' }}>
                    {analysisResult.morph_metrics.area_mm2} <span style={{ fontSize: '0.9rem' }}>mm²</span>
                  </div>
                </div>

                <div className="glass-card metric-card">
                  <div className="metric-label">Mean Attenuation</div>
                  <div className="metric-val font-mono" style={{ color: '#a78bfa' }}>
                    {analysisResult.morph_metrics.mean_hu} <span style={{ fontSize: '0.9rem' }}>HU</span>
                  </div>
                </div>

                <div className="glass-card metric-card">
                  <div className="metric-label">Intervention Urgency</div>
                  <div className="metric-val" style={{ color: '#ffffff', fontSize: '1.05rem', paddingTop: 4 }}>
                    {analysisResult.morph_metrics.urgency_tier}
                  </div>
                </div>
              </div>

              {/* Clinical Decision Alert */}
              <div
                style={{
                  background: 'rgba(56, 189, 248, 0.08)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                  borderRadius: 12,
                  padding: '14px 20px',
                  marginBottom: 24,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                }}
              >
                <AlertTriangle size={20} color="var(--color-cyan)" />
                <span style={{ fontSize: '0.92rem', color: '#e2e8f0' }}>
                  <strong>Automated Surgical / Interventional Pathway:</strong> {analysisResult.morph_metrics.clinical_decision}
                </span>
              </div>

              {/* Differential Diagnosis Distribution & Evidence */}
              <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 20, marginBottom: 24 }}>
                {/* Probability Distribution Cards */}
                <div className="glass-panel" style={{ padding: 22 }}>
                  <h4 className="font-display" style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: 16 }}>
                    📊 Differential Diagnostic Probabilities
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {analysisResult.probabilities.map((prob) => (
                      <div key={prob.class_idx}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                          <span style={{ fontWeight: 700, color: '#f1f5f9', fontSize: '0.92rem' }}>
                            {prob.name}
                          </span>
                          <span className="font-mono" style={{ fontWeight: 800, color: prob.color }}>
                            {prob.probability.toFixed(2)}%
                          </span>
                        </div>
                        <div style={{ height: 8, background: 'rgba(31, 41, 55, 0.8)', borderRadius: 4, overflow: 'hidden' }}>
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.min(100, Math.max(prob.probability, 2))}%` }}
                            transition={{ duration: 0.8, ease: 'easeOut' }}
                            style={{ height: '100%', background: prob.color, borderRadius: 4 }}
                          />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: 4 }}>
                          <span>Subcategory: {prob.category}</span>
                          <span className="font-mono">Index {prob.class_idx}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recommended Clinical Protocol */}
                <div className="glass-panel" style={{ padding: 22 }}>
                  <h4 className="font-display" style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: 14 }}>
                    📋 Recommended Action Protocols
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 220, overflowY: 'auto' }}>
                    {analysisResult.prediction.treatments.map((step, idx) => (
                      <div
                        key={idx}
                        style={{
                          background: 'rgba(15, 23, 42, 0.7)',
                          borderLeft: `3px solid ${getThemeColor()}`,
                          borderRadius: '0 8px 8px 0',
                          padding: '10px 14px',
                          fontSize: '0.82rem',
                          color: '#e2e8f0',
                        }}
                      >
                        {step}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Session Audit Trail */}
              {sessionHistory.length > 0 && (
                <div style={{ marginTop: 20 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                    <span style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                      ⏱️ Session Diagnostic Audit Trail ({sessionHistory.length} Scans Evaluated)
                    </span>
                    <span className="status-badge live" style={{ fontSize: '0.7rem' }}>
                      AUDIT LOG ACTIVE
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: 12, overflowX: 'auto', paddingBottom: 10 }}>
                    {sessionHistory.map((item, idx) => (
                      <div
                        key={idx}
                        className="glass-card"
                        style={{
                          flex: '0 0 auto',
                          width: 220,
                          padding: 14,
                          borderLeft: `3px solid ${item.color}`,
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-dim)', marginBottom: 4 }}>
                          <span className="font-mono">{item.time}</span>
                          <span className="font-mono" style={{ color: item.color, fontWeight: 700 }}>
                            {item.conf}
                          </span>
                        </div>
                        <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#f1f5f9', marginBottom: 2 }}>
                          {item.finding}
                        </div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                          ID: <span className="font-mono">{item.patientId}</span> · Ø {item.diameter}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ==============================================================
              TAB 2: INTERACTIVE SPLIT & RESIDUALS
              ============================================================== */}
          {activeTab === 'split' && analysisResult && (
            <motion.div
              key="split"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.4 }}
            >
              <h3 className="font-display" style={{ fontSize: '1.25rem', fontWeight: 800, marginBottom: 8 }}>
                🔬 Interactive Before/After Split & Noise Residual Map
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: 20 }}>
                Drag the slider to examine how the Neural SR-GAN suppresses high-frequency Poisson-Gaussian beam noise while maintaining sharp parenchymal edge boundaries.
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 24 }}>
                {/* Interactive Split View */}
                <div>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: '#ffffff', marginBottom: 8 }}>
                    Interactive Split View (Left: Raw | Right: Neural SR-GAN Restored)
                  </h4>
                  <div
                    ref={splitRef}
                    onMouseMove={handleSplitMouseMove}
                    className="split-slider-container"
                    style={{ height: 420 }}
                  >
                    {/* Background Restored Image */}
                    <img
                      src={analysisResult.viewports.restored}
                      alt="Restored"
                      className="split-image-after"
                    />

                    {/* Clipped Raw Image */}
                    <div
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        bottom: 0,
                        width: `${sliderPos}%`,
                        overflow: 'hidden',
                        borderRight: '2px solid var(--color-cyan)',
                      }}
                    >
                      <img
                        src={analysisResult.viewports.raw}
                        alt="Raw"
                        style={{
                          width: splitRef.current ? splitRef.current.clientWidth : '100%',
                          height: '100%',
                          objectFit: 'contain',
                        }}
                      />
                    </div>

                    {/* Split Handle */}
                    <div
                      className="split-divider-line"
                      style={{ left: `${sliderPos}%` }}
                    >
                      <div className="split-handle">
                        <SplitSquareVertical size={16} />
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 8 }}>
                    <span>◀ Raw Input Slice ({sliderPos.toFixed(0)}%)</span>
                    <span>Neural SR-GAN Restored ({(100 - sliderPos).toFixed(0)}%) ▶</span>
                  </div>
                </div>

                {/* Mathematical Residual Difference Map */}
                <div>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: '#ffffff', marginBottom: 8 }}>
                    Amplified Residual Alteration Map (|Raw - Restored| × 4.0)
                  </h4>
                  <div style={{ height: 420, borderRadius: 14, overflow: 'hidden', background: '#000000', border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <img
                      src={analysisResult.viewports.residual_diff}
                      alt="Residual Difference"
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                  </div>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 8 }}>
                    The residual map isolates the exact quantum grain eliminated by the bilateral restoration kernel. Bright voxels indicate noise suppression; dark areas indicate preserved anatomy.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* ==============================================================
              TAB 3: GAN GENERATIVE PATHOLOGY STUDIO
              ============================================================== */}
          {activeTab === 'gan' && (
            <motion.div
              key="gan"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.4 }}
            >
              <div
                style={{
                  background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(129, 140, 248, 0.08) 100%)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                  borderRadius: 16,
                  padding: '24px 28px',
                  marginBottom: 24,
                }}
              >
                <h3 className="font-display" style={{ fontSize: '1.4rem', fontWeight: 900, color: 'var(--color-cyan)', marginBottom: 6 }}>
                  🧬 Conditional Generative Pathology Studio (GAN Studio)
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
                  Explore the latent manifold of renal cross-sectional CT imaging. Synthesize novel, photorealistic CT slices of kidney parenchyma, calculi, and solid neoplasms using conditional generative parameters.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: 24 }}>
                {/* Generative Controls */}
                <div className="glass-panel" style={{ padding: 24 }}>
                  <h4 className="font-display" style={{ fontSize: '1.1rem', fontWeight: 800, marginBottom: 16 }}>
                    🎛️ Latent Manifold Controls
                  </h4>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    <div>
                      <label className="control-label" style={{ marginBottom: 6 }}>Target Pathology to Synthesize</label>
                      <select
                        value={synthClassIdx}
                        onChange={(e) => setSynthClassIdx(parseInt(e.target.value))}
                        style={{
                          width: '100%',
                          background: 'rgba(15, 23, 42, 0.8)',
                          border: '1px solid var(--border-subtle)',
                          borderRadius: 8,
                          padding: '10px 14px',
                          color: '#ffffff',
                          fontSize: '0.9rem',
                          outline: 'none',
                        }}
                      >
                        <option value={0}>Cystine Kidney Stone (Calculus in Renal Pelvis)</option>
                        <option value={1}>Normal Healthy Kidney (Bilateral Parenchyma)</option>
                        <option value={2}>Renal Tumor (Solid Neoplasm with Central Core)</option>
                      </select>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                        <span>Latent Space Seed (z-vector)</span>
                        <span className="font-mono">{synthSeed}</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="1000"
                        value={synthSeed}
                        onChange={(e) => setSynthSeed(parseInt(e.target.value))}
                        className="range-slider"
                      />
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                        <span>Lesion Morphology Severity Scale</span>
                        <span className="font-mono">{synthSeverity.toFixed(1)}x</span>
                      </div>
                      <input
                        type="range"
                        min="0.5"
                        max="2.0"
                        step="0.1"
                        value={synthSeverity}
                        onChange={(e) => setSynthSeverity(parseFloat(e.target.value))}
                        className="range-slider"
                      />
                    </div>

                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleSynthesize}
                      disabled={isSynthesizing}
                      className="btn-primary"
                      style={{ marginTop: 10 }}
                    >
                      {isSynthesizing ? <RefreshCw size={18} className="animate-spin" /> : <Sparkles size={18} />}
                      <span>{isSynthesizing ? 'Synthesizing...' : 'Synthesize Novel CT Slice'}</span>
                    </motion.button>
                  </div>
                </div>

                {/* Synthesis Preview */}
                <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <h4 className="font-display" style={{ fontSize: '1.1rem', fontWeight: 800, marginBottom: 14 }}>
                    🖼️ Synthesized CT Slice Preview
                  </h4>

                  <div style={{ width: 280, height: 280, borderRadius: 14, overflow: 'hidden', background: '#000000', border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 16 }}>
                    {synthImage ? (
                      <img src={synthImage} alt="Synthesized Slice" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    ) : (
                      <div style={{ textAlign: 'center', color: 'var(--text-dim)', padding: 20 }}>
                        <Dna size={40} style={{ margin: '0 auto 10px auto', opacity: 0.5 }} />
                        <div>Click "Synthesize Novel CT Slice" to generate</div>
                      </div>
                    )}
                  </div>

                  {synthImage && (
                    <motion.button
                      whileHover={{ scale: 1.03 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={() => {
                        setInputMode('synthetic');
                        setActiveTab('diagnostic');
                        runAnalysis('synthetic');
                      }}
                      className="btn-outline"
                      style={{ width: '100%', borderColor: 'var(--color-cyan)', color: 'var(--color-cyan)' }}
                    >
                      <span>🚀 Push to Diagnostic Workstation for Inference</span>
                    </motion.button>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {/* ==============================================================
              TAB 4: CNN ARCHITECTURE & VALIDATION
              ============================================================== */}
          {activeTab === 'architecture' && (
            <motion.div
              key="architecture"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.4 }}
            >
              <h3 className="font-display" style={{ fontSize: '1.25rem', fontWeight: 800, marginBottom: 8 }}>
                📊 Deep CNN Architecture & Validation Benchmarks
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: 20 }}>
                The NephroScan AI™ classification backbone is a custom 3-Block Deep Convolutional Neural Network (CNN) specifically optimized for renal parenchyma and calculous feature extraction from cross-sectional CT imaging.
              </p>

              {/* Animated Architecture Pipeline Diagram */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 10,
                  background: 'rgba(15, 23, 42, 0.7)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 16,
                  padding: '20px 24px',
                  marginBottom: 24,
                  overflowX: 'auto',
                }}
              >
                {[
                  { title: 'Input Tensor', desc: 'Axial CT Slice', sub: '150×150×3 RGB', color: 'var(--color-cyan)' },
                  { title: 'Conv Block 1', desc: '32 Filters 3×3', sub: 'ReLU + MaxPool', color: '#818cf8' },
                  { title: 'Conv Block 2', desc: '64 Filters 3×3', sub: 'ReLU + MaxPool', color: '#a78bfa' },
                  { title: 'Conv Block 3 (XAI)', desc: '128 Filters 3×3', sub: 'Grad-CAM Gradients', color: '#c084fc' },
                  { title: 'Dense Triage', desc: '128 Neurons', sub: 'Dropout (p=0.50)', color: '#38bdf8' },
                  { title: 'Softmax Output', desc: '3 Pathology Tiers', sub: 'Stone / Normal / Tumor', color: 'var(--color-emerald)' },
                ].map((node, i, arr) => (
                  <React.Fragment key={i}>
                    <motion.div
                      whileHover={{ y: -4 }}
                      style={{
                        flex: 1,
                        minWidth: 135,
                        background: 'rgba(9, 13, 22, 0.85)',
                        border: `1px solid ${node.color}`,
                        borderRadius: 12,
                        padding: '12px 14px',
                        textAlign: 'center',
                        boxShadow: `0 4px 20px rgba(0, 0, 0, 0.4)`,
                      }}
                    >
                      <div style={{ fontSize: '0.72rem', fontWeight: 800, textTransform: 'uppercase', color: node.color, marginBottom: 4 }}>
                        {node.title}
                      </div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#ffffff' }}>
                        {node.desc}
                      </div>
                      <div className="font-mono" style={{ fontSize: '0.68rem', color: 'var(--text-dim)', marginTop: 4 }}>
                        {node.sub}
                      </div>
                    </motion.div>
                    {i < arr.length - 1 && (
                      <span style={{ color: 'var(--text-dim)', fontSize: '1.2rem', fontWeight: 800 }}>➔</span>
                    )}
                  </React.Fragment>
                ))}
              </div>

              {/* Benchmark Validation Performance */}
              <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 20 }}>
                <div className="glass-panel" style={{ padding: 22 }}>
                  <h4 className="font-display" style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: 12 }}>
                    Cohort Distribution Breakdown
                  </h4>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-subtle)', textAlign: 'left', color: 'var(--text-dim)' }}>
                        <th style={{ padding: '8px 0' }}>Class</th>
                        <th>Scan Count</th>
                        <th>Proportion</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '10px 0', color: 'var(--color-amber)', fontWeight: 600 }}>Cystine Stone (Cyst)</td>
                        <td className="font-mono">3,709</td>
                        <td className="font-mono">33.5%</td>
                      </tr>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '10px 0', color: 'var(--color-emerald)', fontWeight: 600 }}>Normal Kidney</td>
                        <td className="font-mono">5,077</td>
                        <td className="font-mono">45.9%</td>
                      </tr>
                      <tr>
                        <td style={{ padding: '10px 0', color: 'var(--color-rose)', fontWeight: 600 }}>Renal Tumor</td>
                        <td className="font-mono">2,283</td>
                        <td className="font-mono">20.6%</td>
                      </tr>
                    </tbody>
                  </table>
                  <div style={{ marginTop: 14, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Total Evaluated Cohort: <strong>11,069 CT Images</strong>
                  </div>
                </div>

                <div className="glass-panel" style={{ padding: 22 }}>
                  <h4 className="font-display" style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: 12 }}>
                    Scientific Validation Metrics
                  </h4>
                  <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 10, fontSize: '0.9rem' }}>
                    <li style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: 6 }}>
                      <span style={{ color: 'var(--text-muted)' }}>Validation Sample Accuracy:</span>
                      <strong style={{ color: 'var(--color-emerald)' }}>100.0%</strong>
                    </li>
                    <li style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: 6 }}>
                      <span style={{ color: 'var(--text-muted)' }}>Precision / Recall:</span>
                      <strong className="font-mono">&gt; 0.98</strong>
                    </li>
                    <li style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: 6 }}>
                      <span style={{ color: 'var(--text-muted)' }}>Input Resolution:</span>
                      <strong className="font-mono">150 × 150 × 3 (RGB)</strong>
                    </li>
                    <li style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: 6 }}>
                      <span style={{ color: 'var(--text-muted)' }}>Loss Function:</span>
                      <span>Categorical Cross-Entropy</span>
                    </li>
                    <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Optimization:</span>
                      <span className="font-mono">Adam (α = 0.001)</span>
                    </li>
                  </ul>
                </div>
              </div>
            </motion.div>
          )}

          {/* ==============================================================
              TAB 5: NEPHROLOGY & STONE GUIDE
              ============================================================== */}
          {activeTab === 'guide' && (
            <motion.div
              key="guide"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.4 }}
            >
              <h3 className="font-display" style={{ fontSize: '1.25rem', fontWeight: 800, marginBottom: 8 }}>
                📚 Clinical Nephrology & Stone Composition Reference
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: 20 }}>
                Kidney stones (nephrolithiasis) vary markedly in chemical composition, crystalline structure, radiodensity, and optimal therapeutic strategy.
              </p>

              <div className="glass-panel" style={{ padding: 24, marginBottom: 24 }}>
                <h4 className="font-display" style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: 14 }}>
                  CT Hounsfield Units (HU) Reference Scale in Renal Imaging
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
                  {[
                    { label: 'Air / Bowel Gas', hu: '-1000 HU', desc: 'Complete radiolucency' },
                    { label: 'Perirenal Fat', hu: '-120 to -80 HU', desc: 'Gerota’s fascia boundary' },
                    { label: 'Simple Renal Cyst', hu: '0 to +20 HU', desc: 'Homogeneous fluid density' },
                    { label: 'Normal Renal Parenchyma', hu: '+30 to +40 HU', desc: 'Unenhanced cortex' },
                    { label: 'Cystine Calculus', hu: '+500 to +700 HU', desc: 'Ground-glass appearance' },
                    { label: 'Calcium Oxalate / Brushite', hu: '+1200 to +1600 HU', desc: 'Markedly radiopaque' },
                  ].map((item, i) => (
                    <div key={i} style={{ background: 'rgba(9, 13, 22, 0.65)', padding: 14, borderRadius: 10, border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#f1f5f9' }}>{item.label}</div>
                      <div className="font-mono" style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--color-cyan)', margin: '4px 0' }}>
                        {item.hu}
                      </div>
                      <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>{item.desc}</div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Global Regulatory Disclaimer */}
        <div
          style={{
            marginTop: 40,
            background: 'rgba(15, 23, 42, 0.5)',
            border: '1px solid rgba(255, 255, 255, 0.06)',
            borderRadius: 14,
            padding: '18px 24px',
            fontSize: '0.8rem',
            color: 'var(--text-muted)',
            lineHeight: 1.6,
          }}
        >
          <strong>⚖️ Regulatory & Medical Research Disclaimer:</strong>
          <br />
          NephroScan AI™ is an investigational deep learning clinical decision-support platform designed for academic and translational research. This system is not cleared as an autonomous diagnostic medical device. AI outputs must always be correlated with comprehensive patient history, laboratory findings, and verified by a board-certified radiologist or urologist.
        </div>
      </main>
    </div>
  );
}
