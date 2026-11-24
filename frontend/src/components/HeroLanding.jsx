import React, { useState, useMemo } from 'react';
import { 
  ShieldCheck, 
  Cpu, 
  ArrowRight, 
  Layers, 
  Activity, 
  Zap, 
  Database, 
  CheckCircle2, 
  Sparkles,
  Compass,
  BarChart3,
  Scale
} from 'lucide-react';
import { motion } from 'framer-motion';
import { HoverLogoVideo } from './LogoVideo.jsx';

/**
 * Twinkling Starfield for Hero Page background only.
 * Does not carry on to subsequent sections.
 */
function TwinklingStars({ count = 50 }) {
  const stars = useMemo(() => {
    return Array.from({ length: count }, (_, i) => {
      const top = ((i * 37) % 94) + 3; // 3% to 97%
      const left = ((i * 61) % 96) + 2; // 2% to 98%
      const size = (i % 4 === 0) ? 2.5 : (i % 3 === 0) ? 2 : 1.5;
      const duration = (2.2 + ((i * 7) % 25) / 10).toFixed(1) + 's';
      const delay = (((i * 11) % 30) / 10).toFixed(1) + 's';
      const color = (i % 6 === 0) 
        ? 'rgba(6, 182, 212, 0.95)' // cyan star
        : (i % 5 === 0) 
        ? 'rgba(168, 85, 247, 0.95)' // purple star
        : 'rgba(255, 255, 255, 0.9)'; // white star

      return { id: i, top: `${top}%`, left: `${left}%`, size, duration, delay, color };
    });
  }, [count]);

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden z-0">
      {stars.map((s) => (
        <span
          key={s.id}
          className="twinkle-star"
          style={{
            top: s.top,
            left: s.left,
            width: `${s.size}px`,
            height: `${s.size}px`,
            backgroundColor: s.color,
            boxShadow: `0 0 ${s.size * 2.5}px ${s.color}`,
            '--duration': s.duration,
            '--delay': s.delay,
          }}
        />
      ))}
    </div>
  );
}

export default function HeroLanding({ onLaunchConsole, sampleScenes = [] }) {
  const [activePreviewIndex, setActivePreviewIndex] = useState(0);

  const previewItem = sampleScenes[activePreviewIndex] || {
    pair_id: 'p11b4ad02',
    disaster: 'Midwest Flooding',
    disaster_type: 'Flood Incident',
    total_buildings: 137,
    total_damaged: 35,
    damage_breakdown: { 'no-damage': 102, 'minor-damage': 15, 'major-damage': 17, 'destroyed': 3 },
    pre_url: '/api/files/raw/Dataset/CIT_Hackathon_PS04_aerial/practice/images/p11b4ad02_pre.jpg',
    post_url: '/api/files/raw/Dataset/CIT_Hackathon_PS04_aerial/practice/images/p11b4ad02_post.jpg',
    overlay_url: '/api/files/viz/p11b4ad02_damage.png',
  };

  const smoothScrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-[#07090E] text-slate-100 bg-grid-pattern relative overflow-hidden font-sans">
      {/* Background Ambient Glows */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[850px] h-[450px] bg-gradient-to-b from-purple-600/18 via-cyan-500/10 to-transparent blur-[120px] pointer-events-none" />
      <div className="absolute top-[1000px] -left-40 w-[600px] h-[500px] bg-cyan-600/10 blur-[140px] pointer-events-none" />
      <div className="absolute top-[1600px] -right-40 w-[600px] h-[500px] bg-purple-600/10 blur-[140px] pointer-events-none" />

      {/* Navigation Header */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#07090E]/85 border-b border-white/[0.06]">
        <div className="max-w-7xl mx-auto px-6 h-24 sm:h-28 flex items-center justify-between">
          {/* Logo & Title - Doubled Logo Size (w-18 h-18 / w-20 h-20) and properly aligned */}
          <button
            onClick={scrollToTop}
            className="flex items-center gap-4 group text-left cursor-pointer transition"
          >
            <HoverLogoVideo className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl" />
            <div className="flex flex-col justify-center">
              <span className="font-heading text-2xl sm:text-3xl tracking-[0.05em] bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent block leading-tight">
                Atlas AI
              </span>
              <span className="font-subtext text-xs text-cyan-400 block font-medium mt-0.5">
                Disaster Intelligence
              </span>
            </div>
          </button>

          {/* Smooth Scroll Navigation Links */}
          <div className="hidden md:flex items-center gap-8 text-sm text-slate-400">
            <button
              onClick={() => smoothScrollTo('benchmarks')}
              className="font-subtext hover:text-cyan-400 transition cursor-pointer"
            >
              Benchmarks
            </button>
            <button
              onClick={() => smoothScrollTo('preview')}
              className="font-subtext hover:text-cyan-400 transition cursor-pointer"
            >
              Visual Triage
            </button>
            <button
              onClick={() => smoothScrollTo('features')}
              className="font-subtext hover:text-cyan-400 transition cursor-pointer"
            >
              Architecture
            </button>
          </div>

          {/* Launch Console Button: Gradient Outline with Solid Black Interior */}
          <button
            onClick={onLaunchConsole}
            className="relative p-[1.5px] rounded-xl bg-gradient-to-r from-purple-500 via-indigo-500 to-cyan-400 group transition duration-300 hover:shadow-lg hover:shadow-cyan-500/20 active:scale-[0.98] cursor-pointer"
          >
            <div className="px-5 py-2.5 rounded-[10.5px] bg-[#07090E] group-hover:bg-[#0c101d] transition flex items-center gap-2 text-white font-medium text-sm">
              <Sparkles className="w-4 h-4 text-cyan-300" />
              <span className="font-subtext font-medium">Launch Console</span>
              <ArrowRight className="w-4 h-4 text-cyan-300 group-hover:translate-x-0.5 transition" />
            </div>
          </button>
        </div>
      </header>

      {/* ── 1. Hero Page Alone (Full Screen Viewport with Twinkling Stars) ── */}
      <section className="relative min-h-[calc(100vh-112px)] flex flex-col justify-center items-center px-6 py-16 text-center overflow-hidden">
        {/* Twinkling Starfield Pattern: Confined to Hero Alone */}
        <TwinklingStars count={52} />

        <div className="max-w-5xl mx-auto relative z-10 flex flex-col items-center">
          {/* Main Heading: Strictly Max 2-3 Lines with Balanced Layout */}
          <motion.h1 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="font-heading text-4xl sm:text-5xl md:text-6xl lg:text-7xl tracking-[0.05em] leading-[1.2] mb-6 text-center text-balance"
          >
            Next-Gen Satellite Disaster VQA <br className="hidden sm:inline" />
            <span className="text-white">& </span>
            <span className="animate-shifting-gradient inline-block">
              Spatial Intelligence
            </span>
          </motion.h1>

          {/* Subtext: Google Sans Style, Clean Sentence/Lower Case */}
          <motion.p 
            initial={{ opacity: 0, y: 25 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.18, ease: "easeOut" }}
            className="font-subtext text-slate-300 text-base sm:text-xl max-w-2xl mx-auto leading-relaxed font-normal normal-case"
          >
            Two-stage deep computer vision (ResNet-34 UNet + Siamese ResNet-18) paired with a deterministic spatial algebra engine and Groq LPU grounded conversational agent.
          </motion.p>
        </div>
      </section>

      {/* ── 2. Benchmarks Section (Safely Below the Fold) ── */}
      <section id="benchmarks" className="max-w-6xl mx-auto px-6 pt-24 pb-20 relative z-10">
        <motion.div 
          initial={{ opacity: 0, y: 35 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.25 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto"
        >
          <div className="glass-card p-5 rounded-2xl text-left border border-white/[0.08] hover:border-cyan-500/30 transition">
            <div className="font-heading text-3xl font-bold text-white mb-1 tracking-[0.05em]">100%</div>
            <div className="font-subtext text-xs text-slate-300 font-medium">CV Grounding</div>
            <div className="text-[11px] text-emerald-400 mt-1 font-mono">Zero Hallucinations</div>
          </div>
          <div className="glass-card p-5 rounded-2xl text-left border border-white/[0.08] hover:border-purple-500/30 transition">
            <div className="font-heading text-3xl font-bold text-white mb-1 tracking-[0.05em]">4-Tier</div>
            <div className="font-subtext text-xs text-slate-300 font-medium">xBD Damage Scale</div>
            <div className="text-[11px] text-cyan-400 mt-1 font-mono">Siamese ResNet-18</div>
          </div>
          <div className="glass-card p-5 rounded-2xl text-left border border-white/[0.08] hover:border-purple-500/30 transition">
            <div className="font-heading text-3xl font-bold text-white mb-1 tracking-[0.05em]">&lt;50ms</div>
            <div className="font-subtext text-xs text-slate-300 font-medium">Spatial Engine</div>
            <div className="text-[11px] text-purple-400 mt-1 font-mono">Instant Scene Caching</div>
          </div>
          <div className="glass-card p-5 rounded-2xl text-left border border-white/[0.08] hover:border-emerald-500/30 transition">
            <div className="font-heading text-3xl font-bold text-white mb-1 tracking-[0.05em]">100%</div>
            <div className="font-subtext text-xs text-slate-300 font-medium">Schema Valid</div>
            <div className="text-[11px] text-emerald-400 mt-1 font-mono">{`{id, answer, region}`}</div>
          </div>
        </motion.div>
      </section>

      {/* ── 3. Visual Triage Preview Section ── */}
      <section id="preview" className="max-w-6xl mx-auto px-6 py-20 relative z-10">
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.25 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-center mb-12"
        >
          <h2 className="font-heading text-2xl sm:text-4xl font-bold tracking-[0.05em] mb-3">
            3-Column Satellite Imagery & AI Damage Triage
          </h2>
          <p className="font-subtext text-slate-400 text-sm sm:text-base max-w-2xl mx-auto">
            Detect building footprints from pre-disaster passes and perform differential Siamese analysis on post-disaster satellite imagery.
          </p>
        </motion.div>

        {/* Scene Selector Pills */}
        {sampleScenes.length > 0 && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.6 }}
            className="flex flex-wrap items-center justify-center gap-2 mb-8"
          >
            {sampleScenes.slice(0, 4).map((s, idx) => (
              <button
                key={s.pair_id}
                onClick={() => setActivePreviewIndex(idx)}
                className={`px-4 py-2 rounded-xl text-xs font-mono transition cursor-pointer ${
                  activePreviewIndex === idx
                    ? 'bg-purple-600/30 border border-purple-500/60 text-white shadow-lg shadow-purple-500/20'
                    : 'glass-card text-slate-400 hover:text-white border-white/[0.06]'
                }`}
              >
                {s.pair_id} &bull; {s.disaster} ({s.total_buildings} bldgs)
              </button>
            ))}
          </motion.div>
        )}

        {/* 3-Column Imagery Showcase Card */}
        <motion.div 
          initial={{ opacity: 0, y: 45 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.8, delay: 0.15, ease: "easeOut" }}
          className="glass-panel rounded-3xl p-6 sm:p-8 border border-white/[0.1] shadow-2xl relative overflow-hidden"
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* 1. Pre-Disaster */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-subtext text-slate-400">
                <span className="font-semibold text-slate-200">1. Pre-Disaster Satellite</span>
                <span className="text-cyan-400 font-mono">Before</span>
              </div>
              <div className="aspect-square rounded-2xl overflow-hidden bg-slate-900 border border-white/[0.08] relative group">
                <img
                  src={previewItem.pre_url}
                  alt="Pre-disaster pass"
                  className="w-full h-full object-cover transition duration-500 group-hover:scale-105"
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
                <div className="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg bg-black/70 backdrop-blur-md text-[11px] font-subtext text-slate-300">
                  Visible Optical Pass
                </div>
              </div>
            </div>

            {/* 2. Post-Disaster */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-subtext text-slate-400">
                <span className="font-semibold text-slate-200">2. Post-Disaster Satellite</span>
                <span className="text-amber-400 font-mono">After</span>
              </div>
              <div className="aspect-square rounded-2xl overflow-hidden bg-slate-900 border border-white/[0.08] relative group">
                <img
                  src={previewItem.post_url}
                  alt="Post-disaster pass"
                  className="w-full h-full object-cover transition duration-500 group-hover:scale-105"
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
                <div className="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg bg-black/70 backdrop-blur-md text-[11px] font-subtext text-slate-300">
                  Disaster Impact Pass
                </div>
              </div>
            </div>

            {/* 3. AI Damage Classification */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-subtext text-slate-400">
                <span className="font-semibold text-slate-200">3. AI Damage Triage Map</span>
                <span className="text-purple-400 font-bold font-mono">{previewItem.total_buildings} Structures</span>
              </div>
              <div className="aspect-square rounded-2xl overflow-hidden bg-slate-900 border border-white/[0.08] relative group">
                <img
                  src={previewItem.overlay_url}
                  alt="AI Damage Overlay"
                  className="w-full h-full object-cover transition duration-500 group-hover:scale-105"
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
                <div className="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg bg-black/70 backdrop-blur-md text-[11px] font-subtext text-emerald-400">
                  Color-Coded 4 Tiers
                </div>
              </div>
            </div>
          </div>

          {/* Damage Tiers Legend */}
          <div className="mt-8 pt-6 border-t border-white/[0.08] flex flex-wrap items-center justify-between gap-4 text-xs font-subtext">
            <div className="flex flex-wrap items-center gap-6">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-emerald-500" />
                <span className="text-slate-300">Undamaged ({previewItem.damage_breakdown['no-damage'] || 0})</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-yellow-400" />
                <span className="text-slate-300">Minor Damage ({previewItem.damage_breakdown['minor-damage'] || 0})</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-amber-500" />
                <span className="text-slate-300">Major Damage ({previewItem.damage_breakdown['major-damage'] || 0})</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-rose-500" />
                <span className="text-slate-300">Destroyed ({previewItem.damage_breakdown['destroyed'] || 0})</span>
              </div>
            </div>

            <button
              onClick={onLaunchConsole}
              className="text-xs font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1 cursor-pointer font-subtext"
            >
              <span>Analyze in Mission Console</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </motion.div>
      </section>

      {/* ── 4. Architecture Highlights Section ── */}
      <section id="features" className="max-w-6xl mx-auto px-6 py-20 relative z-10">
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.25 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full glass-card border border-purple-500/30 text-purple-300 text-xs font-medium mb-4">
            <span className="font-subtext font-medium">The 3-Tier Engine</span>
          </div>
          <h2 className="font-heading text-3xl sm:text-5xl font-extrabold tracking-[0.05em] mb-4">
            Engineered for Precision & Verifiability
          </h2>
          <p className="font-subtext text-slate-400 text-base max-w-2xl mx-auto">
            Eliminates multimodal hallucinations by decoupling visual perception from spatial math and natural conversation.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1 */}
          <motion.div 
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            className="glass-card p-8 rounded-3xl border border-white/[0.08] hover:border-cyan-500/40 transition group"
          >
            <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mb-6 group-hover:scale-110 transition">
              <Layers className="w-6 h-6 text-cyan-400" />
            </div>
            <h3 className="font-heading text-xl font-bold text-white mb-2 tracking-[0.05em]">Stage 1: Building Localization</h3>
            <p className="font-subtext text-sm text-slate-300 leading-relaxed mb-4">
              Unet-ResNet34 paired with Distance-Transform Watershed instance separation detects accurate rooftop polygons without polygon merging.
            </p>
            <div className="text-xs font-mono text-cyan-300 flex items-center gap-1">
              <span>MAE: 7.72 structures</span>
            </div>
          </motion.div>

          {/* Card 2 */}
          <motion.div 
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
            className="glass-card p-8 rounded-3xl border border-white/[0.08] hover:border-purple-500/40 transition group"
          >
            <div className="w-12 h-12 rounded-2xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center mb-6 group-hover:scale-110 transition">
              <Activity className="w-6 h-6 text-purple-400" />
            </div>
            <h3 className="font-heading text-xl font-bold text-white mb-2 tracking-[0.05em]">Stage 2: Siamese Damage Triage</h3>
            <p className="font-subtext text-sm text-slate-300 leading-relaxed mb-4">
              Shared-weight Siamese ResNet-18 extracts multi-scale context crops [f_pre, f_post, f_diff] across 4 xBD damage severity levels.
            </p>
            <div className="text-xs font-mono text-purple-300 flex items-center gap-1">
              <span>Softened Class Weighting</span>
            </div>
          </motion.div>

          {/* Card 3 */}
          <motion.div 
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.6, delay: 0.3, ease: "easeOut" }}
            className="glass-card p-8 rounded-3xl border border-white/[0.08] hover:border-emerald-500/40 transition group"
          >
            <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center mb-6 group-hover:scale-110 transition">
              <Compass className="w-6 h-6 text-emerald-400" />
            </div>
            <h3 className="font-heading text-xl font-bold text-white mb-2 tracking-[0.05em]">Stage 3: Spatial Algebra & LLM</h3>
            <p className="font-subtext text-sm text-slate-300 leading-relaxed mb-4">
              Zero-hallucination deterministic spatial arithmetic executes exact quadrant rankings, cardinal comparisons, and ratio math via Groq LPUs.
            </p>
            <div className="text-xs font-mono text-emerald-300 flex items-center gap-1">
              <span>100% Zero-Hallucination</span>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
