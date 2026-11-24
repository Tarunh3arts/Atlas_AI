import React, { useState, useEffect, useRef } from 'react';
import { 
  Satellite, 
  Upload, 
  Send, 
  Plus, 
  Compass, 
  BarChart3, 
  Scale, 
  ShieldAlert, 
  Copy, 
  Check, 
  Eye, 
  Pin, 
  PinOff,
  Edit2, 
  Trash2, 
  Building2, 
  MessageSquare, 
  User, 
  X,
  Sparkles,
  ArrowRight,
  ChevronDown,
  CheckCircle2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { HoverLogoVideo, LoopLogoLoader, ThinkingLogoVideo } from './LogoVideo.jsx';

const STORAGE_KEY = 'atlas_ai_saved_chats_v1';

export default function ChatConsole({ onBackHome, scenes = [] }) {
  // Saved chats in localStorage
  const [chats, setChats] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) return JSON.parse(saved);
    } catch (e) {}
    return [];
  });

  // Active chat session
  const [activeChatId, setActiveChatId] = useState(null);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editingTitleText, setEditingTitleText] = useState('');

  // Active scene state
  const [selectedPairId, setSelectedPairId] = useState(null);
  const [currentScene, setCurrentScene] = useState(null);
  const [loadingScene, setLoadingScene] = useState(false);

  // Chat message stream
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showQuickPrompts, setShowQuickPrompts] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  // Collapsed JSON verification state per message
  const [expandedJsonIds, setExpandedJsonIds] = useState({});

  // File upload state (staged for Enter key execution)
  const [preFile, setPreFile] = useState(null);
  const [postFile, setPostFile] = useState(null);
  const preInputRef = useRef(null);
  const postInputRef = useRef(null);

  // Modal zoom
  const [modalImage, setModalImage] = useState(null);

  const messagesEndRef = useRef(null);

  // Persist chats to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    } catch (e) {}
  }, [chats]);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSubmitting]);

  // Explicitly select and switch to a chat
  const handleSelectChat = (chatId) => {
    setActiveChatId(chatId);
    const targetChat = chats.find((c) => c.id === chatId);
    if (targetChat) {
      setMessages(targetChat.messages || []);
      setSelectedPairId(targetChat.pairId || null);
    }
  };

  // Load scene data when pairId changes
  useEffect(() => {
    if (!selectedPairId) return;
    setLoadingScene(true);
    fetch(`/api/scenes/${selectedPairId}`)
      .then((res) => res.json())
      .then((data) => {
        setCurrentScene(data);
        setLoadingScene(false);
      })
      .catch((err) => {
        console.error('Failed to load scene:', err);
        setLoadingScene(false);
      });
  }, [selectedPairId]);

  // Create a brand new fresh chat
  const handleStartNewChat = () => {
    setActiveChatId(null);
    setSelectedPairId(null);
    setCurrentScene(null);
    setMessages([]);
    setPreFile(null);
    setPostFile(null);
    setExpandedJsonIds({});
    if (preInputRef.current) preInputRef.current.value = '';
    if (postInputRef.current) postInputRef.current.value = '';
  };

  // Pin / Unpin a chat
  const handleTogglePin = (chatId, e) => {
    e.stopPropagation();
    setChats((prev) =>
      prev.map((c) => (c.id === chatId ? { ...c, isPinned: !c.isPinned } : c))
    );
  };

  // Start renaming
  const handleStartRename = (chat, e) => {
    e.stopPropagation();
    setEditingChatId(chat.id);
    setEditingTitleText(chat.title);
  };

  // Save renamed title
  const handleSaveRename = (chatId) => {
    if (!editingTitleText.trim()) {
      setEditingChatId(null);
      return;
    }
    setChats((prev) =>
      prev.map((c) => (c.id === chatId ? { ...c, title: editingTitleText.trim() } : c))
    );
    setEditingChatId(null);
  };

  // Delete chat
  const handleDeleteChat = (chatId, e) => {
    e.stopPropagation();
    setChats((prev) => prev.filter((c) => c.id !== chatId));
    if (activeChatId === chatId) {
      handleStartNewChat();
    }
  };

  // Toggle JSON expansion
  const toggleJson = (msgId) => {
    setExpandedJsonIds((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  // Quick select a sample disaster scene
  const handleSelectSample = (pairId) => {
    const existing = chats.find((c) => c.pairId === pairId);
    if (existing) {
      handleSelectChat(existing.id);
      return;
    }
    const sceneInfo = scenes.find((s) => s.pair_id === pairId);
    const newChatId = `chat_${Date.now()}`;
    const newChat = {
      id: newChatId,
      title: sceneInfo ? `${sceneInfo.disaster} (${pairId})` : `Incident ${pairId}`,
      createdAt: Date.now(),
      pairId: pairId,
      isPinned: false,
      messages: [],
    };
    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChatId);
    setSelectedPairId(pairId);
    setMessages([]);
  };

  // Unified Send Message & Pipeline Trigger on Enter
  const handleSendMessage = async (customQuery = null) => {
    const query = customQuery || inputQuery.trim();
    if (!query || isSubmitting) return;

    // Check if user uploaded one file but not both
    if ((preFile && !postFile) || (!preFile && postFile)) {
      alert('Please upload both Pre-Disaster and Post-Disaster satellite images before submitting.');
      return;
    }

    setShowQuickPrompts(false);
    setInputQuery('');

    const userMsg = {
      id: `usr_${Date.now()}`,
      role: 'user',
      content: query,
    };

    let chatId = activeChatId;
    let nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setIsSubmitting(true);

    try {
      let currentPairId = selectedPairId;

      // ── Step A: If custom pre & post images are staged, process them automatically ──
      if (preFile && postFile) {
        const pid = `custom_${Date.now().toString().slice(-4)}`;
        const formData = new FormData();
        formData.append('pre_image', preFile);
        formData.append('post_image', postFile);
        formData.append('pair_id', pid);

        const uploadResp = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!uploadResp.ok) throw new Error('Neural upload & inference pipeline failed.');
        const uploadData = await uploadResp.json();

        currentPairId = pid;
        setSelectedPairId(pid);
        setCurrentScene({
          pair_id: pid,
          disaster: `Incident (${pid})`,
          pre_url: uploadData.pre_url,
          post_url: uploadData.post_url,
          overlay_url: uploadData.overlay_url,
          summary: uploadData.summary,
        });
        setPreFile(null);
        setPostFile(null);
      }

      if (!currentPairId && scenes.length > 0) {
        currentPairId = scenes[0].pair_id;
        setSelectedPairId(currentPairId);
      }

      if (!chatId) {
        chatId = `chat_${Date.now()}`;
        setActiveChatId(chatId);
        const newChat = {
          id: chatId,
          title: query.slice(0, 30) + (query.length > 30 ? '...' : ''),
          createdAt: Date.now(),
          pairId: currentPairId,
          isPinned: false,
          messages: nextMessages,
        };
        setChats((prev) => [newChat, ...prev.filter((c) => c.id !== chatId)]);
      }

      // ── Step B: Execute grounded spatial question answering ──
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pair_id: currentPairId || 'p11b4ad02',
          question: query,
        }),
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }

      const data = await resp.json();
      const replyText = data.natural_response || data.reply || (typeof data === 'string' ? data : 'Analysis completed.');

      const asstMsg = {
        id: `asst_${Date.now()}`,
        role: 'assistant',
        content: replyText,
        backend: data.backend || 'Atlas Spatial Engine',
        submission_json: data.submission_json,
        evidence_url: data.evidence_url,
        scene_summary: data.scene_summary,
        exec_time: data.execution_time_seconds,
      };

      const finalMessages = [...nextMessages, asstMsg];
      setMessages(finalMessages);

      setChats((prev) =>
        prev.map((c) => (c.id === chatId ? { ...c, messages: finalMessages, pairId: currentPairId } : c))
      );
    } catch (err) {
      const errMsg = {
        id: `asst_err_${Date.now()}`,
        role: 'assistant',
        content: `⚠️ Error during triage: ${err.message || 'Check server status'}`,
        isError: true,
      };
      const errorMessages = [...nextMessages, errMsg];
      setMessages(errorMessages);
      setChats((prev) =>
        prev.map((c) => (c.id === chatId ? { ...c, messages: errorMessages } : c))
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Sort chats: pinned first, then newest
  const sortedChats = [...chats].sort((a, b) => {
    if (a.isPinned === b.isPinned) return b.createdAt - a.createdAt;
    return a.isPinned ? -1 : 1;
  });

  const isNewChat = messages.length === 0;
  const summary = currentScene?.summary || {};
  const tiers = summary.by_damage_tier || {};

  return (
    <div className="flex h-screen bg-[#07090E] text-slate-100 overflow-hidden font-sans">
      {/* ── Left Sidebar (Past Chats & History) ────────────────────── */}
      <aside className="w-80 bg-[#090D18] border-r border-white/[0.06] flex flex-col justify-between z-20">
        <div className="flex-1 flex flex-col min-h-0">
          {/* Brand Header */}
          <div className="p-5 border-b border-white/[0.06] flex items-center justify-between">
            <button
              onClick={onBackHome}
              className="flex items-center gap-3.5 text-left group cursor-pointer transition"
              title="Return to Landing Page"
            >
              <HoverLogoVideo className="w-10 h-10" />
              <div>
                <span className="font-heading text-xl tracking-[0.05em] text-white block group-hover:text-cyan-300 transition">
                  Atlas AI
                </span>
                <span className="font-subtext text-[11px] text-cyan-400 font-medium block">
                  Disaster Intelligence
                </span>
              </div>
            </button>
          </div>

          {/* New Chat Button */}
          <div className="p-4 border-b border-white/[0.04]">
            <button
              onClick={handleStartNewChat}
              className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-purple-600/30 to-cyan-500/30 hover:from-purple-600/50 hover:to-cyan-500/50 border border-purple-500/40 text-slate-100 text-xs font-semibold flex items-center justify-center gap-2 transition cursor-pointer shadow-lg shadow-purple-900/10"
            >
              <Plus className="w-4 h-4 text-cyan-300" />
              <span className="font-subtext font-medium">New Chat</span>
            </button>
          </div>

          {/* Past Chats List */}
          <div className="flex-1 overflow-y-auto p-3 space-y-1">
            <div className="font-subtext text-[11px] text-slate-400 mb-2 px-2 flex items-center justify-between font-medium">
              <span>Past Chats</span>
              <span>{chats.length}</span>
            </div>

            {sortedChats.length === 0 ? (
              <div className="text-center py-10 px-4 text-slate-500 text-xs">
                <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-30 text-purple-400" />
                <p className="font-subtext text-slate-400">No chat history yet.</p>
                <p className="font-subtext text-[11px] text-slate-500 mt-1">Select an incident or upload satellite passes.</p>
              </div>
            ) : (
              sortedChats.map((c) => {
                const isActive = c.id === activeChatId;
                const isEditing = c.id === editingChatId;

                return (
                  <div
                    key={c.id}
                    onClick={() => {
                      if (!isEditing) handleSelectChat(c.id);
                    }}
                    className={`group relative w-full text-left p-2.5 rounded-xl transition cursor-pointer flex items-center justify-between gap-2 ${
                      isActive
                        ? 'bg-purple-600/20 border border-purple-500/50 text-white shadow-md'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.03] border border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 min-w-0 flex-1">
                      <MessageSquare className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                      
                      {isEditing ? (
                        <input
                          type="text"
                          value={editingTitleText}
                          onChange={(e) => setEditingTitleText(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveRename(c.id);
                            if (e.key === 'Escape') setEditingChatId(null);
                          }}
                          autoFocus
                          className="bg-black/60 border border-purple-400/60 rounded px-1.5 py-0.5 text-xs text-white focus:outline-none w-full font-subtext"
                          onClick={(e) => e.stopPropagation()}
                        />
                      ) : (
                        <div className="truncate flex-1">
                          <span className="text-xs truncate block font-subtext font-medium text-slate-200">
                            {c.title}
                          </span>
                          <span className="text-[10px] font-subtext text-slate-500 block">
                            {new Date(c.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Chat Item Actions (Pin, Rename, Delete) */}
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition flex-shrink-0">
                      {isEditing ? (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSaveRename(c.id);
                          }}
                          className="p-1 hover:text-emerald-400 rounded transition"
                        >
                          <Check className="w-3.5 h-3.5" />
                        </button>
                      ) : (
                        <>
                          <button
                            onClick={(e) => handleTogglePin(c.id, e)}
                            className={`p-1 hover:text-cyan-300 rounded transition ${c.isPinned ? 'text-cyan-400 opacity-100' : 'text-slate-500'}`}
                            title={c.isPinned ? 'Unpin Chat' : 'Pin Chat'}
                          >
                            <Pin className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={(e) => handleStartRename(c, e)}
                            className="p-1 text-slate-500 hover:text-purple-300 rounded transition"
                            title="Rename Chat"
                          >
                            <Edit2 className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={(e) => handleDeleteChat(c.id, e)}
                            className="p-1 text-slate-500 hover:text-rose-400 rounded transition"
                            title="Delete Chat"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </>
                      )}
                    </div>

                    {/* Always visible pin badge if pinned */}
                    {c.isPinned && !isEditing && (
                      <Pin className="w-3 h-3 text-cyan-400 flex-shrink-0 group-hover:hidden" />
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* User Account / Session Footer */}
        <div className="p-4 border-t border-white/[0.06] bg-[#070912] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-600 to-cyan-500 flex items-center justify-center text-white text-xs font-bold font-subtext">
              OP
            </div>
            <div>
              <span className="text-xs font-subtext font-semibold text-slate-200 block">Operator Command</span>
              <span className="font-subtext text-[10px] text-slate-400 block">Autonomous Session</span>
            </div>
          </div>
          <button
            onClick={onBackHome}
            className="text-xs font-subtext text-slate-400 hover:text-white transition cursor-pointer font-medium"
            title="Landing Page"
          >
            Exit
          </button>
        </div>
      </aside>

      {/* ── Main Workspace Area ────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#07090E] relative overflow-hidden">
        {/* Hidden File Inputs */}
        <input
          ref={preInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => setPreFile(e.target.files[0])}
        />
        <input
          ref={postInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => setPostFile(e.target.files[0])}
        />

        {/* Top Header */}
        {!isNewChat && currentScene && (
          <header className="h-16 border-b border-white/[0.06] bg-[#080B15]/90 backdrop-blur-md px-6 flex items-center justify-between z-10 flex-shrink-0">
            {/* Top-Left Upload Controls (Staging) */}
            <div className="flex items-center gap-3">
              <div className="relative flex items-center">
                <button
                  type="button"
                  onClick={() => preInputRef.current?.click()}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition cursor-pointer font-subtext ${
                    preFile
                      ? 'bg-cyan-500/20 border-cyan-500 text-cyan-200'
                      : 'glass-card border-white/[0.08] text-slate-300 hover:text-white hover:bg-white/[0.06]'
                  }`}
                >
                  <Upload className="w-3 h-3 text-cyan-400" />
                  <span>{preFile ? `Pre: ${preFile.name.slice(0, 12)}...` : 'Upload Pre (Before)'}</span>
                </button>
                {preFile && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setPreFile(null);
                      if (preInputRef.current) preInputRef.current.value = '';
                    }}
                    className="ml-1 text-slate-400 hover:text-rose-400 transition cursor-pointer p-0.5"
                    title="Clear Pre pass"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>

              <div className="relative flex items-center">
                <button
                  type="button"
                  onClick={() => postInputRef.current?.click()}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition cursor-pointer font-subtext ${
                    postFile
                      ? 'bg-amber-500/20 border-amber-500 text-amber-200'
                      : 'glass-card border-white/[0.08] text-slate-300 hover:text-white hover:bg-white/[0.06]'
                  }`}
                >
                  <Upload className="w-3 h-3 text-amber-400" />
                  <span>{postFile ? `Post: ${postFile.name.slice(0, 12)}...` : 'Upload Post (After)'}</span>
                </button>
                {postFile && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setPostFile(null);
                      if (postInputRef.current) postInputRef.current.value = '';
                    }}
                    className="ml-1 text-slate-400 hover:text-rose-400 transition cursor-pointer p-0.5"
                    title="Clear Post pass"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>

              {preFile && postFile && (
                <span className="px-3 py-1 rounded-lg bg-purple-500/20 border border-purple-500/40 text-purple-200 text-xs font-subtext flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Ready &bull; Type prompt and hit Enter</span>
                </span>
              )}
            </div>

            {/* Scene Telemetry Pills */}
            <div className="flex items-center gap-3 text-[11px] font-subtext">
              <span className="text-slate-300 font-semibold">{currentScene?.disaster || selectedPairId}</span>
              <span className="text-slate-600">&bull;</span>
              <span className="text-emerald-400">🟢 Undamaged: {tiers['no-damage'] || 0}</span>
              <span className="text-yellow-400">🟡 Minor: {tiers['minor-damage'] || 0}</span>
              <span className="text-amber-400">🟠 Major: {tiers['major-damage'] || 0}</span>
              <span className="text-rose-400">🔴 Destroyed: {tiers['destroyed'] || 0}</span>
            </div>
          </header>
        )}

        {/* Workspace Body */}
        <div className="flex-1 overflow-y-auto relative flex flex-col">
          {/* ── Case A: Brand New Chat (Clean Greeting, Delayed Glow & Staging Buttons) ── */}
          {isNewChat ? (
            <div className="flex-1 flex flex-col items-center justify-center p-8 max-w-4xl mx-auto text-center relative z-10">
              {/* Delayed Background Glow */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1.5, duration: 1.8, ease: 'easeOut' }}
                className="absolute -top-10 left-1/2 -translate-x-1/2 w-[650px] h-[380px] bg-gradient-to-r from-purple-600/25 via-cyan-500/20 to-purple-600/25 blur-[110px] pointer-events-none rounded-full"
              />

              {/* Motivational Contextual Greeting */}
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7 }}
                className="mb-8 relative z-10"
              >
                <h1 className="font-heading text-4xl sm:text-6xl font-extrabold text-white tracking-[0.05em]">
                  Welcome, Commander.
                </h1>
              </motion.div>

              {/* Two Large Transparent-like Buttons with Huge Plus Icons */}
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.2 }}
                className="grid grid-cols-1 sm:grid-cols-2 gap-6 w-full max-w-2xl mb-8 relative z-10"
              >
                {/* Button 1: Pre-Disaster Upload */}
                <div className="relative group">
                  <button
                    type="button"
                    onClick={() => preInputRef.current?.click()}
                    className={`w-full rounded-3xl border-2 border-dashed p-8 flex flex-col items-center justify-center gap-4 transition duration-300 cursor-pointer text-center ${
                      preFile
                        ? 'border-cyan-500/70 bg-cyan-950/20'
                        : 'border-white/10 hover:border-cyan-500/50 bg-white/[0.015] hover:bg-white/[0.03]'
                    }`}
                  >
                    <div className="w-20 h-20 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center group-hover:scale-110 group-hover:border-cyan-500/40 transition duration-300">
                      {preFile ? <Check className="w-10 h-10 text-cyan-400" /> : <Plus className="w-10 h-10 text-cyan-400" />}
                    </div>
                    <div>
                      <h3 className="font-heading text-lg font-bold text-white mb-1 tracking-[0.05em] truncate max-w-[240px]">
                        {preFile ? `Pre: ${preFile.name}` : 'Upload Pre-Disaster'}
                      </h3>
                      <p className="font-subtext text-xs text-slate-400">
                        {preFile ? `${(preFile.size / 1024 / 1024).toFixed(2)} MB • Ready` : 'Baseline Optical Satellite Capture'}
                      </p>
                    </div>
                  </button>
                  {preFile && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setPreFile(null);
                        if (preInputRef.current) preInputRef.current.value = '';
                      }}
                      className="absolute top-4 right-4 w-7 h-7 rounded-full bg-slate-800/90 border border-white/20 hover:border-rose-500 hover:bg-rose-500/20 text-slate-400 hover:text-rose-300 flex items-center justify-center transition cursor-pointer z-20"
                      title="Remove staged Pre-Disaster pass"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>

                {/* Button 2: Post-Disaster Upload */}
                <div className="relative group">
                  <button
                    type="button"
                    onClick={() => postInputRef.current?.click()}
                    className={`w-full rounded-3xl border-2 border-dashed p-8 flex flex-col items-center justify-center gap-4 transition duration-300 cursor-pointer text-center ${
                      postFile
                        ? 'border-purple-500/70 bg-purple-950/20'
                        : 'border-white/10 hover:border-purple-500/50 bg-white/[0.015] hover:bg-white/[0.03]'
                    }`}
                  >
                    <div className="w-20 h-20 rounded-2xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center group-hover:scale-110 group-hover:border-purple-500/40 transition duration-300">
                      {postFile ? <Check className="w-10 h-10 text-purple-400" /> : <Plus className="w-10 h-10 text-purple-400" />}
                    </div>
                    <div>
                      <h3 className="font-heading text-lg font-bold text-white mb-1 tracking-[0.05em] truncate max-w-[240px]">
                        {postFile ? `Post: ${postFile.name}` : 'Upload Post-Disaster'}
                      </h3>
                      <p className="font-subtext text-xs text-slate-400">
                        {postFile ? `${(postFile.size / 1024 / 1024).toFixed(2)} MB • Ready` : 'Post-Event Impact Satellite Pass'}
                      </p>
                    </div>
                  </button>
                  {postFile && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setPostFile(null);
                        if (postInputRef.current) postInputRef.current.value = '';
                      }}
                      className="absolute top-4 right-4 w-7 h-7 rounded-full bg-slate-800/90 border border-white/20 hover:border-rose-500 hover:bg-rose-500/20 text-slate-400 hover:text-rose-300 flex items-center justify-center transition cursor-pointer z-20"
                      title="Remove staged Post-Disaster pass"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </motion.div>

              {/* Instant Test Practice Chips */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="relative z-10 text-center"
              >
                <div className="font-subtext text-xs text-slate-400 mb-3">
                  Or test instantly with a pre-cached disaster incident:
                </div>
                <div className="flex flex-wrap items-center justify-center gap-2">
                  {scenes.slice(0, 3).map((s) => (
                    <button
                      key={s.pair_id}
                      onClick={() => handleSelectSample(s.pair_id)}
                      className="px-3.5 py-1.5 rounded-xl glass-card hover:bg-white/[0.08] border border-white/[0.08] text-xs font-mono text-slate-300 hover:text-white transition cursor-pointer"
                    >
                      {s.disaster} ({s.total_buildings} bldgs)
                    </button>
                  ))}
                </div>
              </motion.div>
            </div>
          ) : (
            /* ── Case B: Ongoing Chat Stream with Evidence & Imagery ── */
            <div className="flex-1 p-6 space-y-6 max-w-4xl mx-auto w-full">
              {/* Scene loading spinner */}
              {loadingScene && (
                <div className="p-8 rounded-3xl glass-panel border border-white/[0.08] flex items-center justify-center my-4">
                  <LoopLogoLoader 
                    size="xl" 
                    label="Loading Optical Feeds & Instance Polygons..." 
                    sublabel="Retrieving 1024x1024 high-resolution imagery" 
                  />
                </div>
              )}

              {/* Messages Stream */}
              {messages.map((msg, index) => {
                const isFirstAssistant = messages.findIndex((m) => m.role === 'assistant') === index;

                return (
                  <React.Fragment key={msg.id}>
                    {/* ── 1. The 3-Column Imagery Strip: Appears after user chat and before AI answer ── */}
                    {msg.role === 'assistant' && isFirstAssistant && currentScene && (
                      <motion.div
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4 }}
                        className="grid grid-cols-3 gap-3 p-3 rounded-2xl bg-black/40 border border-white/[0.06] my-4"
                      >
                        {/* Pre */}
                        <div className="space-y-1 text-center">
                          <span className="font-subtext text-[11px] text-slate-400 block font-medium">1. Pre-Disaster</span>
                          <div 
                            onClick={() => setModalImage({ url: currentScene.pre_url, title: 'Pre-Disaster Optical Pass' })}
                            className="aspect-square rounded-xl overflow-hidden bg-slate-900 border border-white/[0.08] cursor-zoom-in group relative"
                          >
                            <img src={currentScene.pre_url} alt="Pre" className="w-full h-full object-cover group-hover:scale-105 transition duration-300" />
                            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center text-[11px] font-mono text-white">
                              <Eye className="w-4 h-4 mr-1" /> Expand
                            </div>
                          </div>
                        </div>

                        {/* Post */}
                        <div className="space-y-1 text-center">
                          <span className="font-subtext text-[11px] text-amber-400 block font-medium">2. Post-Disaster</span>
                          <div 
                            onClick={() => setModalImage({ url: currentScene.post_url, title: 'Post-Disaster Impact Pass' })}
                            className="aspect-square rounded-xl overflow-hidden bg-slate-900 border border-white/[0.08] cursor-zoom-in group relative"
                          >
                            <img src={currentScene.post_url} alt="Post" className="w-full h-full object-cover group-hover:scale-105 transition duration-300" />
                            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center text-[11px] font-mono text-white">
                              <Eye className="w-4 h-4 mr-1" /> Expand
                            </div>
                          </div>
                        </div>

                        {/* AI Overlay */}
                        <div className="space-y-1 text-center">
                          <span className="font-subtext text-[11px] text-purple-400 block font-medium">3. AI Triage Map</span>
                          <div 
                            onClick={() => setModalImage({ url: currentScene.overlay_url, title: 'AI Damage Triage Overlay' })}
                            className="aspect-square rounded-xl overflow-hidden bg-slate-900 border border-white/[0.08] cursor-zoom-in group relative"
                          >
                            <img src={currentScene.overlay_url} alt="Overlay" className="w-full h-full object-cover group-hover:scale-105 transition duration-300" />
                            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center text-[11px] font-mono text-white">
                              <Eye className="w-4 h-4 mr-1" /> Expand
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}

                    {/* ── 2. Message Bubble ── */}
                    <motion.div
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                      className={`flex gap-3.5 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      {msg.role === 'assistant' && (
                        <div className="w-8 h-8 rounded-xl overflow-hidden border border-purple-500/40 shadow-md shadow-purple-500/20 flex-shrink-0 mt-1 bg-black/60">
                          <video src="/logo.mp4" autoPlay loop muted playsInline className="w-full h-full object-cover" />
                        </div>
                      )}

                      <div className={`max-w-[85%] sm:max-w-[78%] space-y-2.5 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
                        <div
                          className={`p-4 rounded-2xl text-sm leading-relaxed ${
                            msg.role === 'user'
                              ? 'bg-gradient-to-r from-purple-600/90 to-cyan-600/90 text-white rounded-tr-sm shadow-lg font-subtext'
                              : msg.isError
                              ? 'bg-rose-950/40 border border-rose-500/40 text-rose-200 rounded-tl-sm font-subtext'
                              : 'glass-panel border border-white/[0.08] text-slate-100 rounded-tl-sm shadow-md font-subtext'
                          }`}
                        >
                          {/* Formatted Text Content */}
                          <div className="prose prose-invert text-sm max-w-none space-y-2 font-subtext">
                            {String(msg.content || '').split('\n\n').map((par, i) => (
                              <p key={i} className="m-0">
                                {par}
                              </p>
                            ))}
                          </div>

                          {/* Minimized Grounded Visual Crop Evidence */}
                          {msg.evidence_url && (
                            <div className="mt-3 pt-3 border-t border-white/[0.08]">
                              <button
                                onClick={() => setModalImage({ url: msg.evidence_url, title: 'Spatial Region Evidence Crop' })}
                                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-cyan-950/40 hover:bg-cyan-900/50 border border-cyan-500/30 text-xs font-subtext text-cyan-300 transition cursor-pointer group"
                              >
                                <Eye className="w-3.5 h-3.5 text-cyan-400 group-hover:scale-110 transition" />
                                <span>View Grounded Visual Crop Evidence</span>
                                <span className="text-[10px] font-mono text-cyan-400/60 uppercase">(Click to view)</span>
                              </button>
                            </div>
                          )}
                        </div>

                        {/* Minimized Collapsible Official Competition JSON Inspector */}
                        {msg.role === 'assistant' && msg.submission_json && msg.id !== 'init' && (
                          <div className="mt-1 text-xs">
                            <div className="flex items-center gap-2">
                              <button
                                onClick={() => toggleJson(msg.id)}
                                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-black/40 hover:bg-white/[0.06] border border-white/[0.08] text-[11px] font-subtext text-slate-400 hover:text-cyan-300 transition cursor-pointer"
                              >
                                <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${expandedJsonIds[msg.id] ? 'rotate-180 text-cyan-400' : ''}`} />
                                <span>{expandedJsonIds[msg.id] ? 'Hide Verification JSON' : 'Inspect Official Competition JSON'}</span>
                                <span className="text-[10px] font-mono text-slate-500 font-normal">({msg.backend || 'Atlas Spatial Engine'} &bull; {msg.exec_time || 0.02}s)</span>
                              </button>

                              {expandedJsonIds[msg.id] && (
                                <button
                                  onClick={() => copyToClipboard(JSON.stringify(msg.submission_json, null, 2), msg.id)}
                                  className="hover:text-cyan-400 flex items-center gap-1 text-[11px] font-subtext text-slate-400 transition cursor-pointer px-2 py-1 rounded-lg bg-black/40 border border-white/[0.08]"
                                >
                                  {copiedId === msg.id ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                                  <span>{copiedId === msg.id ? 'Copied' : 'Copy'}</span>
                                </button>
                              )}
                            </div>

                            {expandedJsonIds[msg.id] && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                className="mt-2 p-2.5 rounded-xl bg-black/60 border border-white/[0.06] font-mono text-[11px] text-slate-400 overflow-x-auto"
                              >
                                <pre className="m-0 text-slate-300">
                                  {JSON.stringify(msg.submission_json, null, 2)}
                                </pre>
                              </motion.div>
                            )}
                          </div>
                        )}
                      </div>

                      {msg.role === 'user' && (
                        <div className="w-8 h-8 rounded-xl bg-slate-800 border border-white/[0.1] flex items-center justify-center text-slate-300 flex-shrink-0 mt-1">
                          <User className="w-4 h-4" />
                        </div>
                      )}
                    </motion.div>
                  </React.Fragment>
                );
              })}

              {/* ── 3. AI Thinking State: ONLY logo video, NO text, NO borders, 2x loop ── */}
              {isSubmitting && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex justify-start my-4 pl-1"
                >
                  <ThinkingLogoVideo className="w-20 h-20 shadow-2xl shadow-purple-500/20" />
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* ── Bottom Chat Bar with Gradient Glow & Unified Upload Prompt ── */}
        <div className="p-6 bg-gradient-to-t from-[#07090E] via-[#07090E]/95 to-transparent relative z-20 flex-shrink-0">
          <div className="max-w-4xl mx-auto relative">
            {/* Staged Images Notification Banner */}
            {(preFile || postFile) && (
              <motion.div 
                initial={{ opacity: 0, y: 8 }} 
                animate={{ opacity: 1, y: 0 }} 
                className="mb-3 inline-flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-purple-900/40 via-cyan-900/30 to-purple-900/40 border border-purple-500/40 text-slate-200 text-xs font-subtext shadow-lg"
              >
                <Sparkles className="w-3.5 h-3.5 text-cyan-300 flex-shrink-0" />
                <span>
                  {preFile && postFile ? (
                    <>Pre &amp; Post satellite passes staged! Type your prompt below and press <strong>Enter</strong> to triage &amp; answer.</>
                  ) : preFile ? (
                    <>Pre-Disaster pass staged. Now select <strong>Post-Disaster</strong> pass above.</>
                  ) : (
                    <>Post-Disaster pass staged. Now select <strong>Pre-Disaster</strong> pass above.</>
                  )}
                </span>
              </motion.div>
            )}

            {/* Plus Icon Popover (Expands Upwards) */}
            <AnimatePresence>
              {showQuickPrompts && (
                <motion.div
                  initial={{ opacity: 0, y: 15, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 15, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className="absolute bottom-full left-0 mb-3 w-80 glass-panel rounded-2xl border border-purple-500/30 p-2 shadow-2xl glow-purple z-50 space-y-1"
                >
                  <div className="text-[10px] font-subtext text-purple-300 font-semibold px-3 py-1.5 uppercase tracking-wider">
                    ⚡ Spatial Query Accelerators
                  </div>

                  <button
                    onClick={() => handleSendMessage('Which quadrant contains the largest number of damaged buildings?')}
                    className="w-full text-left p-2.5 rounded-xl hover:bg-white/[0.06] text-xs text-slate-200 flex items-center gap-2.5 transition cursor-pointer"
                  >
                    <Compass className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                    <span className="font-subtext">Quadrant with Most Damage</span>
                  </button>

                  <button
                    onClick={() => handleSendMessage('Count the total undamaged buildings in the image.')}
                    className="w-full text-left p-2.5 rounded-xl hover:bg-white/[0.06] text-xs text-slate-200 flex items-center gap-2.5 transition cursor-pointer"
                  >
                    <Building2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                    <span className="font-subtext">Count Undamaged Structures</span>
                  </button>

                  <button
                    onClick={() => handleSendMessage('Is the top half more damaged than the bottom half?')}
                    className="w-full text-left p-2.5 rounded-xl hover:bg-white/[0.06] text-xs text-slate-200 flex items-center gap-2.5 transition cursor-pointer"
                  >
                    <Scale className="w-4 h-4 text-purple-400 flex-shrink-0" />
                    <span className="font-subtext">Compare North vs South Halves</span>
                  </button>

                  <button
                    onClick={() => handleSendMessage('calculate the ratio of undamaged buildings to damaged buildings')}
                    className="w-full text-left p-2.5 rounded-xl hover:bg-white/[0.06] text-xs text-slate-200 flex items-center gap-2.5 transition cursor-pointer"
                  >
                    <BarChart3 className="w-4 h-4 text-pink-400 flex-shrink-0" />
                    <span className="font-subtext">Ratio of Undamaged to Damaged</span>
                  </button>

                  <button
                    onClick={() => handleSendMessage('Which building in the image is a hospital?')}
                    className="w-full text-left p-2.5 rounded-xl hover:bg-white/[0.06] text-xs text-slate-200 flex items-center gap-2.5 transition cursor-pointer border-t border-white/[0.05]"
                  >
                    <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0" />
                    <span className="font-subtext">Hospital / Function Check (Type 4)</span>
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Glowing Input Bar */}
            <div className="relative rounded-2xl p-1 bg-gradient-to-r from-purple-500/30 via-indigo-500/20 to-cyan-500/30 glow-subtle">
              <div className="bg-[#0B0F1E] rounded-xl flex items-center px-3 py-1.5 border border-white/[0.08] focus-within:border-purple-500/60 transition">
                {/* Plus Button Expanding Upwards */}
                <button
                  type="button"
                  onClick={() => setShowQuickPrompts(!showQuickPrompts)}
                  className={`w-9 h-9 rounded-lg flex items-center justify-center transition cursor-pointer flex-shrink-0 mr-2 ${
                    showQuickPrompts
                      ? 'bg-purple-600 text-white rotate-45'
                      : 'text-slate-400 hover:text-white hover:bg-white/[0.06]'
                  }`}
                  title="Quick Spatial Prompts"
                >
                  <Plus className="w-5 h-5 transition duration-200" />
                </button>

                {/* Text Input */}
                <input
                  type="text"
                  value={inputQuery}
                  onChange={(e) => setInputQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSendMessage();
                  }}
                  placeholder={
                    preFile && postFile
                      ? "Ask your question and press Enter to triage imagery & answer..."
                      : "Ask about damage counts, quadrant hotspots, ratios, or cardinal halves..."
                  }
                  className="flex-1 bg-transparent text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none px-2 py-2 font-subtext"
                />

                {/* Send Button */}
                <button
                  type="button"
                  onClick={() => handleSendMessage()}
                  disabled={!inputQuery.trim() || isSubmitting}
                  className={`w-9 h-9 rounded-lg flex items-center justify-center transition cursor-pointer flex-shrink-0 ${
                    inputQuery.trim() && !isSubmitting
                      ? 'bg-gradient-to-r from-purple-600 to-cyan-500 text-white shadow-md shadow-purple-600/30 hover:scale-105 active:scale-95'
                      : 'text-slate-600 bg-white/[0.02] cursor-not-allowed'
                  }`}
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ── Fullscreen Zoom Modal ────────────────────────────────────── */}
      <AnimatePresence>
        {modalImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setModalImage(null)}
            className="fixed inset-0 z-50 bg-black/90 backdrop-blur-xl flex items-center justify-center p-6 cursor-zoom-out"
          >
            <div className="relative max-w-4xl max-h-[90vh] glass-panel p-3 rounded-2xl border border-white/[0.1] shadow-2xl">
              <div className="flex items-center justify-between pb-3 px-2 border-b border-white/[0.08] mb-3">
                <span className="font-heading text-sm text-slate-200 font-semibold tracking-[0.05em]">{modalImage.title}</span>
                <button
                  onClick={() => setModalImage(null)}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.1]"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <img
                src={modalImage.url}
                alt=""
                className="max-h-[75vh] w-auto mx-auto rounded-xl object-contain"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
