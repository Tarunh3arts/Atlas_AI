import React, { useState, useEffect } from 'react';
import HeroLanding from './components/HeroLanding.jsx';
import ChatConsole from './components/ChatConsole.jsx';
import { AnimatePresence, motion } from 'framer-motion';

export default function App() {
  const [currentView, setCurrentView] = useState('landing'); // 'landing' | 'chat'
  const [scenes, setScenes] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetch('/api/scenes')
      .then((res) => res.json())
      .then((data) => {
        if (data.scenes && data.scenes.length > 0) {
          setScenes(data.scenes);
        }
        setIsLoading(false);
      })
      .catch((err) => {
        console.error('Failed to fetch scenes from API:', err);
        setIsLoading(false);
      });
  }, []);

  if (isLoading) {
    return (
      <div className="bg-[#07090E] min-h-screen flex flex-col items-center justify-center p-6 text-center select-none">
        <div className="w-16 h-16 rounded-2xl overflow-hidden border border-purple-500/40 shadow-2xl shadow-purple-500/30 mb-4 bg-black/60">
          <video src="/logo.mp4" autoPlay loop muted playsInline className="w-full h-full object-cover" />
        </div>
        <div className="font-heading text-2xl tracking-[0.06em] text-white mb-1">Atlas AI</div>
        <div className="font-subtext text-xs text-cyan-400 tracking-wider">Loading Spatial Intelligence System...</div>
      </div>
    );
  }

  return (
    <div className="bg-[#07090E] min-h-screen text-slate-100">
      <AnimatePresence mode="wait">
        {currentView === 'landing' ? (
          <motion.div
            key="landing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <HeroLanding
              onLaunchConsole={() => setCurrentView('chat')}
              sampleScenes={scenes}
            />
          </motion.div>
        ) : (
          <motion.div
            key="chat"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <ChatConsole
              onBackHome={() => setCurrentView('landing')}
              scenes={scenes}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
