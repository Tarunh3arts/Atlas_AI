import React, { useRef, useEffect } from 'react';

/**
 * Interactive hover-to-play logo video:
 * When hovered, plays full until the end on every cursor hover.
 */
export function HoverLogoVideo({ src = "/logo.mp4", className = "w-20 h-20", onClick }) {
  const videoRef = useRef(null);

  const handleMouseEnter = () => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    video.play().catch(() => {});
  };

  const handleEnded = () => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    video.pause();
  };

  return (
    <div
      onMouseEnter={handleMouseEnter}
      onClick={onClick}
      className={`relative overflow-hidden rounded-2xl border border-white/10 shadow-xl shadow-purple-500/20 group-hover:scale-105 transition flex items-center justify-center bg-black/60 flex-shrink-0 ${className}`}
    >
      <video
        ref={videoRef}
        src={src}
        muted
        playsInline
        preload="auto"
        onEnded={handleEnded}
        className="w-full h-full object-cover pointer-events-none"
      />
    </div>
  );
}

/**
 * Borderless 2x playback loop logo video for AI thinking state:
 * - No text
 * - No borders or box outlines
 * - Plays strictly at 2x speed in continuous loop
 */
export function ThinkingLogoVideo({ className = "w-20 h-20" }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = 2.0;
    }
  }, []);

  return (
    <div className={`overflow-hidden rounded-2xl flex items-center justify-center pointer-events-none border-0 ring-0 outline-none ${className}`}>
      <video
        ref={videoRef}
        src="/logo.mp4"
        autoPlay
        loop
        muted
        playsInline
        onPlay={(e) => {
          e.currentTarget.playbackRate = 2.0;
        }}
        className="w-full h-full object-cover border-0"
      />
    </div>
  );
}

/**
 * Looping logo video loader for general uploading and page loading.
 */
export function LoopLogoLoader({ size = "xl", label, sublabel }) {
  const sizeClasses = {
    sm: "w-6 h-6 rounded-lg",
    md: "w-10 h-10 rounded-xl",
    lg: "w-14 h-14 rounded-2xl",
    xl: "w-20 h-20 rounded-2xl",
    "2xl": "w-24 h-24 rounded-3xl",
  };

  return (
    <div className="flex items-center gap-4">
      <div className={`overflow-hidden border border-purple-500/50 shadow-2xl shadow-purple-500/30 flex-shrink-0 bg-black/70 ${sizeClasses[size] || sizeClasses.xl}`}>
        <video
          src="/logo.mp4"
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        />
      </div>
      {(label || sublabel) && (
        <div className="flex flex-col text-left justify-center">
          {label && (
            <span className="font-subtext text-sm sm:text-base text-purple-200 font-semibold tracking-wide">
              {label}
            </span>
          )}
          {sublabel && (
            <span className="font-subtext text-xs text-slate-400 mt-0.5">
              {sublabel}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
