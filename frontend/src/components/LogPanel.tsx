import React, { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface LogPanelProps {
  /** The full log text to display (newline-separated). */
  logs: string;
  /** Panel heading, e.g. "Inference log" or "Recording log". */
  title?: string;
  /** Start collapsed. Defaults to expanded. */
  defaultCollapsed?: boolean;
  className?: string;
}

/**
 * A collapsible, monospace, auto-scrolling log viewer.
 *
 * Shared by the Inference and Record pages to surface backend log output that
 * previously only reached the server console. The parent polls its endpoint and
 * feeds the latest text in via `logs`; this component only renders and manages
 * scroll/collapse state. Auto-scrolls to the bottom on new content unless the
 * user has scrolled up (so reading history isn't yanked away by new lines).
 */
const LogPanel: React.FC<LogPanelProps> = ({
  logs,
  title = "Log",
  defaultCollapsed = false,
  className = "",
}) => {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const scrollRef = useRef<HTMLDivElement>(null);
  // Whether the view is pinned to the bottom. Starts true; flips false when the
  // user scrolls up, back true when they scroll back to the bottom.
  const pinnedRef = useRef(true);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight;
    pinnedRef.current = distanceFromBottom < 24;
  };

  useEffect(() => {
    if (collapsed) return;
    const el = scrollRef.current;
    if (el && pinnedRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs, collapsed]);

  return (
    <div
      className={`rounded-lg border border-gray-700 bg-gray-950 overflow-hidden ${className}`}
    >
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-400 hover:bg-gray-900"
      >
        {collapsed ? (
          <ChevronRight className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )}
        {title}
      </button>
      {!collapsed && (
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="max-h-64 overflow-y-auto px-3 pb-3 font-mono text-xs leading-relaxed text-gray-300 whitespace-pre-wrap break-words"
        >
          {logs ? (
            logs
          ) : (
            <span className="text-gray-600">Waiting for log output…</span>
          )}
        </div>
      )}
    </div>
  );
};

export default LogPanel;
