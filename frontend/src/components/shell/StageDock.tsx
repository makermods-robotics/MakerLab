import React from "react";
import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";

const STAGES: { to: string; label: string; gloss: string; end?: boolean }[] = [
  { to: "/", label: "Robot", gloss: "set up", end: true },
  { to: "/collect", label: "Collect", gloss: "teach" },
  { to: "/training", label: "Train & Deploy", gloss: "improve", end: true },
  { to: "/market", label: "Market", gloss: "discover" },
];

/** Floating bottom stage dock — the journey nav on stage pages. Offset right by
 * half the robots sidebar so it centers within the content area. */
const StageDock: React.FC = () => (
  <nav
    aria-label="Stages"
    className="fixed bottom-4 left-[calc(50%+144px)] z-30 flex -translate-x-1/2 gap-0.5 rounded-xl border border-border bg-card/95 p-1 shadow-2 backdrop-blur"
  >
    {STAGES.map((s) => (
      <NavLink
        key={s.to}
        to={s.to}
        end={s.end}
        className={({ isActive }) =>
          cn(
            "grid rounded-lg px-4 py-1.5 text-center no-underline",
            isActive ? "bg-primary" : "hover:bg-accent"
          )
        }
      >
        {({ isActive }) => (
          <>
            <span
              className={cn(
                "text-[12.5px] font-semibold",
                isActive ? "text-primary-foreground" : "text-muted-foreground"
              )}
            >
              {s.label}
            </span>
            <span
              className={cn(
                "text-[10.5px]",
                isActive
                  ? "text-primary-foreground/70"
                  : "text-muted-foreground/75"
              )}
            >
              {s.gloss}
            </span>
          </>
        )}
      </NavLink>
    ))}
  </nav>
);

export default StageDock;
