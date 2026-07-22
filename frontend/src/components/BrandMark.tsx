import React from "react";
import { cn } from "@/lib/utils";
import logoMark from "@/assets/logo-mark.png";

/**
 * The app's brand block: the MakerMods bracket mark beside the MakerLab
 * wordtext. `sm` fits the Launchpad route header; `md` is the studio overlay
 * header; `lg` is the Launchpad hero treatment.
 */
const BrandMark: React.FC<{
  size?: "sm" | "md" | "lg";
  className?: string;
}> = ({ size = "sm", className }) => (
  <span
    className={cn(
      "flex items-center",
      size === "sm" ? "gap-2" : "gap-2.5",
      className,
    )}
  >
    <img
      src={logoMark}
      alt="MakerMods"
      className={cn(
        "w-auto",
        size === "sm" ? "h-4" : size === "md" ? "h-6" : "h-7",
      )}
    />
    <span
      className={cn(
        "font-display font-semibold tracking-tight",
        size === "sm" ? "text-sm" : size === "md" ? "text-xl" : "text-2xl",
      )}
    >
      MakerLab
    </span>
  </span>
);

export default BrandMark;
