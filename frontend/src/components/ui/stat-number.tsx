import * as React from "react"

import { cn } from "@/lib/utils"

interface StatNumberProps extends React.HTMLAttributes<HTMLDivElement> {
  value: React.ReactNode
  label?: React.ReactNode
  sublabel?: React.ReactNode
  accent?: boolean
}

export function StatNumber({
  value,
  label,
  sublabel,
  accent = false,
  className,
  ...props
}: StatNumberProps) {
  return (
    <div className={cn("flex flex-col", className)} {...props}>
      {label && (
        <span className="font-mono text-[10px] uppercase tracking-[0.08em] text-muted-foreground">
          {label}
        </span>
      )}
      <span
        className={cn(
          "font-display text-3xl font-bold leading-none tracking-tight",
          accent ? "text-brand" : "text-foreground"
        )}
      >
        {value}
      </span>
      {sublabel && (
        <span className="mt-1 font-mono text-[10px] text-muted-foreground">
          {sublabel}
        </span>
      )}
    </div>
  )
}
