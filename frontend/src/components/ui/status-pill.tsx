import * as React from "react"

import { Badge, BadgeDot } from "@/components/ui/badge"

export type SessionPhase =
  | "recording"
  | "resetting"
  | "running"
  | "setup"
  | "idle"

const phaseVariant: Record<SessionPhase, "danger" | "warn" | "ok" | "outline"> =
  {
    recording: "danger",
    resetting: "warn",
    running: "ok",
    setup: "outline",
    idle: "outline",
  }

export function StatusPill({
  phase,
  label,
  pulse = true,
}: {
  phase: SessionPhase
  label: React.ReactNode
  pulse?: boolean
}) {
  return (
    <Badge variant={phaseVariant[phase]}>
      <BadgeDot pulse={pulse && (phase === "recording" || phase === "running")} />
      {label}
    </Badge>
  )
}
