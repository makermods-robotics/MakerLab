import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 font-display text-[11px] font-semibold uppercase tracking-[0.12em] transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground",
        secondary: "bg-secondary text-secondary-foreground",
        outline: "border border-input text-foreground",
        ok: "bg-ok/15 text-ok",
        warn: "bg-warn/15 text-warn",
        danger: "bg-destructive/15 text-destructive",
        destructive: "bg-destructive/15 text-destructive",
        stencil:
          "notch-sm rounded-none bg-primary px-2.5 py-1 font-bold tracking-[0.16em] text-primary-foreground",
        "stencil-brand":
          "notch-sm rounded-none bg-brand px-2.5 py-1 font-bold tracking-[0.16em] text-brand-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

function BadgeDot({
  pulse = false,
  className,
}: {
  pulse?: boolean
  className?: string
}) {
  return (
    <span
      className={cn(
        "h-1.5 w-1.5 rounded-full bg-current",
        pulse && "animate-pulse",
        className
      )}
    />
  )
}

export { Badge, BadgeDot, badgeVariants }
