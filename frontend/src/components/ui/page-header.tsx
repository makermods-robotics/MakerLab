import * as React from "react"

import { cn } from "@/lib/utils"
import { Eyebrow } from "@/components/ui/eyebrow"

interface PageHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  eyebrow?: React.ReactNode
  title: React.ReactNode
  description?: React.ReactNode
  actions?: React.ReactNode
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
  ...props
}: PageHeaderProps) {
  return (
    <div
      className={cn("flex flex-wrap items-end justify-between gap-4", className)}
      {...props}
    >
      <div>
        {eyebrow && <Eyebrow className="mb-2">{eyebrow}</Eyebrow>}
        <h1 className="text-3xl">{title}</h1>
        {description && (
          <p className="mt-1 max-w-[68ch] text-sm text-muted-foreground">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  )
}
