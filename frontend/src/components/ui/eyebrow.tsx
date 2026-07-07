import * as React from "react"

import { cn } from "@/lib/utils"

export function Eyebrow({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("eyebrow", className)} {...props} />
}
