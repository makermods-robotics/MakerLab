// Sonner-compatible shim over the app's single (Radix) toast system.
// Keeps the sonner call-site API (toast.info/success/error/loading/dismiss)
// while rendering through the <Toaster /> mounted in App.tsx.
import { toast as radixToast } from "@/hooks/use-toast"

type SonnerOptions = {
  description?: string
  duration?: number
}

const dismissers = new Map<string, () => void>()

function show(
  title: string,
  options?: SonnerOptions,
  variant?: "default" | "destructive"
): string {
  const t = radixToast({
    title,
    description: options?.description,
    duration: options?.duration,
    variant,
  })
  dismissers.set(t.id, t.dismiss)
  return t.id
}

const toast = Object.assign(
  (title: string, options?: SonnerOptions) => show(title, options),
  {
    info: (title: string, options?: SonnerOptions) => show(title, options),
    success: (title: string, options?: SonnerOptions) => show(title, options),
    warning: (title: string, options?: SonnerOptions) => show(title, options),
    loading: (title: string, options?: SonnerOptions) => show(title, options),
    error: (title: string, options?: SonnerOptions) =>
      show(title, options, "destructive"),
    dismiss: (id?: string) => {
      if (id === undefined) {
        dismissers.forEach((dismiss) => dismiss())
        dismissers.clear()
        return
      }
      dismissers.get(id)?.()
      dismissers.delete(id)
    },
  }
)

export { toast }
