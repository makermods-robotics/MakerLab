# MakerMods Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-skin the entire MakerLab frontend onto the MakerMods Design System (spec: `docs/superpowers/specs/2026-07-07-makermods-redesign-design.md`) with zero functionality loss.

**Architecture:** Foundation-first: rewrite the token layer + shadcn primitives + a new AppShell (Tasks 1–11, sequential), then re-skin pages/feature areas in parallel worktrees (Tasks 12–17), then merge, sweep, and verify visually + behaviorally without hardware (Tasks 18–20).

**Tech Stack:** React 18, Vite, TypeScript, Tailwind 3.4 + CSS variables, shadcn/ui (Radix + cva), lucide-react, @fontsource. Package manager: **npm** (package-lock.json). All frontend commands run in `frontend/`.

## Global Constraints

- **Functionality is untouchable.** Only classNames, JSX structure, copy casing, and imports of UI primitives may change. Never modify hooks, handlers, effects, API calls, WebSocket logic, polling, hotkeys, audio cues, guards, or props passed to logic components. Never delete "dead-looking" code.
- **Semantic tokens only** in `src/pages/**` and `src/components/**`: banned classes (grep-enforced): `bg-black`, `bg-white`, `text-white`, `text-black`, and any `gray-*`, `slate-*`, `zinc-*`, `neutral-*`, `stone-*`, `green-*`, `red-*`, `yellow-*`, `blue-*`, `orange-*` color utility. Functional states use `ok`/`warn`/`destructive`/`info` tokens; brand accent uses `brand`.
- **Orange budget:** one static orange anchor per screen (primary CTA *or* key stat). Focus rings + small stencil tags also may be orange. Never text color, page background, or state color.
- **Notch budget:** max one stencil-cut element (`.notch-md`/`.notch-lg` or Button `notch` / Card `notch`) per screen; small `Badge stencil` tags excluded.
- **Type:** `font-display` (Chakra Petch) for headings/buttons/eyebrows/big numbers; `font-body` (Space Grotesk) default; `font-mono` (JetBrains Mono) for ports, IDs, timers, joint values, logs, paths, hotkey hints, numeric dataset stats.
- **No** gradients (except the highlighter-underline motif), glassmorphism (except AppShell's single backdrop-blur), glow/colored shadows, hover translate lifts, emoji in UI copy, Title Case (sentence case everywhere).
- Motion: 120/200ms, `cubic-bezier(0.2,0,0,1)`; press = `active:scale-[0.98]`.
- Both themes must work on every screen; light is default.
- Fonts self-hosted via `@fontsource/*` — no CDN `<link>`s.
- Commit per task on the `redesign` branch (Phase 2 workers commit in their worktree branch).
- Verification per task: `npm run build` must pass (there is no JS unit-test suite; the Python tests don't cover the frontend).

---

## Phase 1 — Foundation (orchestrator/Fable 5, sequential)

### Task 1: Self-hosted fonts

**Files:**
- Modify: `frontend/package.json` (via npm install)
- Modify: `frontend/src/main.tsx`

**Interfaces:**
- Produces: font families available as CSS `font-family` names `"Chakra Petch"`, `"Space Grotesk"`, `"JetBrains Mono"` (consumed by Task 3's tailwind `fontFamily`).

- [ ] **Step 1: Install fontsource packages**

```bash
cd frontend && npm install @fontsource/chakra-petch @fontsource/space-grotesk @fontsource/jetbrains-mono
```

- [ ] **Step 2: Import weights in `src/main.tsx`** (top of file, before `./index.css`)

```tsx
import "@fontsource/chakra-petch/500.css";
import "@fontsource/chakra-petch/600.css";
import "@fontsource/chakra-petch/700.css";
import "@fontsource/space-grotesk/400.css";
import "@fontsource/space-grotesk/500.css";
import "@fontsource/space-grotesk/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/700.css";
import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

createRoot(document.getElementById("root")!).render(<App />);
```

- [ ] **Step 3: Verify build**

Run: `cd frontend && npm run build` → exits 0; `dist/assets` contains woff2 files.

- [ ] **Step 4: Commit** — `git add -A frontend/package.json frontend/package-lock.json frontend/src/main.tsx && git commit -m "feat(redesign): self-host MakerMods font stack via fontsource"`

### Task 2: Token layer — rewrite `src/index.css`

**Files:**
- Modify: `frontend/src/index.css` (full replace)

**Interfaces:**
- Produces: shadcn semantic vars re-valued; new vars `--brand`, `--brand-foreground`, `--brand-hover`, `--brand-tint`, `--ok`, `--warn`, `--info`, `--shadow-1`, `--shadow-2`, `--notch-sm/md/lg`; utilities `.grid-bg`, `.notch-sm/md/lg`, `.eyebrow`; base heading font rules. Consumed by every later task.

- [ ] **Step 1: Replace `frontend/src/index.css` entirely with:**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* MakerMods Design System tokens — black + white + maker orange.
   HSL triples so Tailwind opacity modifiers work. */

@layer base {
  :root {
    --background: 0 0% 98%;        /* paper #FAFAFA */
    --foreground: 0 0% 4%;         /* ink #0A0A0A */

    --card: 0 0% 100%;             /* surface #FFFFFF */
    --card-foreground: 0 0% 4%;

    --popover: 0 0% 100%;
    --popover-foreground: 0 0% 4%;

    --primary: 0 0% 4%;            /* ink fill */
    --primary-foreground: 0 0% 98%;

    --secondary: 0 0% 96%;         /* #F5F5F5 */
    --secondary-foreground: 0 0% 4%;

    --muted: 0 0% 96%;
    --muted-foreground: 0 0% 45%;  /* mute #737373 */

    --accent: 0 0% 90%;            /* line #E5E5E5 — hover fills */
    --accent-foreground: 0 0% 4%;

    --destructive: 0 72% 51%;      /* #DC2626 */
    --destructive-foreground: 0 0% 98%;

    --border: 0 0% 90%;            /* line #E5E5E5 */
    --input: 0 0% 83%;             /* line-2 #D4D4D4 */
    --ring: 18 100% 59%;           /* orange focus ring */

    --brand: 18 100% 59%;          /* #FF6B2C */
    --brand-foreground: 0 0% 4%;   /* always dark on orange */
    --brand-hover: 17 81% 50%;     /* #E55418 */
    --brand-tint: 23 100% 91%;     /* #FFE3D2 */

    --ok: 142 76% 36%;             /* #16A34A */
    --warn: 32 95% 44%;            /* #D97706 */
    --info: 221 83% 53%;           /* #2563EB */

    --radius: 0.375rem;            /* 6px → lg 6 / md 4 / sm 2 */

    --shadow-1: 0 1px 0 rgba(10, 10, 10, 0.06), 0 1px 2px rgba(10, 10, 10, 0.04);
    --shadow-2: 0 4px 12px rgba(10, 10, 10, 0.08), 0 2px 4px rgba(10, 10, 10, 0.04);

    --notch-sm: polygon(8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%, 0 8px);
    --notch-md: polygon(14px 0, 100% 0, 100% calc(100% - 14px), calc(100% - 14px) 100%, 0 100%, 0 14px);
    --notch-lg: polygon(24px 0, 100% 0, 100% calc(100% - 24px), calc(100% - 24px) 100%, 0 100%, 0 24px);
  }

  .dark {
    --background: 0 0% 4%;
    --foreground: 0 0% 96%;        /* #F5F5F5 */

    --card: 0 0% 8%;               /* #141414 */
    --card-foreground: 0 0% 96%;

    --popover: 0 0% 8%;
    --popover-foreground: 0 0% 96%;

    --primary: 0 0% 96%;
    --primary-foreground: 0 0% 4%;

    --secondary: 0 0% 15%;         /* #262626 */
    --secondary-foreground: 0 0% 96%;

    --muted: 0 0% 15%;
    --muted-foreground: 0 0% 64%;  /* #A3A3A3 */

    --accent: 0 0% 15%;
    --accent-foreground: 0 0% 96%;

    --destructive: 0 84% 60%;      /* #EF4444 */
    --destructive-foreground: 0 0% 98%;

    --border: 0 0% 15%;
    --input: 0 0% 25%;             /* #404040 */
    --ring: 18 100% 63%;

    --brand: 18 100% 63%;          /* #FF7A40 */
    --brand-foreground: 0 0% 4%;
    --brand-hover: 20 100% 69%;    /* #FF9460 */
    --brand-tint: 20 53% 15%;      /* #3A1F12 */

    --ok: 142 71% 45%;             /* #22C55E */
    --warn: 38 92% 50%;            /* #F59E0B */
    --info: 213 94% 68%;           /* #60A5FA */

    --shadow-1: 0 1px 0 rgba(0, 0, 0, 0.6), 0 1px 2px rgba(0, 0, 0, 0.4);
    --shadow-2: 0 4px 12px rgba(0, 0, 0, 0.6), 0 2px 4px rgba(0, 0, 0, 0.4);
  }
}

@layer base {
  * {
    @apply border-border;
  }

  body {
    @apply bg-background font-body text-foreground antialiased;
  }

  h1, h2, h3, h4 {
    @apply font-display font-bold tracking-tight;
    text-wrap: balance;
  }

  ::selection {
    @apply bg-foreground text-background;
  }
}

@layer utilities {
  .grid-bg {
    background-image:
      linear-gradient(to right, hsl(var(--border)) 1px, transparent 1px),
      linear-gradient(to bottom, hsl(var(--border)) 1px, transparent 1px);
    background-size: 24px 24px;
    background-position: -1px -1px;
  }

  .notch-sm { clip-path: var(--notch-sm); border-radius: 0; }
  .notch-md { clip-path: var(--notch-md); border-radius: 0; }
  .notch-lg { clip-path: var(--notch-lg); border-radius: 0; }

  .eyebrow {
    @apply font-display text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground;
  }
}
```

- [ ] **Step 2: Verify build** — `cd frontend && npm run build` → exits 0.
- [ ] **Step 3: Commit** — `git commit -am "feat(redesign): MakerMods token layer (light default + dark), grid/notch/eyebrow utilities"`

### Task 3: Tailwind config

**Files:**
- Modify: `frontend/tailwind.config.ts` (full replace)

**Interfaces:**
- Produces classes: `font-display/font-body/font-mono`, `bg-brand`, `text-brand-foreground`, `bg-brand-hover`, `bg-brand-tint`, `text-ok/warn/info`, `bg-ok/warn/info` (+ `/12` opacity variants via HSL), `shadow-1`, `shadow-2`. Sidebar colors removed.

- [ ] **Step 1: Replace `frontend/tailwind.config.ts` with:**

```ts
import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1440px'
			}
		},
		extend: {
			fontFamily: {
				display: ['"Chakra Petch"', 'system-ui', 'sans-serif'],
				body: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
				mono: ['"JetBrains Mono"', 'ui-monospace', 'Menlo', 'monospace'],
			},
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				brand: {
					DEFAULT: 'hsl(var(--brand))',
					foreground: 'hsl(var(--brand-foreground))',
					hover: 'hsl(var(--brand-hover))',
					tint: 'hsl(var(--brand-tint))'
				},
				ok: 'hsl(var(--ok))',
				warn: 'hsl(var(--warn))',
				info: 'hsl(var(--info))'
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			boxShadow: {
				'1': 'var(--shadow-1)',
				'2': 'var(--shadow-2)'
			},
			transitionTimingFunction: {
				std: 'cubic-bezier(0.2, 0, 0, 1)'
			},
			keyframes: {
				'accordion-down': {
					from: { height: '0' },
					to: { height: 'var(--radix-accordion-content-height)' }
				},
				'accordion-up': {
					from: { height: 'var(--radix-accordion-content-height)' },
					to: { height: '0' }
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out'
			}
		}
	},
	plugins: [require("tailwindcss-animate")],
} satisfies Config;
```

- [ ] **Step 2: Verify build + no sidebar refs** — `npm run build` exits 0; `grep -rn "sidebar-" frontend/src --include='*.tsx' --include='*.ts' --include='*.css'` → empty (if any hits, they're template leftovers to delete in that file).
- [ ] **Step 3: Commit** — `git commit -am "feat(redesign): tailwind fonts, brand/functional colors, DS radii and shadows"`

### Task 4: Brand assets

**Files:**
- Create: `frontend/public/makermods/logo-mark.png`, `frontend/public/makermods/logo-mark-white.png`

**Interfaces:**
- Produces: `<img src="/makermods/logo-mark.png">` (light) / `/makermods/logo-mark-white.png` (dark) used by Task 11's Logo.

- [ ] **Step 1 (orchestrator only — needs DesignSync):** fetch `assets/logo-mark.png` and `assets/logo-mark-white.png` from the design project (`get_file`, base64) and write to `frontend/public/makermods/`. Verify both files open as valid PNGs (`file frontend/public/makermods/*.png` reports PNG image data).
- [ ] **Step 2: Commit** — `git add frontend/public/makermods && git commit -m "feat(redesign): add MakerMods logo mark assets"`

### Task 5: Button primitive

**Files:**
- Modify: `frontend/src/components/ui/button.tsx`

**Interfaces:**
- Produces variants: `default | secondary | outline | ghost | link | destructive | brand | notch | notch-brand`; sizes `sm | default | lg | icon`. `outline` kept as alias of secondary styling (existing pages import it). Everything else about ButtonProps is unchanged.

- [ ] **Step 1: Replace the cva block in `button.tsx` with:**

```tsx
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md font-display text-sm font-semibold tracking-[0.02em] ring-offset-background transition-all duration-[120ms] ease-std focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:opacity-85",
        secondary: "border border-input bg-card text-foreground hover:bg-accent",
        outline: "border border-input bg-card text-foreground hover:bg-accent",
        ghost: "text-foreground hover:bg-accent",
        link: "text-foreground underline underline-offset-4 hover:opacity-70",
        destructive: "border border-destructive bg-transparent text-destructive hover:bg-destructive hover:text-destructive-foreground",
        brand: "bg-brand text-brand-foreground hover:bg-brand-hover",
        notch: "notch-sm bg-primary text-primary-foreground hover:opacity-85",
        "notch-brand": "notch-sm bg-brand text-brand-foreground hover:bg-brand-hover",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 px-3 text-xs",
        lg: "h-11 px-6 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  }
)
```

Keep the rest of the file (imports, ButtonProps, forwardRef, exports) byte-identical.

- [ ] **Step 2: Verify** — `npm run build` exits 0 (all existing `variant="outline"` etc. still typecheck).
- [ ] **Step 3: Commit** — `git commit -am "feat(redesign): MakerMods button variants incl. brand + stencil notch"`

### Task 6: Badge primitive

**Files:**
- Modify: `frontend/src/components/ui/badge.tsx` (full replace)

**Interfaces:**
- Produces: `Badge` variants `default | outline | ok | warn | danger | stencil | stencil-brand`; export `BadgeDot` (6px currentColor dot, optional `pulse`). Existing imports of `Badge`/`badgeVariants` keep working (`secondary`/`destructive` kept as aliases).

- [ ] **Step 1: Replace file with:**

```tsx
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
    defaultVariants: { variant: "default" },
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

function BadgeDot({ pulse = false, className }: { pulse?: boolean; className?: string }) {
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
```

- [ ] **Step 2: Verify** — `npm run build` exits 0.
- [ ] **Step 3: Commit** — `git commit -am "feat(redesign): badge status/stencil variants + BadgeDot"`

### Task 7: Card primitive

**Files:**
- Modify: `frontend/src/components/ui/card.tsx`

**Interfaces:**
- Produces: `Card` gains optional `variant?: "default" | "flat" | "notch" | "inverted"` prop (cva). `CardTitle` becomes `font-display`. All sub-components (`CardHeader`, `CardContent`, etc.) keep identical signatures.

- [ ] **Step 1: In `card.tsx`, add cva import and replace the `Card` component:**

```tsx
import { cva, type VariantProps } from "class-variance-authority"

const cardVariants = cva("rounded-lg border text-card-foreground", {
  variants: {
    variant: {
      default: "border-border bg-card shadow-1",
      flat: "border-border bg-card",
      notch: "notch-md border-border bg-card",
      inverted: "border-primary bg-primary text-primary-foreground",
    },
  },
  defaultVariants: { variant: "default" },
})

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof cardVariants>
>(({ className, variant, ...props }, ref) => (
  <div ref={ref} className={cn(cardVariants({ variant }), className)} {...props} />
))
```

Also change `CardTitle`'s classes to `"font-display text-xl font-bold leading-snug tracking-tight"` (keep everything else identical).

- [ ] **Step 2: Verify** — `npm run build` exits 0.
- [ ] **Step 3: Commit** — `git commit -am "feat(redesign): card variants (flat/notch/inverted), display-font titles"`

### Task 8: Form primitives

**Files:**
- Modify: `frontend/src/components/ui/input.tsx`, `frontend/src/components/ui/label.tsx`, `frontend/src/components/ui/textarea.tsx`, `frontend/src/components/ui/select.tsx`, `frontend/src/components/ui/number-input.tsx` (class strings only)

**Interfaces:**
- No API changes anywhere — className strings only.

- [ ] **Step 1: `input.tsx`** — replace the input's class string with:

```
"flex h-10 w-full rounded-sm border border-input bg-card px-3 py-2 text-sm text-foreground ring-offset-background transition-colors duration-[120ms] file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
```

- [ ] **Step 2: `label.tsx`** — replace the cva base string with:

```
"font-display text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
```

- [ ] **Step 3: `textarea.tsx`** — same treatment as input (rounded-sm, border-input, bg-card, focus ring-ring).
- [ ] **Step 4: `select.tsx`** — SelectTrigger: `rounded-sm border-input bg-card focus:ring-ring`; SelectContent: `rounded-md border-border bg-popover shadow-2`.
- [ ] **Step 5: `number-input.tsx`** — align its input classes with Step 1's string (keep its custom stepper logic byte-identical).
- [ ] **Step 6: Verify** — `npm run build` exits 0.
- [ ] **Step 7: Commit** — `git commit -am "feat(redesign): DS form fields (uppercase labels, orange focus, 2-4px radii)"`

### Task 9: Overlays + toast consolidation

**Files:**
- Modify: `frontend/src/components/ui/dialog.tsx`, `frontend/src/components/ui/alert-dialog.tsx`, `frontend/src/components/ui/toast.tsx`, `frontend/src/components/ui/dropdown-menu.tsx`, `frontend/src/components/ui/popover.tsx`, `frontend/src/components/ui/tooltip.tsx`
- Modify: `frontend/src/contexts/UrdfContext.tsx`, `frontend/src/lib/urdfViewerHelpers.ts` (sonner → use-toast)
- Delete: `frontend/src/components/ui/sonner.tsx`
- Modify: `frontend/package.json` (remove `sonner`, `next-themes`)

**Interfaces:**
- Consumes: nothing new. Produces: single toast system (`@/hooks/use-toast`). Overlay styling per DS.

- [ ] **Step 1: `dialog.tsx` + `alert-dialog.tsx`** — Overlay class → `"fixed inset-0 z-50 bg-foreground/60"` (keep animation classes, **no** backdrop-blur); Content class → swap `rounded-lg` for `rounded-xl` (12px), add `shadow-2 border-border bg-card`; DialogTitle → `font-display text-lg font-bold tracking-tight`.
- [ ] **Step 2: `dropdown-menu.tsx`, `popover.tsx`, `tooltip.tsx`** — content surfaces: `rounded-md border-border bg-popover shadow-2`.
- [ ] **Step 3: `toast.tsx`** — root: `rounded-md border-border bg-card shadow-2`; destructive variant: `border-destructive bg-destructive text-destructive-foreground`; title `font-display font-semibold`.
- [ ] **Step 4: Migrate sonner call sites.** In `UrdfContext.tsx` and `urdfViewerHelpers.ts`: replace `import { toast } from "sonner"` with `import { toast } from "@/hooks/use-toast"` and convert each call — sonner `toast.error("msg")` → `toast({ title: "msg", variant: "destructive" })`; `toast.success("msg")` / `toast("msg")` → `toast({ title: "msg" })`; `toast.info("msg", { description: d })` → `toast({ title: "msg", description: d })`. **Do not change surrounding logic.** (Note: the sonner `<Toaster/>` was never mounted, so these were silent — after migration they surface via the mounted Radix toaster, which is the intended fix.)
- [ ] **Step 5: Remove dead deps** — delete `frontend/src/components/ui/sonner.tsx`; `cd frontend && npm uninstall sonner next-themes`.
- [ ] **Step 6: Verify** — `npm run build` exits 0; `grep -rn "sonner\|next-themes" frontend/src` → empty.
- [ ] **Step 7: Commit** — `git commit -am "feat(redesign): DS overlays, single toast system, drop sonner + next-themes"`

### Task 10: New primitives

**Files:**
- Create: `frontend/src/components/ui/eyebrow.tsx`, `frontend/src/components/ui/stat-number.tsx`, `frontend/src/components/ui/status-pill.tsx`, `frontend/src/components/ui/page-header.tsx`

**Interfaces (produces — exact signatures Phase 2 relies on):**

```tsx
<Eyebrow>{children}</Eyebrow>                                  // brackets NOT auto-added; write "[ Robot ]" yourself
<StatNumber value="48.2k" label="of 100k" accent />            // accent = the screen's one orange
<StatusPill phase="recording" label="Recording · ep 7/50" />   // phase: "recording"|"resetting"|"running"|"setup"|"idle"
<PageHeader eyebrow="[ Training ]" title="Configure a run" actions={<.../>} />
```

- [ ] **Step 1: `eyebrow.tsx`**

```tsx
import { cn } from "@/lib/utils"

export function Eyebrow({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("eyebrow", className)} {...props} />
}
```

- [ ] **Step 2: `stat-number.tsx`**

```tsx
import { cn } from "@/lib/utils"

interface StatNumberProps extends React.HTMLAttributes<HTMLDivElement> {
  value: React.ReactNode
  label?: React.ReactNode
  sublabel?: React.ReactNode
  accent?: boolean
}

export function StatNumber({ value, label, sublabel, accent = false, className, ...props }: StatNumberProps) {
  return (
    <div className={cn("flex flex-col", className)} {...props}>
      {label && (
        <span className="font-mono text-[10px] uppercase tracking-[0.08em] text-muted-foreground">{label}</span>
      )}
      <span className={cn("font-display text-3xl font-bold leading-none tracking-tight", accent ? "text-brand" : "text-foreground")}>
        {value}
      </span>
      {sublabel && <span className="mt-1 font-mono text-[10px] text-muted-foreground">{sublabel}</span>}
    </div>
  )
}
```

- [ ] **Step 3: `status-pill.tsx`**

```tsx
import { Badge, BadgeDot } from "@/components/ui/badge"

export type SessionPhase = "recording" | "resetting" | "running" | "setup" | "idle"

const phaseVariant: Record<SessionPhase, "danger" | "warn" | "ok" | "outline"> = {
  recording: "danger",
  resetting: "warn",
  running: "ok",
  setup: "outline",
  idle: "outline",
}

export function StatusPill({ phase, label, pulse = true }: { phase: SessionPhase; label: React.ReactNode; pulse?: boolean }) {
  return (
    <Badge variant={phaseVariant[phase]}>
      <BadgeDot pulse={pulse && (phase === "recording" || phase === "running")} />
      {label}
    </Badge>
  )
}
```

- [ ] **Step 4: `page-header.tsx`**

```tsx
import { cn } from "@/lib/utils"
import { Eyebrow } from "@/components/ui/eyebrow"

interface PageHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  eyebrow?: React.ReactNode
  title: React.ReactNode
  description?: React.ReactNode
  actions?: React.ReactNode
}

export function PageHeader({ eyebrow, title, description, actions, className, ...props }: PageHeaderProps) {
  return (
    <div className={cn("flex flex-wrap items-end justify-between gap-4", className)} {...props}>
      <div>
        {eyebrow && <Eyebrow className="mb-2">{eyebrow}</Eyebrow>}
        <h1 className="text-3xl">{title}</h1>
        {description && <p className="mt-1 max-w-[68ch] text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  )
}
```

(Add `import * as React from "react"` where TS requires it.)

- [ ] **Step 5: Verify** — `npm run build` exits 0.
- [ ] **Step 6: Commit** — `git commit -am "feat(redesign): Eyebrow, StatNumber, StatusPill, PageHeader primitives"`

### Task 11: Logo, ThemeToggle, AppShell, light default

**Files:**
- Modify: `frontend/src/components/Logo.tsx`
- Create: `frontend/src/components/shell/ThemeToggle.tsx`, `frontend/src/components/shell/AppShell.tsx`
- Modify: `frontend/src/contexts/ThemeContext.tsx:12` and `:27` (`"system"` → `"light"` in `initialState` and `defaultTheme`)

**Interfaces (produces):**

```tsx
<AppShell
  back?: { to?: string; onClick?: () => void; label?: string }   // renders "← <label|back>" ghost button
  status?: React.ReactNode      // StatusPill slot, centered
  actions?: React.ReactNode     // right side, page-specific (e.g. Stop button)
  showAuthChip?: boolean        // default true; HfAuthChip
  fullBleed?: boolean           // default false; true = children not wrapped in max-w container
>{children}</AppShell>
```

- [ ] **Step 1: `Logo.tsx`** — keep the `iconOnly` prop and default export; new body:

```tsx
import React from 'react';
import { cn } from '@/lib/utils';

interface LogoProps extends React.HTMLAttributes<HTMLDivElement> {
  iconOnly?: boolean;
}

const Logo: React.FC<LogoProps> = ({ className, iconOnly = false }) => {
  return (
    <div className={cn('flex items-center gap-2.5', className)}>
      <img src="/makermods/logo-mark.png" alt="MakerLab" className="h-7 w-7 dark:hidden" />
      <img src="/makermods/logo-mark-white.png" alt="MakerLab" className="hidden h-7 w-7 dark:block" />
      {!iconOnly && (
        <span className="font-display text-[15px] font-bold tracking-[0.06em] text-foreground">MAKERLAB</span>
      )}
    </div>
  );
};

export default Logo;
```

- [ ] **Step 2: `shell/ThemeToggle.tsx`**

```tsx
import { useContext } from 'react';
import { Moon, Sun } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ThemeProviderContext } from '@/contexts/ThemeContext';

export function ThemeToggle() {
  const { theme, setTheme } = useContext(ThemeProviderContext);
  const isDark =
    theme === 'dark' ||
    (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
    >
      {isDark ? <Sun /> : <Moon />}
    </Button>
  );
}
```

- [ ] **Step 3: `shell/AppShell.tsx`**

```tsx
import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import Logo from '@/components/Logo';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/shell/ThemeToggle';
import HfAuthChip from '@/components/landing/HfAuthChip';
import { cn } from '@/lib/utils';

interface AppShellProps {
  back?: { to?: string; onClick?: () => void; label?: string };
  status?: React.ReactNode;
  actions?: React.ReactNode;
  showAuthChip?: boolean;
  fullBleed?: boolean;
  children: React.ReactNode;
}

export function AppShell({ back, status, actions, showAuthChip = true, fullBleed = false, children }: AppShellProps) {
  const navigate = useNavigate();
  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur-[8px] backdrop-saturate-150">
        <div className="mx-auto flex h-[52px] max-w-[1440px] items-center gap-3 px-4">
          <Link to="/" aria-label="Home">
            <Logo />
          </Link>
          {back && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => (back.onClick ? back.onClick() : back.to ? navigate(back.to) : navigate(-1))}
            >
              <ArrowLeft /> {back.label ?? 'back'}
            </Button>
          )}
          <div className="flex flex-1 items-center justify-center">{status}</div>
          <div className="flex items-center gap-2">
            {actions}
            {showAuthChip && <HfAuthChip />}
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main className={cn('flex-1', !fullBleed && 'mx-auto w-full max-w-[1440px] px-4 py-6')}>{children}</main>
    </div>
  );
}
```

**Note:** check `HfAuthChip`'s actual export name/props first (`src/components/landing/HfAuthChip.tsx`) and adapt the import; if it requires props, wire the same ones `LandingTopBar` passes today.

- [ ] **Step 4: ThemeContext default** — change `"system"` → `"light"` at both `initialState.theme` and `defaultTheme` param. Nothing else.
- [ ] **Step 5: Verify** — `npm run build` exits 0. Run `npm run dev`, load `/` — old pages still render (still dark-styled, that's Phase 2); toggle in DOM not yet mounted anywhere (AppShell unused until Phase 2) — that's expected.
- [ ] **Step 6: Commit** — `git commit -am "feat(redesign): AppShell + ThemeToggle + MakerMods logo, light default"`

**Phase 1 checkpoint (orchestrator):** `npm run build` + `npm run lint` clean; screenshot `/` in dev to confirm app still functions before fan-out.

---

## Phase 2 — Page fan-out (parallel workers, one worktree each)

**Worker setup (every task):** work in a git worktree branched off `redesign` after Phase 1 is merged. Claude workers: Opus 4.8 (`model: 'opus'`). gpt-5.5 workers: thin sonnet wrapper running `codex exec` per the orchestrator's wrapper pattern. Every worker prompt must embed: the **Global Constraints** section verbatim, the interfaces from Tasks 5–11, and the per-task recipe below.

**Universal mapping recipe (applies to every Phase 2 task):**

| Old (hardcoded) | New (token) |
| --- | --- |
| `bg-black`, `bg-gray-900`, `bg-slate-900` (page bg) | remove — `AppShell` provides `bg-background` |
| `bg-gray-800/900`, `bg-slate-800` (panels) | `Card` (`bg-card border-border`) |
| `border-gray-700`, `border-slate-700` | `border-border` |
| `text-white` | `text-foreground` |
| `text-gray-400/500`, `text-slate-400` | `text-muted-foreground` |
| `bg-green-500`, `text-green-*` (status) | `ok` token via Badge `ok` / `text-ok` |
| `bg-red-*`, `text-red-*` (errors/rec) | `destructive` token |
| `bg-yellow-*`/`amber` | `warn` token |
| `bg-blue-*` (links/info) | `info` token |
| headings `text-2xl/3xl font-bold` | `<PageHeader>` or `font-display` heading |
| timers/ports/counts | add `font-mono` |
| ad-hoc header (`ArrowLeft` + Logo + h1) | `AppShell` `back` prop + `PageHeader` |

**Universal must-preserve:** every `useEffect`, handler, fetch, WebSocket, `navigate()` target, dialog open/close state, keyboard listener, audio element, and conditional render must remain. Diff should show only className/markup/import changes plus the AppShell wrapper.

**Universal verification (every Phase 2 task):** `npm run build` exits 0; `npm run lint` no new errors; banned-class grep over the task's files returns empty:

```bash
grep -rnE "(bg|text|border|ring|from|to|via|fill|stroke)-(gray|slate|zinc|neutral|stone|green|red|yellow|amber|blue|orange)-[0-9]+|bg-black|bg-white|text-white|text-black" <task files> || echo CLEAN
```

Then commit: `git commit -am "feat(redesign): re-skin <area>"`.

### Task 12: Landing + landing/* + jobs/* + Footer

**Files:** `src/pages/Landing.tsx`, `src/components/landing/*.tsx` (18 files), `src/components/jobs/*.tsx` (6 files), `src/components/Footer.tsx`. Delete `src/components/landing/LandingTopBar.tsx` **only if** Landing is its sole importer (grep first); its HfAuthChip usage moves to AppShell.

**Recipe:** wrap page in `<AppShell showAuthChip>` (no `back`). Hero strip: `grid-bg` section with `<Eyebrow>[ SO-101 · leader + follower ]</Eyebrow>`, display headline, mono robot-connection line. Robot/Dataset cards: `Card` with `<Eyebrow>[ Robot ]</Eyebrow>` / `[ Dataset ]`. **The screen's one orange** = the record CTA (`Button variant="notch-brand"`, copy `[ Record episodes ]`). Jobs: bordered `Card variant="flat"` rows, mono metadata, `Badge ok/warn/danger` for statuses. Footer: `border-t border-border`, mono small links, sentence case. All dialogs opened from Landing (recording modal, create/merge/upload dataset, usage instructions, delete confirm, HfAuth dialog) restyled via the primitives — structure/props untouched.

**Verify (behavioral, no hardware):** dev server: `/` renders in both themes; robot selector opens; dataset picker opens; every dialog opens/closes; record flow still navigates to `/recording` (will error without backend — same as before).

### Task 13: Teleoperation + control/*

**Files:** `src/pages/Teleoperation.tsx`, `src/components/control/*.tsx` (5 files), `src/components/UrdfViewer.tsx`, `src/components/BackendCameraStream.tsx` (chrome only — canvas/stream internals untouched).

**Recipe:** `<AppShell fullBleed status={<StatusPill phase="running" label="Teleoperating" />} actions={<Done button>}>`. Viewer + camera panels: `Card variant="flat"` frames with 1px `border-border`; the 3D canvas/camera pixels inside are content — leave their internal colors alone. Joint readouts/metrics: `font-mono`. Command bar: bottom toolbar `border-t border-border bg-card`. **No orange** on this screen unless there's a primary CTA (there isn't — Done is `secondary`). Preserve every exit-path effect that stops teleop (unmount, beforeunload, Done, back).

**Verify:** `/teleoperation` renders both themes; Done navigates home; no console errors beyond the pre-existing backend-unreachable ones.

### Task 14: Recording + Inference + recording/*

**Files:** `src/pages/Recording.tsx`, `src/pages/Inference.tsx`, `src/pages/Upload.tsx`, `src/components/recording/CameraConfiguration.tsx`.

**Recipe:** shared HUD family. `<AppShell fullBleed status={<StatusPill …/>} actions={<Stop/>}>`; center a `Card variant="notch"` (the screen's one notch) max-w-md on a `grid-bg` field: `<Eyebrow>[ Episode 07 · dataset-name ]</Eyebrow>`, timer `font-mono text-6xl font-bold`, `/ 60s max` mono caption, 4px ink progress bar (`bg-secondary` track, `bg-primary` fill; keep exact progress math), primary action `Button variant="brand"` full-width (`End episode → next` / current copy), ghost re-record/mute row, mono hotkey hints (`[ space ] end episode`). Phase mapping: recording→`danger` pill, resetting→`warn`, inference RUNNING→`ok`, SETTING UP→`setup`. Keep Space/→ keyboard listeners, polling, audio cues, all confirm dialogs. Upload page: `AppShell` + centered `Card` confirmation.

**Verify:** `/recording`, `/inference`, `/upload` render both themes; hotkey listeners still bound (keydown handlers present); stop-confirm dialog opens.

### Task 15: Training + training/* + CheckpointDropdown

**Files:** `src/pages/Training.tsx`, `src/components/training/**/*.tsx` (~12 files).

**Recipe:** ConfigurationMode: `<AppShell back={{to:'/'}}>` + `PageHeader eyebrow="[ Training ]"`; two-column form on `Card`s; uppercase `Label`s; mono inputs for numeric/technical values (steps, lr, batch); Start = `Button variant="brand"` (**screen's orange**). MonitoringMode: `PageHeader` with job name (mono) + `StatusPill`; 4 `StatNumber` tiles in `Card`s — step count gets `accent` (**that screen's orange; Start button isn't rendered in this mode so no conflict**); `TrainingLogs` inside `Card variant="inverted"` with `font-mono text-xs leading-relaxed`; checkpoint dropdown via restyled dropdown primitive. W&B links `text-info underline`.

**Verify:** `/training` renders both themes; mode switch works (`/training` vs `/training/:jobId`); form inputs accept values; install/gate dialogs open.

### Task 16: Calibration + calibration/*

**Files:** `src/pages/Calibration.tsx` (~2022 lines — highest regression risk, change classNames only), `src/components/calibration/*.tsx`.

**Recipe:** `<AppShell back={{to:'/'}}>` + `PageHeader eyebrow="[ Calibration ]"`. Two-column: config `Card` (device/arm/port selects; port values `font-mono`; Detect/Wiggle `secondary`) + progress panel. Stepper: mono step numbers (`01 02 03` style), current step `text-foreground`, done `text-ok`, pending `text-muted-foreground`. Ranges table `font-mono text-xs`. Start-calibration CTA = `Button variant="brand"` (**screen's orange**). Motor-power slider keeps slider primitive (restyle track to `bg-secondary`, thumb `bg-primary`). Every dialog (overwrite, auto-cal warning, port swap) restyled, logic untouched.

**Verify:** `/calibration` renders both themes; device/arm selects work; dialogs open; port rescan button triggers its handler (network error toast acceptable without backend).

### Task 17: Misc surfaces

**Files:** `src/pages/NotFound.tsx`, `src/pages/EditDataset.tsx`, `src/components/TeleopStopNotice.tsx`, `src/components/UpdateNotice.tsx`, `src/components/SingleTabGuard.tsx`, `src/components/replay/DatasetCombobox.tsx`.

**Recipe:** NotFound: `AppShell` + centered `Badge variant="stencil"`-style `[ 404 ]` + display heading "page not found" + ghost link `← back to the workshop`. Note NotFound currently is the app's only light page (`bg-gray-100`) — now it's just tokens. EditDataset: `AppShell` + `Card` with `<Eyebrow>[ Under construction ]</Eyebrow>`. Notices/guards: token colors, `Card`/`Badge` primitives, mono details. DatasetCombobox: restyled command/popover primitives (Task 9 already did most).

**Verify:** `/nonexistent` shows 404 both themes; `/edit-dataset` renders.

---

## Phase 3 — Merge, sweep, inspect (orchestrator/Fable 5)

### Task 18: Merge + global sweep

- [ ] **Step 1:** merge each Phase 2 worktree branch into `redesign` (resolve conflicts — expected only in shared imports).
- [ ] **Step 2:** global banned-class grep over `frontend/src/pages frontend/src/components` (command from Phase 2 preamble, minus `components/ui`) → must print CLEAN; fix stragglers.
- [ ] **Step 3:** `grep -rn "font-bold text-2xl\|text-3xl font-bold" frontend/src/pages` → each hit should be a `PageHeader`/display heading, not body text.
- [ ] **Step 4:** orange budget audit: `grep -rln "brand" frontend/src/pages frontend/src/components | xargs grep -cn "variant=\"brand\"\|variant=\"notch-brand\"\|accent"` — manually confirm ≤1 static anchor per screen.
- [ ] **Step 5:** `npm run build && npm run lint` clean. Commit merge.

### Task 19: Visual inspection (both themes, every route)

- [ ] **Step 1:** `cd frontend && npm run dev` (port 8080). Backend optional: if `makerlab` CLI is installed, run it for live data; otherwise pages must degrade exactly as they did pre-redesign.
- [ ] **Step 2:** screenshot every route (`/`, `/teleoperation`, `/recording`, `/upload`, `/training`, `/training/fake-id`, `/inference`, `/calibration`, `/edit-dataset`, `/nonexistent`) in light and dark via browser automation.
- [ ] **Step 3:** judge each against the DS: paper/ink contrast, one orange, one notch, mono where technical, eyebrows bracketed, no lingering gray/slate surfaces, dialogs 12px/solid overlay, shell blur only.
- [ ] **Step 4:** punch-list fixes applied directly; re-screenshot; commit `fix(redesign): visual punch list`.

### Task 20: Behavioral sanity (no hardware) + independent review

- [ ] **Step 1 (browser-driven):** theme toggle flips + persists across reload; nav flows Landing↔each page; every Landing dialog opens/closes; Training mode switch; Recording hotkey handlers attached; Calibration selects/dialogs; 404 fallback.
- [ ] **Step 2 (codex):** `codex exec -s read-only` with a self-contained prompt: "Diff `main...redesign` in this repo. Verify no functional changes: hooks/handlers/effects/API calls/WebSocket/keyboard/audio logic identical; only classNames, markup, imports, copy. List any behavioral drift with file:line." Also run `codex review` on the branch. Explicit timeout ≥ 20 min or background.
- [ ] **Step 3:** fix anything surfaced, re-verify, commit.
- [ ] **Step 4:** final `npm run build`; done — do **not** commit `frontend/dist` (CI rebuilds on main).
```
