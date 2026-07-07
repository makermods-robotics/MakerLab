import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import Logo from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/shell/ThemeToggle";
import HfAuthChip from "@/components/landing/HfAuthChip";
import { cn } from "@/lib/utils";

interface AppShellProps {
  back?: { to?: string; onClick?: () => void; label?: string };
  status?: React.ReactNode;
  actions?: React.ReactNode;
  showAuthChip?: boolean;
  fullBleed?: boolean;
  /** Set false on live-session screens so the logo can't end a session with a stray click. */
  logoLink?: boolean;
  children: React.ReactNode;
}

export function AppShell({
  back,
  status,
  actions,
  showAuthChip = true,
  fullBleed = false,
  logoLink = true,
  children,
}: AppShellProps) {
  const navigate = useNavigate();
  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur-[8px] backdrop-saturate-150">
        <div className="mx-auto flex h-[52px] max-w-[1440px] items-center gap-3 px-4">
          {logoLink ? (
            <Link to="/" aria-label="Home">
              <Logo />
            </Link>
          ) : (
            <Logo />
          )}
          {back && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                back.onClick
                  ? back.onClick()
                  : back.to
                    ? navigate(back.to)
                    : navigate(-1)
              }
            >
              <ArrowLeft /> {back.label ?? "back"}
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
      <main
        className={cn(
          "flex-1",
          !fullBleed && "mx-auto w-full max-w-[1440px] px-4 py-6"
        )}
      >
        {children}
      </main>
    </div>
  );
}
