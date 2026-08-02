import React, { useState } from "react";
import {
  Check,
  Laptop,
  Loader2,
  Plus,
  RefreshCw,
  Server,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useApi } from "@/contexts/ApiContext";
import type { HostReachability } from "@/lib/apiHosts";
import { cn } from "@/lib/utils";

/** Reachability dot, same visual grammar as RobotCorner's StatusDot. */
const HostDot: React.FC<{ status: HostReachability; className?: string }> = ({
  status,
  className,
}) => (
  <span
    aria-hidden
    className={cn(
      "inline-block h-2 w-2 shrink-0 rounded-full",
      status === "online"
        ? "bg-ok"
        : status === "offline"
          ? "bg-destructive"
          : "border border-muted-foreground/60 bg-transparent",
      className,
    )}
  />
);

const statusWord = (status: HostReachability): string =>
  status === "online"
    ? "reachable"
    : status === "offline"
      ? "not answering"
      : "not checked yet";

const lastSeenText = (lastSeen: number | null): string | null => {
  if (!lastSeen) return null;
  const mins = Math.round((Date.now() - lastSeen) / 60000);
  if (mins < 1) return "seen just now";
  if (mins < 60) return `seen ${mins} min ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `seen ${hours} h ago`;
  return `seen ${Math.round(hours / 24)} d ago`;
};

/**
 * Which computer the GUI is driving — the first segment of the robot corner.
 *
 * MakerLab can run split across two machines: a headless server owning the
 * follower arm + cameras, and this laptop owning the leader arm. Everything in
 * the app (robots, datasets, training, inference) talks to the host selected
 * here; only the leader-bridge is pinned to this machine.
 *
 * With no remote added — the default — this collapses to an icon-only button,
 * so a single-machine install keeps the header it always had.
 */
const HostSwitcher: React.FC = () => {
  const {
    hosts,
    activeHost,
    isRemote,
    setActiveHostUrl,
    addHost,
    removeHost,
    probeAllHosts,
  } = useApi();

  const [menuOpen, setMenuOpen] = useState(false);
  const [addOpen, setAddOpen] = useState(false);
  const [name, setName] = useState("");
  const [address, setAddress] = useState("");
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Set after a failed reachability check so a booting/sleeping server can
  // still be saved on a second, deliberate click.
  const [allowForce, setAllowForce] = useState(false);

  const hasRemotes = hosts.some((h) => !h.isLocal);
  // Only surface the label when there is something to disambiguate; otherwise
  // the corner reads exactly as it did before remote hosts existed.
  const showLabel = hasRemotes || isRemote;

  const handleMenuOpenChange = (open: boolean) => {
    setMenuOpen(open);
    if (open) probeAllHosts();
  };

  const resetAddForm = () => {
    setName("");
    setAddress("");
    setError(null);
    setAllowForce(false);
  };

  const submitAdd = async (force: boolean) => {
    setChecking(true);
    setError(null);
    try {
      const result = await addHost(name, address, force);
      if (result.ok) {
        setAddOpen(false);
        resetAddForm();
      } else {
        setError(result.message ?? "Couldn't add that computer.");
        // A malformed address is not worth forcing; an unanswered one is.
        setAllowForce(!!result.message?.startsWith("No answer"));
      }
    } finally {
      setChecking(false);
    }
  };

  return (
    <>
      <DropdownMenu open={menuOpen} onOpenChange={handleMenuOpenChange}>
        <Tooltip>
          <TooltipTrigger asChild>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                aria-label="Computer"
                className={cn(
                  "h-7 shrink-0 rounded-full",
                  showLabel ? "gap-2 px-2.5 font-medium" : "w-7 p-0",
                )}
              >
                {isRemote ? (
                  <Server className="h-3.5 w-3.5" />
                ) : (
                  <Laptop className="h-3.5 w-3.5" />
                )}
                {showLabel && (
                  <>
                    <span className="max-w-[140px] truncate">
                      {activeHost.name}
                    </span>
                    <HostDot status={activeHost.status} />
                  </>
                )}
              </Button>
            </DropdownMenuTrigger>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            {isRemote
              ? `Robot work runs on ${activeHost.name} (${statusWord(activeHost.status)})`
              : "Running on this computer — add a remote computer"}
          </TooltipContent>
        </Tooltip>

        <DropdownMenuContent align="end" className="w-80">
          <DropdownMenuLabel className="eyebrow">Computer</DropdownMenuLabel>
          {hosts.map((h) => {
            const active = h.url === activeHost.url;
            const seen = lastSeenText(h.lastSeen);
            return (
              <DropdownMenuItem
                key={h.url}
                onSelect={() => setActiveHostUrl(h.url)}
                className={cn("items-start gap-2", active && "bg-accent")}
              >
                <HostDot status={h.status} className="mt-1.5" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate">{h.name}</span>
                  <span className="block truncate font-mono text-[10px] text-muted-foreground">
                    {h.url}
                  </span>
                  <span className="block text-[10px] text-muted-foreground">
                    {statusWord(h.status)}
                    {seen ? ` · ${seen}` : ""}
                  </span>
                </span>
                {active && <Check className="mt-1 h-3.5 w-3.5 shrink-0" />}
                {!h.isLocal && (
                  <button
                    type="button"
                    aria-label={`Forget ${h.name}`}
                    title={`Forget ${h.name}`}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      removeHost(h.url);
                    }}
                    className="mt-0.5 shrink-0 rounded p-1 text-muted-foreground hover:text-destructive"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
              </DropdownMenuItem>
            );
          })}
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onSelect={() => {
              resetAddForm();
              setAddOpen(true);
            }}
            className="gap-2"
          >
            <Plus className="h-4 w-4" />
            Add remote computer…
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={probeAllHosts} className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Check again
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog
        open={addOpen}
        onOpenChange={(next) => {
          setAddOpen(next);
          if (!next) resetAddForm();
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Add remote computer</DialogTitle>
            <DialogDescription>
              The machine running <span className="font-mono">makerlab</span>{" "}
              with the follower arm and cameras attached. Start it there with{" "}
              <span className="font-mono">--lan</span> so it accepts connections
              from this laptop, then paste the address it prints.
            </DialogDescription>
          </DialogHeader>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              submitAdd(false);
            }}
            className="space-y-4"
          >
            <div>
              <Label htmlFor="host-address">Address</Label>
              <Input
                id="host-address"
                autoFocus
                placeholder="192.168.1.20:8000"
                value={address}
                onChange={(e) => {
                  setAddress(e.target.value);
                  setError(null);
                  setAllowForce(false);
                }}
                className="mt-1 font-mono text-sm"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                An IP or hostname. Port 8000 is assumed if you leave it off.
              </p>
            </div>
            <div>
              <Label htmlFor="host-name">Name</Label>
              <Input
                id="host-name"
                placeholder="Mac Mini"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Optional — shown in the corner so you can tell the machines
                apart.
              </p>
            </div>
            {error && (
              <p className="rounded-md border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
                {error}
              </p>
            )}
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setAddOpen(false)}
              >
                Cancel
              </Button>
              {allowForce && (
                <Button
                  type="button"
                  variant="secondary"
                  disabled={checking}
                  onClick={() => submitAdd(true)}
                  title="Save it anyway — useful when the server is still booting"
                >
                  Add anyway
                </Button>
              )}
              <Button type="submit" disabled={checking || !address.trim()}>
                {checking ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Checking…
                  </>
                ) : (
                  "Check & add"
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default HostSwitcher;
