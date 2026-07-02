import React, { useState } from "react";
import { Check, Loader2, LogOut, ShieldAlert, UserPlus } from "lucide-react";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useToast } from "@/hooks/use-toast";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import HfAuthDialog from "./HfAuthDialog";

const HfAuthChip: React.FC = () => {
  const { auth, accounts, switchAccount, logout } = useHfAuth();
  const { toast } = useToast();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [confirmLogout, setConfirmLogout] = useState(false);
  const [busy, setBusy] = useState(false);

  const handleSwitch = async (name: string) => {
    if (name === accounts.active || busy) return;
    setBusy(true);
    try {
      await switchAccount(name);
      toast({ title: `Switched to ${name}` });
    } catch (e) {
      toast({
        title: "Couldn't switch account",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setBusy(false);
    }
  };

  const handleLogout = async () => {
    setBusy(true);
    try {
      await logout();
      toast({ title: "Logged out" });
    } catch (e) {
      toast({
        title: "Couldn't log out",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setBusy(false);
      setConfirmLogout(false);
    }
  };

  if (auth.status === "loading") {
    return (
      <div className="inline-flex items-center gap-2 rounded-full border border-gray-800 bg-gray-900/60 px-3 py-1 text-xs text-gray-400">
        <Loader2 className="w-3 h-3 animate-spin" />
        <span>Checking HF…</span>
      </div>
    );
  }

  if (auth.status === "authenticated") {
    const envToken = auth.envToken;
    return (
      <>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-full border border-gray-800 bg-gray-900/60 px-3 py-1 text-xs text-gray-200 hover:bg-gray-800/60 transition-colors disabled:opacity-60"
              disabled={busy}
              title={
                envToken
                  ? "Identity pinned by the HF_TOKEN environment variable"
                  : "Hugging Face account"
              }
            >
              {busy ? (
                <Loader2 className="w-3 h-3 animate-spin" aria-hidden="true" />
              ) : (
                <span
                  className="h-2 w-2 rounded-full bg-emerald-400"
                  aria-hidden="true"
                />
              )}
              <span>{auth.username}</span>
              {envToken && (
                <span className="rounded bg-sky-900/60 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-sky-200">
                  env token
                </span>
              )}
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            className="w-64 border-gray-800 bg-gray-900 text-gray-200"
          >
            <DropdownMenuLabel className="text-gray-400">
              Signed in as{" "}
              <span className="text-gray-100">{auth.username}</span>
            </DropdownMenuLabel>
            <DropdownMenuSeparator className="bg-gray-800" />

            {envToken ? (
              <div className="px-2 py-1.5 text-xs text-gray-400">
                <div className="flex items-start gap-2">
                  <ShieldAlert className="mt-0.5 h-4 w-4 flex-shrink-0 text-sky-300" />
                  <p>
                    Identity is pinned by the{" "}
                    <span className="font-mono text-gray-300">HF_TOKEN</span>{" "}
                    environment variable. Unset it and restart the server to
                    manage accounts here.
                  </p>
                </div>
              </div>
            ) : (
              <>
                {accounts.accounts.map((name) => (
                  <DropdownMenuItem
                    key={name}
                    onClick={() => handleSwitch(name)}
                    disabled={busy}
                    className="cursor-pointer focus:bg-gray-800"
                  >
                    <Check
                      className={`mr-2 h-4 w-4 ${
                        name === accounts.active
                          ? "text-emerald-400"
                          : "text-transparent"
                      }`}
                    />
                    <span className="truncate">{name}</span>
                  </DropdownMenuItem>
                ))}
                <DropdownMenuSeparator className="bg-gray-800" />
                <DropdownMenuItem
                  onClick={() => setDialogOpen(true)}
                  className="cursor-pointer focus:bg-gray-800"
                >
                  <UserPlus className="mr-2 h-4 w-4" />
                  Add account…
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => setConfirmLogout(true)}
                  className="cursor-pointer text-red-300 focus:bg-gray-800 focus:text-red-200"
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Log out
                </DropdownMenuItem>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>

        <HfAuthDialog open={dialogOpen} onOpenChange={setDialogOpen} />

        <AlertDialog open={confirmLogout} onOpenChange={setConfirmLogout}>
          <AlertDialogContent className="border-gray-800 bg-gray-900 text-white">
            <AlertDialogHeader>
              <AlertDialogTitle>Log out of {auth.username}?</AlertDialogTitle>
              <AlertDialogDescription className="text-gray-400">
                This removes the token from the machine-global Hugging Face
                token store, which is shared with the{" "}
                <span className="font-mono">hf</span> CLI and any other tool on
                this machine.
                {accounts.accounts.length > 1
                  ? " You'll be switched to another stored account."
                  : ""}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel
                disabled={busy}
                className="border-gray-700 bg-transparent text-gray-200 hover:bg-gray-800 hover:text-white"
              >
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                onClick={(e) => {
                  e.preventDefault();
                  handleLogout();
                }}
                disabled={busy}
                className="bg-red-600 text-white hover:bg-red-700"
              >
                {busy ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Logging out…
                  </>
                ) : (
                  "Log out"
                )}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </>
    );
  }

  // unauthenticated
  return (
    <>
      <button
        type="button"
        onClick={() => setDialogOpen(true)}
        className="inline-flex items-center gap-2 rounded-full border border-amber-700/60 bg-amber-950/40 px-3 py-1 text-xs text-amber-100 hover:bg-amber-900/40 transition-colors"
        aria-label="Hugging Face not configured — add an account"
      >
        <span
          className="h-2 w-2 rounded-full bg-amber-400"
          aria-hidden="true"
        />
        <span>HF not configured</span>
      </button>
      <HfAuthDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </>
  );
};

export default HfAuthChip;
