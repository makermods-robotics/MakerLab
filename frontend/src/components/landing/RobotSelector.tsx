import React, { useState } from "react";
import { Plus, Check, ChevronsUpDown, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

interface RobotSelectorProps {
  selectedName: string | null;
  availableNames: string[];
  onSelect: (name: string) => void;
  onCreateNew: (name: string) => Promise<boolean>;
  isLoading: boolean;
}

const RobotSelector: React.FC<RobotSelectorProps> = ({
  selectedName,
  availableNames,
  onSelect,
  onCreateNew,
  isLoading,
}) => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  const handlePickExisting = (name: string) => {
    onSelect(name);
    setQuery("");
    setOpen(false);
  };

  const nameExists = (name: string) =>
    availableNames.some((n) => n.toLowerCase() === name.toLowerCase());

  const openCreateDialog = () => {
    // Continuity: if a fresh name is already typed in the search box,
    // carry it into the dialog.
    const seed = query.trim();
    setNewName(seed !== "" && !nameExists(seed) ? seed : "");
    setOpen(false);
    setCreateOpen(true);
  };

  const trimmedNewName = newName.trim();
  const newNameExists = trimmedNewName !== "" && nameExists(trimmedNewName);
  const canConfirm = trimmedNewName !== "" && !newNameExists && !creating;

  const handleCreateConfirm = async () => {
    if (!canConfirm) return;
    setCreating(true);
    try {
      // useRobots handles validation, API errors, and toasts; on success it
      // also selects the new robot. We only manage the dialog here.
      const ok = await onCreateNew(trimmedNewName);
      if (ok) {
        setCreateOpen(false);
        setNewName("");
        setQuery("");
      }
    } finally {
      setCreating(false);
    }
  };

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            disabled={isLoading}
            className="w-full justify-between bg-gray-900 border-gray-700 text-white hover:bg-gray-700 hover:text-white font-normal"
          >
            <span
              className={cn("truncate", selectedName ? "" : "text-gray-400")}
            >
              {isLoading
                ? "Loading..."
                : selectedName ?? "Select a robot or create a new one"}
            </span>
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent
          className="p-0 bg-gray-800 border-gray-700 text-white"
          style={{ width: "var(--radix-popover-trigger-width)" }}
          align="start"
        >
          <Command className="bg-gray-800">
            <CommandInput
              placeholder="Search robots..."
              value={query}
              onValueChange={setQuery}
              className="text-white"
            />
            <CommandList>
              {availableNames.length === 0 && (
                <CommandEmpty className="py-4 text-sm text-gray-400 text-center">
                  No robots yet.
                </CommandEmpty>
              )}
              {availableNames.length > 0 && (
                <CommandGroup heading="Existing">
                  {availableNames.map((name) => (
                    <CommandItem
                      key={name}
                      value={name}
                      onSelect={() => handlePickExisting(name)}
                      className="text-white aria-selected:bg-gray-700"
                    >
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          selectedName === name ? "opacity-100" : "opacity-0"
                        )}
                      />
                      {name}
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}
            </CommandList>
            <button
              type="button"
              onClick={openCreateDialog}
              className="flex w-full items-center gap-2 border-t border-gray-700 px-3 py-2 text-sm text-white hover:bg-gray-700"
            >
              <Plus className="h-4 w-4" />
              Create new robot…
            </button>
          </Command>
        </PopoverContent>
      </Popover>

      <Dialog
        open={createOpen}
        onOpenChange={(o) => {
          setCreateOpen(o);
          if (!o) setNewName("");
        }}
      >
        <DialogContent className="bg-gray-800 border-gray-700 text-white sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-white">Create a new robot</DialogTitle>
            <DialogDescription className="text-gray-400">
              Choose a name for the new robot.
            </DialogDescription>
          </DialogHeader>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleCreateConfirm();
            }}
            className="space-y-4"
          >
            <div>
              <Label htmlFor="new-robot-name" className="text-gray-300">
                Name
              </Label>
              <Input
                id="new-robot-name"
                autoFocus
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="my_robot"
                aria-invalid={newNameExists}
                className="mt-1 bg-gray-900 border-gray-600 text-white aria-[invalid=true]:border-red-500/70"
              />
              {newNameExists && (
                <p className="mt-1 text-xs text-red-400">
                  A robot with this name already exists.
                </p>
              )}
            </div>
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setCreateOpen(false)}
                className="bg-transparent border-gray-600 text-white hover:bg-gray-700 hover:text-white"
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={!canConfirm}
                className="bg-green-500 hover:bg-green-600 text-white"
              >
                {creating ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Creating…
                  </>
                ) : (
                  <>
                    <Plus className="w-4 h-4 mr-2" /> Create
                  </>
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default RobotSelector;
