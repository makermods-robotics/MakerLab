import React, { useRef, useState } from "react";
import { Plus, Check, ChevronsUpDown } from "lucide-react";
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
  const inputRef = useRef<HTMLInputElement>(null);

  const trimmed = query.trim();
  const matchesExisting = availableNames.some(
    (n) => n.toLowerCase() === trimmed.toLowerCase()
  );
  const canCreate = trimmed.length > 0 && !matchesExisting;

  // Only a name collision disables the row; with an empty query it stays
  // clickable and focuses the input, so the "type a name first" step is
  // discoverable instead of a dead grayed-out button.
  const createDisabled = matchesExisting;
  const createLabel = matchesExisting
    ? "Already exists"
    : trimmed === ""
      ? "Type a name to create a new robot…"
      : `Create "${trimmed}"`;

  const reset = () => {
    setQuery("");
    setOpen(false);
  };

  const handlePickExisting = (name: string) => {
    onSelect(name);
    reset();
  };

  const handleCreate = async () => {
    if (trimmed === "") {
      inputRef.current?.focus();
      return;
    }
    if (!canCreate) return;
    const ok = await onCreateNew(trimmed);
    if (ok) reset();
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={isLoading}
          className="w-full justify-between bg-gray-900 border-gray-700 text-white hover:bg-gray-700 hover:text-white font-normal"
        >
          <span className={cn("truncate", selectedName ? "" : "text-gray-400")}>
            {isLoading
              ? "Loading..."
              : selectedName ?? "Select a robot or type a new name"}
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
            ref={inputRef}
            placeholder="Search or type new name..."
            value={query}
            onValueChange={setQuery}
            onKeyDown={(e) => {
              if (e.key === "Enter" && canCreate) {
                e.preventDefault();
                handleCreate();
              }
            }}
            className="text-white"
          />
          <CommandList>
            {availableNames.length === 0 && (
              <CommandEmpty className="py-4 text-sm text-gray-400 text-center">
                No robots yet. Type a name to create one.
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
            onClick={handleCreate}
            disabled={createDisabled}
            className="flex w-full items-center gap-2 border-t border-gray-700 px-3 py-2 text-sm text-white hover:bg-gray-700 disabled:cursor-not-allowed disabled:text-gray-500 disabled:hover:bg-transparent"
          >
            <Plus className="h-4 w-4" />
            {createLabel}
          </button>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default RobotSelector;
