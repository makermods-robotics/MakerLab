import React, { useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
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
import { RobotMode } from "@/hooks/useRobots";
import { cn } from "@/lib/utils";

interface RobotSelectorProps {
  selectedName: string | null;
  availableNames: string[];
  // Layout used only to label the filtered empty state.
  defaultMode: RobotMode;
  onSelect: (name: string) => void;
  isLoading: boolean;
}

const MODE_LABEL: Record<RobotMode, string> = {
  single: "single arm",
  bimanual: "bimanual",
};

const RobotSelector: React.FC<RobotSelectorProps> = ({
  selectedName,
  availableNames,
  defaultMode,
  onSelect,
  isLoading,
}) => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const handlePickExisting = (name: string) => {
    onSelect(name);
    setQuery("");
    setOpen(false);
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
                No {MODE_LABEL[defaultMode]} robots yet — use “New robot”.
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
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default RobotSelector;
