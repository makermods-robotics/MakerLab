import * as React from "react";

import { cn } from "@/lib/utils";
import { Input } from "./input";

type NumberInputProps = Omit<
  React.ComponentProps<typeof Input>,
  "type" | "value" | "onChange"
> & {
  value: number | undefined | null;
  onChange: (value: number | undefined) => void;
  integer?: boolean;
};

const NumberInput = React.forwardRef<HTMLInputElement, NumberInputProps>(
  ({ value, onChange, integer = true, className, ...props }, ref) => {
    const [display, setDisplay] = React.useState(
      value == null ? "" : String(value)
    );
    // Track the last value we saw from props so we only resync the
    // visible string when the prop changes externally. This lets the
    // user clear the field even when the parent keeps the previous
    // numeric value (because our onChange(undefined) was ignored).
    const lastPropRef = React.useRef<number | undefined | null>(value);

    React.useEffect(() => {
      if (value !== lastPropRef.current) {
        lastPropRef.current = value;
        setDisplay(value == null ? "" : String(value));
      }
    }, [value]);

    return (
      <Input
        ref={ref}
        type="number"
        inputMode={integer ? "numeric" : "decimal"}
        value={display}
        onChange={(e) => {
          const next = e.target.value;
          setDisplay(next);
          if (next === "") {
            onChange(undefined);
            return;
          }
          const n = integer ? parseInt(next, 10) : parseFloat(next);
          if (Number.isFinite(n)) onChange(n);
        }}
        className={cn(
          "font-mono [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:m-0 [&::-webkit-outer-spin-button]:m-0",
          className
        )}
        {...props}
      />
    );
  }
);
NumberInput.displayName = "NumberInput";

export { NumberInput };
