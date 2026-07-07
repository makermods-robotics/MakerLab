
import React from 'react';
import { cn } from '@/lib/utils';

interface LogoProps extends React.HTMLAttributes<HTMLDivElement> {
  iconOnly?: boolean;
}

const Logo: React.FC<LogoProps> = ({
  className,
  iconOnly = false
}) => {
  return <div className={cn("flex items-center gap-2.5", className)}>
      <img
        src="/makermods/logo-mark-white.png"
        alt="MakerLab Logo"
        className="h-6 w-auto brightness-0 dark:brightness-100"
      />
      {!iconOnly && <span className="font-display text-[15px] font-bold tracking-[0.06em] text-foreground">MAKERLAB</span>}
    </div>;
};

export default Logo;
