
import React from 'react';
import { cn } from '@/lib/utils';
import logoMark from '@/assets/logo-mark.png';

interface LogoProps extends React.HTMLAttributes<HTMLDivElement> {
  iconOnly?: boolean;
}

const Logo: React.FC<LogoProps> = ({
  className,
  iconOnly = false
}) => {
  return <div className={cn("flex items-center gap-2", className)}>
      <img src={logoMark} alt="MakerMods" className="h-7 w-auto" />
      {!iconOnly && <span className="font-bold text-foreground text-2xl">MakerLab</span>}
    </div>;
};

export default Logo;
