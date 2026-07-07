import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import Logo from '@/components/Logo';

const TrainingHeader: React.FC = () => {
  const navigate = useNavigate();
  return (
    <div className="mb-8 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" onClick={() => navigate('/')}>
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <Logo />
        <h1 className="font-display text-2xl font-bold text-foreground">
          Training
        </h1>
      </div>
    </div>
  );
};

export default TrainingHeader;
