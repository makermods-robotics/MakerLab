
import React from 'react';
import { Mic, MicOff, Send, Camera } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

interface CommandBarProps {
  command: string;
  setCommand: (command: string) => void;
  handleSendCommand: () => void;
  isVoiceActive: boolean;
  setIsVoiceActive: (isActive: boolean) => void;
  showCamera: boolean;
  setShowCamera: (show: boolean) => void;
  handleEndSession: () => void;
}

const CommandBar: React.FC<CommandBarProps> = ({
  command,
  setCommand,
  handleSendCommand,
  isVoiceActive,
  setIsVoiceActive,
  showCamera,
  setShowCamera,
  handleEndSession
}) => {
  return (
    <div className="space-y-4 border-t border-border bg-card p-4">
      <div className="mx-auto flex w-full max-w-4xl flex-col items-center gap-4 sm:flex-row">
        <Input
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          placeholder="Tell the robot what to do..."
          className="flex-1 py-3 font-mono text-lg"
          onKeyPress={(e) => e.key === 'Enter' && handleSendCommand()}
        />
        <Button
          onClick={handleSendCommand}
          className="self-stretch px-6 py-3 sm:self-auto"
        >
          <Send strokeWidth={1.5} />
          Send
        </Button>
      </div>

      <div className="flex items-center justify-center gap-6">
        <div className="flex flex-wrap justify-center gap-2 sm:gap-4">
          <Button
            onClick={() => setIsVoiceActive(!isVoiceActive)}
            variant={isVoiceActive ? "default" : "secondary"}
            className="px-6 py-2"
          >
            {isVoiceActive ? <Mic strokeWidth={1.5} /> : <MicOff strokeWidth={1.5} />}
            Voice command
          </Button>

          <Button
            onClick={() => setShowCamera(!showCamera)}
            variant={showCamera ? "default" : "secondary"}
            className="px-6 py-2"
          >
            <Camera strokeWidth={1.5} />
            Show camera
          </Button>

          <Button
            onClick={handleEndSession}
            variant="destructive"
            className="px-6 py-2"
          >
            End session
          </Button>
        </div>
      </div>
    </div>
  );
};

export default CommandBar;
