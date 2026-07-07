
import React, { useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Camera, MicOff } from 'lucide-react';
import { Card } from '@/components/ui/card';

interface MetricsPanelProps {
  activeTab: 'SENSORS' | 'MOTORS';
  setActiveTab: (tab: 'SENSORS' | 'MOTORS') => void;
  sensorData: any[];
  motorData: any[];
  hasPermissions: boolean;
  streamRef: React.RefObject<MediaStream | null>;
  isVoiceActive: boolean;
  micLevel: number;
}

const MetricsPanel: React.FC<MetricsPanelProps> = ({
  activeTab,
  setActiveTab,
  sensorData,
  motorData,
  hasPermissions,
  streamRef,
  isVoiceActive,
  micLevel,
}) => {
  const sensorVideoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (activeTab === 'SENSORS' && hasPermissions && sensorVideoRef.current && streamRef.current) {
      if (sensorVideoRef.current.srcObject !== streamRef.current) {
        sensorVideoRef.current.srcObject = streamRef.current;
      }
    }
  }, [activeTab, hasPermissions, streamRef]);

  return (
    <div className="w-full p-2 sm:p-4 lg:w-1/2">
      <Card variant="flat" className="flex h-full flex-col p-4">
        {/* Tab Headers */}
        <div className="flex mb-4">
          <button
            onClick={() => setActiveTab('MOTORS')}
            className={`rounded-t-md px-6 py-2 font-display text-sm font-semibold sm:text-base ${
              activeTab === 'MOTORS'
                ? 'bg-primary text-primary-foreground'
                : 'bg-card text-muted-foreground hover:bg-accent'
            }`}
          >
            Motors
          </button>
          <button
            onClick={() => setActiveTab('SENSORS')}
            className={`ml-2 rounded-t-md px-6 py-2 font-display text-sm font-semibold sm:text-base ${
              activeTab === 'SENSORS'
                ? 'bg-primary text-primary-foreground'
                : 'bg-card text-muted-foreground hover:bg-accent'
            }`}
          >
            Sensors
          </button>
        </div>

        {/* Chart Content */}
        <div className="flex-1 overflow-y-auto">
          {activeTab === 'SENSORS' && (
            <div className="space-y-4">
              {/* Webcam Feed */}
              <div className="flex h-64 flex-col rounded-md border border-border p-2">
                <h3 className="mb-2 text-sm font-medium">Live camera feed</h3>
                {hasPermissions ? (
                  <div className="flex-1 overflow-hidden rounded bg-muted">
                    <video
                      ref={sensorVideoRef}
                      autoPlay
                      muted
                      playsInline
                      className="h-full w-full object-contain"
                    />
                  </div>
                ) : (
                  <div className="flex flex-1 items-center justify-center rounded bg-muted">
                    <div className="text-center">
                      <Camera className="mx-auto mb-2 h-12 w-12 text-muted-foreground" />
                      <p className="text-muted-foreground">Camera permission not granted.</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Mic Detection & Other Sensors */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="flex min-h-[120px] flex-col justify-center rounded-md border border-border p-2">
                    <h3 className="mb-2 text-center text-sm font-medium">Voice activity</h3>
                  {hasPermissions ? (
                    <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
                      <div className="flex h-10 w-full items-end justify-center gap-px">
                        {[...Array(15)].map((_, i) => {
                          const barIsActive = isVoiceActive && i < (micLevel / 120 * 15);
                          return (
                            <div
                              key={i}
                              className={`w-1.5 rounded-full transition-colors duration-75 ${barIsActive ? 'bg-info' : 'bg-muted'}`}
                              style={{ height: `${(i / 15 * 60) + 20}%` }}
                            />
                          );
                        })}
                      </div>
                      <p className="font-mono text-xs text-muted-foreground">
                        {isVoiceActive ? "Voice commands active" : "Voice commands muted"}
                      </p>
                    </div>
                  ) : (
                    <div className="flex flex-1 items-center justify-center rounded bg-muted">
                      <div className="text-center">
                        <MicOff className="mx-auto mb-2 h-8 w-8 text-muted-foreground" />
                        <p className="text-muted-foreground">Microphone permission not granted.</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Sensor Charts */}
                {['sensor3', 'sensor4'].map((sensor, index) => (
                  <div key={sensor} className="flex h-auto min-h-[120px] flex-col rounded-md border border-border p-2">
                    <h3 className="mb-2 text-sm font-medium">Sensor {index + 3}</h3>
                    <ResponsiveContainer width="100%" height="90%">
                      <LineChart data={sensorData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis hide />
                        <YAxis fontSize={12} stroke="hsl(var(--muted-foreground))" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            color: 'hsl(var(--foreground))'
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey={sensor}
                          stroke={index % 2 === 1 ? 'hsl(var(--info))' : 'hsl(var(--ok))'}
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'MOTORS' && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {['motor1', 'motor2', 'motor3', 'motor4', 'motor5', 'motor6'].map((motor, index) => (
                <div key={motor} className="h-40 rounded-md border border-border p-2">
                  <h3 className="mb-2 text-sm font-medium">Motor {index + 1}</h3>
                  <ResponsiveContainer width="100%" height="80%">
                    <LineChart data={motorData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis hide />
                      <YAxis fontSize={12} stroke="hsl(var(--muted-foreground))" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'hsl(var(--card))',
                          border: '1px solid hsl(var(--border))',
                          color: 'hsl(var(--foreground))'
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey={motor}
                        stroke={index % 2 === 0 ? 'hsl(var(--info))' : 'hsl(var(--ok))'}
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default MetricsPanel;
