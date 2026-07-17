import { createRoot } from 'react-dom/client'
// Orbitron Black — the studio overlay's "by MakerMods" chip. Self-hosted so
// the UI renders identically offline.
import '@fontsource/orbitron/900.css'
import App from './App.tsx'
import './index.css'

createRoot(document.getElementById("root")!).render(<App />);
