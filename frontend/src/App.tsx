// frontend/src/App.tsx

import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { SimulationProvider } from './context/SimulationContext';
import { AppLayout } from './components/layout/AppLayout';

import { Overview } from './pages/Overview';
import { NewSimulation } from './pages/NewSimulation';
import { LiveSimulation } from './pages/LiveSimulation';
import { ForecastVerdict } from './pages/ForecastVerdict';
import { NarrativeAnalysis } from './pages/NarrativeAnalysis';
import { InfluenceNetwork } from './pages/InfluenceNetwork';
import { MacroTrajectory } from './pages/MacroTrajectory';
import { SimulationHistory } from './pages/SimulationHistory';
import { ExperimentLab } from './pages/ExperimentLab';
import { Settings } from './pages/Settings';

export function App() {
  return (
    <SimulationProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Overview />} />
            <Route path="/new" element={<NewSimulation />} />
            <Route path="/live" element={<LiveSimulation />} />
            <Route path="/forecast" element={<ForecastVerdict />} />
            <Route path="/narrative" element={<NarrativeAnalysis />} />
            <Route path="/network" element={<InfluenceNetwork />} />
            <Route path="/trajectory" element={<MacroTrajectory />} />
            <Route path="/history" element={<SimulationHistory />} />
            <Route path="/lab" element={<ExperimentLab />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </SimulationProvider>
  );
}

export default App;
