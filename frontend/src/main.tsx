import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App'
import AuthGate from './features/auth/AuthGate'
import SignInPage from './features/auth/SignInPage'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import BillingReturn from './pages/BillingReturn'
import DesignGallery from './pages/DesignGallery'
import CasesDashboard from './features/cases/CasesDashboard'
import CaseWorkspace from './features/cases/CaseWorkspace'
import AnalysisWorkspace from './features/analysis/AnalysisWorkspace'
import SectionRoute from './features/analysis/SectionRoute'
import EvidenceViewer from './features/viewer/EvidenceViewer'
import BatchPage from './features/batches/BatchPage'
import { DemoProvider } from './demo/DemoProvider'
import DemoOverlay from './demo/DemoOverlay'
import NotFound from './components/NotFound'
import { makeQueryClient } from './lib/api/queryClient'

const qc = makeQueryClient()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <DemoProvider>
          <Routes>
            {/* ARCHITECTURE §7.2: the cases dashboard is the home; the legacy analyses list lives at /analyses. */}
            {/* L-6: without working credentials the workspace is replaced by the sign-in form. */}
            <Route element={<AuthGate><App /></AuthGate>}>
              <Route index element={<CasesDashboard />} />
              <Route path="cases/:caseId" element={<CaseWorkspace />} />
              <Route path="analyses" element={<Dashboard />} />
              <Route path="upload" element={<Upload />} />
              <Route path="billing/return" element={<BillingReturn />} />
              {/* ARCHITECTURE §7.2: upload into a case, with the acquisition fields open (UX §5.1). */}
              <Route path="cases/:caseId/upload" element={<Upload />} />
              {/* Batch progress (ARCHITECTURE §7.2 `/batches/:id`, PRD API-01). */}
              <Route path="batches/:batchId" element={<BatchPage />} />
              {/* Evidence viewer (UX §5.9, ARCHITECTURE §7.2 `evidence`): a full-viewport route with its view state in the URL. */}
              <Route path="analyses/:id/evidence" element={<EvidenceViewer />} />
              {/* One scrolling document (UX §2.3); the nested routes deep-link to its sections. */}
              <Route path="analyses/:id" element={<AnalysisWorkspace />}>
                <Route index element={<SectionRoute section="summary" />} />
                <Route path="findings" element={<SectionRoute section="findings" />} />
                <Route path="image-evidence" element={<SectionRoute section="image-evidence" />} />
                <Route path="media-evidence" element={<SectionRoute section="media-evidence" />} />
                <Route path="exhibit" element={<SectionRoute section="exhibit" />} />
                <Route path="reached" element={<SectionRoute section="reached" />} />
                <Route path="ledger" element={<SectionRoute section="ledger" />} />
                <Route path="provenance" element={<SectionRoute section="provenance" />} />
                <Route path="methods" element={<SectionRoute section="methods" />} />
                <Route path="appendix" element={<SectionRoute section="appendix" />} />
                <Route path="revision" element={<SectionRoute section="revision" />} />
                <Route path="review" element={<Navigate to="../?view=plain" replace />} />
              </Route>
              {/* Preserve old bookmarks without exposing the retired catalogue. */}
              <Route path="models" element={<Navigate to="/" replace />} />
              <Route path="models/:detectorId" element={<Navigate to="/" replace />} />
              {/* Deep link: /demo starts the guided walkthrough (the provider consumes and clears `?demo=1`). */}
              <Route path="demo" element={<Navigate to="/analyses?demo=1" replace />} />
              {/* L-7: an address that matches no page says so inside the workspace shell instead of rendering nothing. */}
              <Route path="*" element={<NotFound />} />
            </Route>
            {/* N-9b: the sign-in form for a browser the instance lets in without an account (single-user mode, a stored key). */}
            <Route path="sign-in" element={<SignInPage />} />
            {/* Design contract gallery (E-0): standalone, renders without the engine. N-19a: on an instance that needs
                credentials it opens for a signed-in person only, as every other page does. */}
            <Route path="design" element={<AuthGate><DesignGallery /></AuthGate>} />
          </Routes>
          <DemoOverlay />
        </DemoProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)
