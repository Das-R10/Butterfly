// frontend/src/components/common/ReportDownloadButton.tsx

import React, { useEffect, useState } from 'react';
import { FileText, Download, Loader2 } from 'lucide-react';
import { api } from '../../api/client';
import { useSimulation } from '../../context/SimulationContext';

interface ReportDownloadButtonProps {
  simulationId: string;
  initialHasReport?: boolean;
  initialFilename?: string | null;
  disabled?: boolean;
  disabledReason?: string;
  compact?: boolean;
  onError?: (msg: string) => void;
  onSuccess?: (info: { filename: string; cached: boolean }) => void;
}

export const ReportDownloadButton: React.FC<ReportDownloadButtonProps> = ({
  simulationId,
  initialHasReport,
  initialFilename,
  disabled = false,
  disabledReason,
  compact = false,
  onError,
  onSuccess,
}) => {
  const { markReportAvailable } = useSimulation();
  const [override, setOverride] = useState<{
    simId: string;
    hasReport: boolean;
    filename: string | null;
  } | null>(null);
  const [busy, setBusy] = useState<'idle' | 'generating' | 'downloading'>('idle');

  const hasReport =
    override?.simId === simulationId ? override.hasReport : Boolean(initialHasReport);
  const filename =
    override?.simId === simulationId ? override.filename : initialFilename || null;

  // If caller did not pass initialHasReport explicitly, query lightweight status endpoint
  useEffect(() => {
    if (initialHasReport !== undefined || !simulationId || disabled) return;
    let active = true;
    const controller = new AbortController();
    api
      .getReportStatus(simulationId, controller.signal)
      .then((res) => {
        if (!active) return;
        setOverride({
          simId: simulationId,
          hasReport: Boolean(res.report_exists),
          filename: res.filename || null,
        });
      })
      .catch(() => {
        // Silent ignore (e.g., transient or unpersisted session)
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [simulationId, initialHasReport, disabled]);

  const handleClick = async () => {
    if (disabled || busy !== 'idle' || !simulationId) return;
    try {
      let targetFilename = filename || undefined;
      let cached = hasReport;

      if (hasReport) {
        setBusy('downloading');
        try {
          const dl = await api.downloadReportPdf(simulationId, targetFilename);
          targetFilename = dl.filename;
        } catch {
          // If cached report missing on disk, transparently rebuild from saved simulation JSON
          setBusy('generating');
          const gen = await api.generateReport(simulationId);
          setOverride({
            simId: simulationId,
            hasReport: true,
            filename: gen.filename,
          });
          targetFilename = gen.filename;
          cached = gen.cached;
          markReportAvailable(simulationId, gen.filename, gen.generated_at);
          setBusy('downloading');
          await api.downloadReportPdf(simulationId, targetFilename);
        }
      } else {
        setBusy('generating');
        const gen = await api.generateReport(simulationId);
        setOverride({
          simId: simulationId,
          hasReport: true,
          filename: gen.filename,
        });
        targetFilename = gen.filename;
        cached = gen.cached;
        markReportAvailable(simulationId, gen.filename, gen.generated_at);
        setBusy('downloading');
        await api.downloadReportPdf(simulationId, targetFilename);
      }

      onSuccess?.({
        filename: targetFilename || `Butterfly_Research_Report_${simulationId}.pdf`,
        cached,
      });
    } catch (err: any) {
      onError?.(err?.message || `Failed to generate or download PDF for ${simulationId}`);
    } finally {
      setBusy('idle');
    }
  };

  const label =
    busy === 'generating'
      ? compact
        ? 'Generating...'
        : 'Generating PDF...'
      : busy === 'downloading'
      ? compact
        ? 'Downloading...'
        : 'Downloading PDF...'
      : hasReport
      ? 'Download PDF'
      : compact
      ? 'Generate PDF'
      : 'Generate PDF Report';

  const title = disabled
    ? disabledReason || 'PDF report unavailable for this simulation'
    : hasReport
    ? 'Download generated PDF research report (no inference rerun)'
    : 'Generate & download professional PDF research report from saved data';

  if (compact) {
    return (
      <button
        type="button"
        onClick={handleClick}
        disabled={disabled || busy !== 'idle'}
        title={title}
        className={`inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border font-medium transition text-[11px] disabled:opacity-50 disabled:cursor-not-allowed ${
          hasReport
            ? 'bg-emerald-500/15 hover:bg-emerald-500 hover:text-slate-950 text-emerald-300 border-emerald-500/35'
            : 'bg-violet-500/15 hover:bg-violet-500 hover:text-slate-950 text-violet-300 border-violet-500/35'
        }`}
      >
        {busy !== 'idle' ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        ) : hasReport ? (
          <Download className="w-3.5 h-3.5" />
        ) : (
          <FileText className="w-3.5 h-3.5" />
        )}
        <span>{label}</span>
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={disabled || busy !== 'idle'}
      title={title}
      className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-xl border text-xs font-semibold transition disabled:opacity-50 disabled:cursor-not-allowed ${
        hasReport
          ? 'bg-emerald-500/15 hover:bg-emerald-500 hover:text-slate-950 text-emerald-300 border-emerald-500/40'
          : 'bg-violet-500/15 hover:bg-violet-500 hover:text-slate-950 text-violet-300 border-violet-500/40'
      }`}
    >
      {busy !== 'idle' ? (
        <Loader2 className="w-4 h-4 animate-spin" />
      ) : hasReport ? (
        <Download className="w-4 h-4" />
      ) : (
        <FileText className="w-4 h-4" />
      )}
      <span>{label}</span>
    </button>
  );
};
