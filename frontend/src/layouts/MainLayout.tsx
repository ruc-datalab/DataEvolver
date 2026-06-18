import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { LiveArtifactsForCanvas } from '../lib/buildEvolutionRowsFromPipeline'
import { buildEvolutionRowsFromPipeline } from '../lib/buildEvolutionRowsFromPipeline'
import type { ReactNode } from 'react'
import {
  Check,
  ChevronRight,
  Database,
  Eye,
  FileJson2,
  FileText,
  Image,
  Languages,
  Loader2,
  Play,
  RotateCcw,
  Save,
  Settings,
  Sparkles,
  Trash2,
  Upload,
  UploadCloud,
  X,
  Zap,
} from 'lucide-react'
import { useAppStore } from '../stores/appStore'
import { EvolutionCanvas } from '../components/EvolutionCanvas'
import {
  advanceWorkflow,
  ApiError,
  fetchArtifactHistory,
  fetchExperience,
  fetchInstantiationSteps,
  fetchLlmConfigFromServer,
  fetchOperators,
  fetchOrchestrationDag,
  fetchPipelineRunLatest,
  fetchPipelinePreview,
  fetchQualityCheck,
  fetchTrialResult,
  fetchUnderstandingResult,
  fetchWorkflowTokens,
  fetchWorkflowState,
  mapApiOperatorsToPoolItems,
  parseFastApiErrorBody,
  resetWorkflowForDebug,
  resetWorkflowState,
  rerunWorkflowFromStep,
  runPipelineFull,
  saveLlmConfigToServer,
  startSessionUpload,
} from '../api'
import type { WorkflowStateDto, WorkflowStateResponse } from '../api'
import type { DagResult, OperatorItem } from '../types'

type Language = 'zh' | 'en'
type Metric = { sec: number; tokens: number }
type DagStatus = 'passed' | 'failed'

interface DagTab {
  id: number
  title: string
  status: DagStatus
  summary: string
  metrics: Metric
  nodes: string[]
  dag?: DagResult
}

interface InstantiationCard {
  id: string
  name: string
  summary: string
  code: string
  metrics: Metric
}

interface EvolutionRow {
  id: number
  understandingDone: boolean
  understandingMetrics?: Metric
  dagTabs: DagTab[]
  activeDagTabId?: number
  instantiationCards: InstantiationCard[]
  sampleScore?: number
  sampleMetrics?: Metric
  experience?: string
  experienceMetrics?: Metric
  needNext: boolean
  completed: boolean
}

type StepTimingMap = Record<string, number>

function timingKey(round: number, step: string): string {
  return `${Math.max(1, round)}:${step}`
}

function stepDuration(timing: StepTimingMap, round: number, step: string): number | undefined {
  const v = timing[timingKey(round, step)]
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

function applyMeasuredDurations(rows: EvolutionRow[], timing: StepTimingMap): EvolutionRow[] {
  return rows.map((row) => {
    const round = Math.max(1, row.id)
    const understandingSec = stepDuration(timing, round, 'understanding')
    const orchSec = stepDuration(timing, round, 'orchestration')
    const evoSec = stepDuration(timing, round, 'operator_evolution')
    const instSec = stepDuration(timing, round, 'instantiation')
    const trialSec = stepDuration(timing, round, 'trial_run')
    const qualitySec = stepDuration(timing, round, 'quality_check')
    const expSec = stepDuration(timing, round, 'experience')
    const sampleSec =
      typeof trialSec === 'number' || typeof qualitySec === 'number'
        ? (trialSec ?? 0) + (qualitySec ?? 0)
        : undefined

    const dagTabs = row.dagTabs.map((tab) => {
      const title = (tab.title || '').toLowerCase()
      let sec = tab.metrics.sec
      if (typeof orchSec === 'number' && (title.includes('编排') || title.includes('orchestration') || title.includes('校验') || title.includes('validation'))) {
        sec = orchSec
      }
      if (typeof evoSec === 'number' && (title.includes('进化') || title.includes('evolution'))) {
        sec = evoSec
      }
      return sec !== tab.metrics.sec ? { ...tab, metrics: { ...tab.metrics, sec } } : tab
    })

    const instantiationCards =
      typeof instSec === 'number' && row.instantiationCards.length > 0
        ? row.instantiationCards.map((card) => ({ ...card, metrics: { ...card.metrics, sec: instSec / row.instantiationCards.length } }))
        : row.instantiationCards

    return {
      ...row,
      understandingMetrics:
        row.understandingMetrics && typeof understandingSec === 'number'
          ? { ...row.understandingMetrics, sec: understandingSec }
          : row.understandingMetrics,
      dagTabs,
      instantiationCards,
      sampleMetrics:
        typeof sampleSec === 'number'
          ? { ...(row.sampleMetrics ?? { sec: 0, tokens: 0 }), sec: sampleSec }
          : row.sampleMetrics,
      experienceMetrics:
        row.experienceMetrics && typeof expSec === 'number'
          ? { ...row.experienceMetrics, sec: expSec }
          : row.experienceMetrics,
    }
  })
}

function makeEntryOnlyRow(round = 1): EvolutionRow {
  return {
    id: Math.max(1, round),
    understandingDone: false,
    dagTabs: [],
    instantiationCards: [],
    needNext: false,
    completed: false,
  }
}

type SettingsTab = 'general' | 'model'
type UploadFileKey = 'raw' | 'seed' | 'description'
type ProviderMode = 'openai-official' | 'third-party'

interface UploadPreviewState {
  raw: string[]
  seed: string[]
  description: string[]
}

const PREVIEW_LINE_COUNT = 12
const PREVIEW_JSON_KEY_COLORS = [
  'var(--preview-json-key-0)',
  'var(--preview-json-key-1)',
  'var(--preview-json-key-2)',
  'var(--preview-json-key-3)',
] as const

function renderLineWithJsonKeyHighlight(line: string, lineKey: string): ReactNode {
  const re = /\"([^\"\\]|\\.)*\"\s*:/g
  const parts: ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  let k = 0
  while ((m = re.exec(line)) !== null) {
    if (m.index > last) parts.push(line.slice(last, m.index))
    parts.push(
      <span key={`${lineKey}-k${k}`} style={{ color: PREVIEW_JSON_KEY_COLORS[k % PREVIEW_JSON_KEY_COLORS.length] }}>
        {m[0]}
      </span>
    )
    k++
    last = m.index + m[0].length
  }
  if (last < line.length) parts.push(line.slice(last))
  return parts.length ? <>{parts}</> : line
}

function formatPreviewLinesForDisplay(lines: string[]): string[] {
  const normalized = lines.map((line) => line.replace(/\r$/, ''))
  const joined = normalized.join('\n').trim()
  if (!joined) return []
  try {
    const parsed = JSON.parse(joined)
    return JSON.stringify(parsed, null, 2).split('\n').slice(0, 400)
  } catch { /* JSONL */ }
  const out: string[] = []
  for (const line of normalized) {
    const trimmed = line.trim()
    if (!trimmed) { out.push(''); continue }
    try {
      const parsed = JSON.parse(trimmed)
      out.push(...JSON.stringify(parsed, null, 2).split('\n'))
    } catch { out.push(line) }
    if (out.length >= 400) break
  }
  return out.slice(0, 400)
}

const MODEL_OPTIONS = [
  'gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4.1',
  'gpt-4o-mini', 'gpt-4o', 'gpt-4o-2024-08-06',
  'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-3.5-turbo',
  'o1', 'o1-mini', 'o3-mini', 'o4-mini',
] as const

// ─── Step definitions ───────────────────────────────────────────
const STEPS_ZH = [
  { id: 'upload', label: '数据准备', desc: '上传 Raw / Seed 数据' },
  { id: 'pipeline', label: 'Pipeline 自进化', desc: '编排、实例化、评估' },
  { id: 'run', label: '全量执行', desc: '应用 pipeline 到全部数据' },
]
const STEPS_EN = [
  { id: 'upload', label: 'Data Setup', desc: 'Upload Raw / Seed data' },
  { id: 'pipeline', label: 'Pipeline Evolution', desc: 'Orchestrate, instantiate, evaluate' },
  { id: 'run', label: 'Full Run', desc: 'Apply pipeline to all data' },
]

type AppStep = 'upload' | 'pipeline' | 'run'

function formatSec(sec: number): string {
  if (sec < 60) return `${sec.toFixed(1)}s`
  return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`
}
function formatTokens(n: number): string {
  if (n < 1000) return `${n}`
  return `${(n / 1000).toFixed(1)}k`
}

// ─── Multimodal data type detection ────────────────────────────
function detectDataTypes(preview: UploadPreviewState): string[] {
  const types: string[] = []
  const allLines = [...preview.raw, ...preview.seed].join(' ')
  if (/image_path|\.jpg|\.png|\.jpeg|\.webp/i.test(allLines)) types.push('image')
  if (/pdf_path|chunk_text|\.pdf/i.test(allLines)) types.push('doc')
  if (/video_path|\.mp4|\.mov/i.test(allLines)) types.push('video')
  if (types.length === 0) types.push('text')
  return types
}

// ─── Main component ──────────────────────────────────────────────
export function MainLayout() {
  const runMode = useAppStore((s) => s.runMode)
  const setRunMode = useAppStore((s) => s.setRunMode)
  const upload = useAppStore((s) => s.upload)
  const setUpload = useAppStore((s) => s.setUpload)
  const themeMode = useAppStore((s) => s.themeMode)
  const setThemeMode = useAppStore((s) => s.setThemeMode)
  const language = useAppStore((s) => s.language)
  const setLanguage = useAppStore((s) => s.setLanguage)
  const llmConfig = useAppStore((s) => s.llmConfig)
  const setLlmConfig = useAppStore((s) => s.setLlmConfig)
  const setToast = useAppStore((s) => s.setToast)
  const pipelineId = useAppStore((s) => s.pipelineId)
  const setPipelineId = useAppStore((s) => s.setPipelineId)

  const isZh = language === 'zh'
  const steps = isZh ? STEPS_ZH : STEPS_EN

  // ── Core state ──
  const [activeStep, setActiveStep] = useState<AppStep>('upload')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsTab, setSettingsTab] = useState<SettingsTab>('general')
  const [providerMode, setProviderMode] = useState<ProviderMode>('openai-official')
  const [apiKeyDraft, setApiKeyDraft] = useState('')
  const [baseUrlDraft, setBaseUrlDraft] = useState('')
  const [serverKeyHint, setServerKeyHint] = useState<{ configured: boolean; masked: string } | null>(null)
  const [rows, setRows] = useState<EvolutionRow[]>([])
  const [operatorPool, setOperatorPool] = useState<OperatorItem[]>([])
  const [activeRowId, setActiveRowId] = useState(1)
  const [finished, setFinished] = useState(false)
  const [uploadPreview, setUploadPreview] = useState<UploadPreviewState>({ raw: [], seed: [], description: [] })
  const [previewModalOpen, setPreviewModalOpen] = useState(false)
  const [previewModalTab, setPreviewModalTab] = useState<UploadFileKey>('raw')
  const [startModalOpen, setStartModalOpen] = useState(false)
  const [pipelineIdInput, setPipelineIdInput] = useState('')
  const [metaDomainInput, setMetaDomainInput] = useState('')
  const [metaTaskTypeInput, setMetaTaskTypeInput] = useState('')
  const [metaLanguageInput, setMetaLanguageInput] = useState('')
  const [startSubmitting, setStartSubmitting] = useState(false)
  const [configSaving, setConfigSaving] = useState(false)
  const [liveArtifacts, setLiveArtifacts] = useState<LiveArtifactsForCanvas | null>(null)
  const [workflowBusy, setWorkflowBusy] = useState(false)
  const [workflowApiAvailable, setWorkflowApiAvailable] = useState(true)
  const [workflowExecutingStep, setWorkflowExecutingStep] = useState<string | null>(null)
  const [lastExecutedStep, setLastExecutedStep] = useState<string | null>(null)
  const [stepDurations, setStepDurations] = useState<StepTimingMap>({})
  const [canRunFull, setCanRunFull] = useState(false)
  const [workflowState, setWorkflowState] = useState<WorkflowStateDto | null>(null)
  const [artifactHistoryCount, setArtifactHistoryCount] = useState(0)
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false)
  const [dragOver, setDragOver] = useState<UploadFileKey | null>(null)

  const uploadFilesRef = useRef<Partial<Record<UploadFileKey, File>>>({})
  const rawInputRef = useRef<HTMLInputElement | null>(null)
  const seedInputRef = useRef<HTMLInputElement | null>(null)
  const descInputRef = useRef<HTMLInputElement | null>(null)
  const sessionInputRef = useRef<HTMLInputElement | null>(null)

  const detectedTypes = useMemo(() => detectDataTypes(uploadPreview), [uploadPreview])

  // ── Step progress ──
  const stepProgress = useMemo((): number => {
    if (!pipelineId) return 0
    const order = workflowState?.step_order ?? []
    const idx = workflowState?.step_index ?? 0
    if (!order.length) return 0
    return Math.round((idx / order.length) * 100)
  }, [pipelineId, workflowState])

  const stepsDone = useMemo(() => {
    const done = new Set<AppStep>()
    if (upload.raw || upload.seed) done.add('upload')
    if (pipelineId && (rows.some(r => r.understandingDone) || workflowState)) done.add('upload')
    if (canRunFull || workflowState?.ready_for_full_run) done.add('pipeline')
    if (finished) { done.add('pipeline'); done.add('run') }
    return done
  }, [upload, pipelineId, rows, workflowState, canRunFull, finished])

  // ── LLM config init ──
  useEffect(() => {
    const storedApiKey = localStorage.getItem('dataevolver.apiKey')
    if (storedApiKey) setApiKeyDraft(storedApiKey)
  }, [])

  useEffect(() => {
    let cancelled = false
    void fetchLlmConfigFromServer()
      .then((data) => {
        if (cancelled) return
        setProviderMode(data.provider_mode)
        setLlmConfig({ model: data.model, temperature: data.temperature, maxTokens: data.max_tokens, provider: data.provider_mode === 'openai-official' ? 'openai' : 'custom' })
        setBaseUrlDraft(data.provider_mode === 'third-party' ? data.base_url : '')
        setServerKeyHint({ configured: data.api_key_configured, masked: data.api_key_masked })
      })
      .catch(() => {
        if (cancelled) return
        const storedBaseUrl = localStorage.getItem('dataevolver.baseUrl')
        const storedProviderMode = localStorage.getItem('dataevolver.providerMode')
        if (storedBaseUrl) setBaseUrlDraft(storedBaseUrl)
        if (storedProviderMode === 'openai-official' || storedProviderMode === 'third-party') setProviderMode(storedProviderMode)
        setServerKeyHint(null)
      })
    return () => { cancelled = true }
  }, [setLlmConfig])

  useEffect(() => { localStorage.setItem('dataevolver.apiKey', apiKeyDraft) }, [apiKeyDraft])
  useEffect(() => { localStorage.setItem('dataevolver.baseUrl', baseUrlDraft) }, [baseUrlDraft])
  useEffect(() => {
    localStorage.setItem('dataevolver.providerMode', providerMode)
    setLlmConfig({ provider: providerMode === 'openai-official' ? 'openai' : 'custom' })
  }, [providerMode, setLlmConfig])

  // ── Save LLM config ──
  const handleSaveLlmConfig = useCallback(async () => {
    setConfigSaving(true)
    try {
      await saveLlmConfigToServer({ provider_mode: providerMode, model: llmConfig.model, temperature: llmConfig.temperature, max_tokens: llmConfig.maxTokens, api_key: apiKeyDraft, base_url: baseUrlDraft })
      try {
        const refreshed = await fetchLlmConfigFromServer()
        setServerKeyHint({ configured: refreshed.api_key_configured, masked: refreshed.api_key_masked })
      } catch { setServerKeyHint((prev) => prev ?? { configured: true, masked: '' }) }
      setToast({ message: isZh ? '配置已保存' : 'Config saved', type: 'success' })
    } catch (e) {
      let msg = isZh ? '保存失败' : 'Save failed'
      if (e instanceof ApiError && e.detail) msg += `: ${e.detail.slice(0, 200)}`
      setToast({ message: msg, type: 'error' })
    } finally { setConfigSaving(false) }
  }, [apiKeyDraft, baseUrlDraft, llmConfig, providerMode, setToast, isZh])

  // ── Operator pool ──
  const reloadOperatorPool = useCallback(async () => {
    try {
      const data = await fetchOperators()
      setOperatorPool(mapApiOperatorsToPoolItems(data.operators, language))
    } catch (e) {
      const detail = e instanceof ApiError ? e.detail : String(e)
      setToast({ message: `${isZh ? '算子池加载失败' : 'Failed to load operator pool'}${detail ? `: ${detail.slice(0, 160)}` : ''}`, type: 'error' })
      setOperatorPool([])
    }
  }, [language, setToast, isZh])

  useEffect(() => { void reloadOperatorPool() }, [reloadOperatorPool])

  // ── Pipeline refresh ──
  async function fetchOptional404<T>(fn: () => Promise<T>): Promise<T | null> {
    try { return await fn() }
    catch (e) { if (e instanceof ApiError && e.status === 404) return null; throw e }
  }

  const refreshPipelineFromServer = useCallback(async (pid: string, timingPatch?: StepTimingMap) => {
    let wf: WorkflowStateResponse | null = null
    try { wf = await fetchWorkflowState(pid); setWorkflowApiAvailable(true) }
    catch (e) { if (!(e instanceof ApiError) || e.status !== 404) throw e; setWorkflowApiAvailable(false) }
    const [underRes, dagJson, inst, quality, trial, experience, tokens, _runLatest, _history] = await Promise.all([
      fetchOptional404(() => fetchUnderstandingResult(pid)),
      fetchOptional404(() => fetchOrchestrationDag(pid)),
      fetchOptional404(() => fetchInstantiationSteps(pid, { include_code: true })),
      fetchOptional404(() => fetchQualityCheck(pid)),
      fetchOptional404(() => fetchTrialResult(pid)),
      fetchOptional404(() => fetchExperience(pid)),
      fetchOptional404(() => fetchWorkflowTokens(pid)),
      fetchOptional404(() => fetchPipelineRunLatest(pid)),
      fetchOptional404(() => fetchArtifactHistory(pid, 40)),
    ])
    const understanding = underRes?.data ?? null
    void _runLatest
    setArtifactHistoryCount(_history?.entries?.length ?? 0)
    if (!wf) {
      wf = { ok: true, pipeline_id: pid, state: { pipeline_id: pid, step_index: 0, steps_completed: understanding ? ['understanding'] : [], last_message: '', updated_at: '', round: 1, quality_passed: false, ready_for_full_run: false, next_action: 'advance', step_order: ['understanding'], is_complete: false }, artifacts: { understanding: Boolean(understanding), orchestration: Boolean(dagJson), instantiation: Boolean(inst?.steps?.length), trial_run: Boolean(trial?.data), pipeline_run: Boolean(_runLatest?.latest || _runLatest?.report), quality_check: Boolean(quality?.data), experience: Boolean(experience?.data) } }
    }
    const built = buildEvolutionRowsFromPipeline(wf, { understanding, dagResponse: dagJson, instantiationSteps: inst?.steps ?? null, quality: quality?.data ?? null, trial: trial?.data ?? null, experience: experience?.data ?? null, tokens: tokens ?? null, history: _history ?? null, lastAdvanceMessage: wf.state.last_message ?? '', language })
    const timing = timingPatch ? { ...stepDurations, ...timingPatch } : stepDurations
    const measuredRows = applyMeasuredDurations(built.rows as EvolutionRow[], timing)
    setRows((prev) => {
      const incoming = measuredRows
      if (!incoming.length) return []
      if (incoming.length === 1) {
        const cur = incoming[0]
        const prevRoundId = Math.max(0, cur.id - 1)
        const preserved = prev.filter((r) => r.id < cur.id).map((r) => r.id === prevRoundId ? { ...r, completed: true, needNext: true } : { ...r, completed: true })
        const merged = [...preserved, cur]
        merged.sort((a, b) => a.id - b.id)
        return merged
      }
      return incoming
    })
    setFinished(built.finished)
    setLiveArtifacts(built.live)
    setWorkflowState(wf.state)
    setCanRunFull(Boolean(wf.state.ready_for_full_run || wf.state.is_complete || built.canRunFullByJudge))
    setActiveRowId(built.rows[built.rows.length - 1]?.id ?? 1)
    if (wf.state.ready_for_full_run || built.canRunFullByJudge) setActiveStep('run')
    else if (wf.state.step_index > 0 || understanding) setActiveStep('pipeline')
  }, [language, stepDurations])

  const syncPreviewsFromServer = useCallback(async (pid: string) => {
    const kinds: UploadFileKey[] = ['raw', 'seed', 'description']
    const next: UploadPreviewState = { raw: [], seed: [], description: [] }
    for (const kind of kinds) {
      try {
        const p = await fetchPipelinePreview(pid, kind, { max_lines: PREVIEW_LINE_COUNT })
        if (p.lines?.length) next[kind] = p.lines
      } catch { /* no file */ }
    }
    setUploadPreview((prev) => ({ ...prev, ...next }))
  }, [])

  useEffect(() => {
    if (!pipelineId) { setLiveArtifacts(null); setWorkflowState(null); return }
    let cancelled = false
    setWorkflowBusy(true)
    void (async () => {
      try { await refreshPipelineFromServer(pipelineId) }
      catch (e) {
        if (!cancelled) {
          const detail = e instanceof ApiError ? e.detail : String(e)
          setToast({ message: (isZh ? '同步管线状态失败' : 'Failed to sync pipeline state') + (detail ? `: ${detail.slice(0, 160)}` : ''), type: 'error' })
        }
      } finally { if (!cancelled) setWorkflowBusy(false) }
    })()
    return () => { cancelled = true }
  }, [pipelineId, refreshPipelineFromServer, isZh, setToast])

  const totalMetrics = useMemo(() => {
    let sec = 0; let tokens = 0
    rows.forEach((row) => {
      if (row.understandingMetrics) { sec += row.understandingMetrics.sec; tokens += row.understandingMetrics.tokens }
      row.dagTabs.forEach((tab) => { sec += tab.metrics.sec; tokens += tab.metrics.tokens })
      row.instantiationCards.forEach((card) => { sec += card.metrics.sec; tokens += card.metrics.tokens })
      if (row.sampleMetrics) { sec += row.sampleMetrics.sec; tokens += row.sampleMetrics.tokens }
      if (row.experienceMetrics) { sec += row.experienceMetrics.sec; tokens += row.experienceMetrics.tokens }
    })
    return { sec, tokens }
  }, [rows])

  const formatSize = (size: number) => {
    if (size < 1024) return `${size} B`
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
    return `${(size / (1024 * 1024)).toFixed(1)} MB`
  }

  const parsePreviewLines = async (file: File) => {
    const text = await file.text()
    const rawLines = text.split(/\r?\n/)
    const rows = rawLines.filter((line) => line.trim().length > 0).length
    const preview = rawLines.slice(0, PREVIEW_LINE_COUNT)
    return { rows, preview }
  }

  const handleFileSelect = (key: UploadFileKey) => async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    const { rows: lineCount, preview } = await parsePreviewLines(file)
    uploadFilesRef.current[key] = file
    setUpload({ [key]: { name: file.name, size: file.size, rows: lineCount } })
    setUploadPreview((prev) => ({ ...prev, [key]: preview }))
  }

  const handleDrop = (key: UploadFileKey) => async (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(null)
    const file = e.dataTransfer.files?.[0]
    if (!file) return
    const { rows: lineCount, preview } = await parsePreviewLines(file)
    uploadFilesRef.current[key] = file
    setUpload({ [key]: { name: file.name, size: file.size, rows: lineCount } })
    setUploadPreview((prev) => ({ ...prev, [key]: preview }))
  }

  const selectDagTab = (rowId: number, tabId: number) => {
    setRows((prev) => prev.map((row) => (row.id === rowId ? { ...row, activeDagTabId: tabId } : row)))
  }

  // ── Workflow step forward ──
  const stepForward = useCallback(async () => {
    if (!pipelineId) { setToast({ message: isZh ? '请先启动会话。' : 'Start a session first.', type: 'info' }); return }
    if (finished) return
    if (!workflowApiAvailable) { setToast({ message: isZh ? '后端 workflow 接口不可用。' : 'Workflow API unavailable.', type: 'error' }); return }
    if (workflowState?.next_action === 'run_full' || workflowState?.ready_for_full_run) {
      setToast({ message: isZh ? '闭环已通过，请点击"应用全量"。' : 'Quality loop passed. Click "Apply Full Run".', type: 'info' }); return
    }
    setWorkflowBusy(true)
    const order = workflowState?.step_order ?? []
    const idx = workflowState?.step_index ?? 0
    const stepKey = idx >= 0 && idx < order.length ? order[idx] : null
    const roundId = Math.max(1, Number(workflowState?.round ?? 1))
    setWorkflowExecutingStep(stepKey)
    const startedAt = Date.now()
    try {
      const res = await advanceWorkflow(pipelineId)
      const elapsedSec = Math.max(0.01, (Date.now() - startedAt) / 1000)
      const executedStep = (res.step || stepKey || '').trim()
      const timingPatch: StepTimingMap = executedStep.length > 0 ? { [timingKey(roundId, executedStep)]: elapsedSec } : {}
      if (executedStep.length > 0) setLastExecutedStep(executedStep)
      if (Object.keys(timingPatch).length > 0) setStepDurations((prev) => ({ ...prev, ...timingPatch }))
      await refreshPipelineFromServer(pipelineId, timingPatch)
      if (res.done) setToast({ message: isZh ? '全部流程已完成。' : 'All steps complete.', type: 'success' })
    } catch (e) {
      const raw = e instanceof ApiError ? e.detail : String(e)
      const parsed = parseFastApiErrorBody(raw)
      try { await refreshPipelineFromServer(pipelineId) } catch { /* sync fail */ }
      setToast({ message: (isZh ? '推进失败：' : 'Step failed: ') + parsed.userMessage, type: 'error' })
    } finally { setWorkflowExecutingStep(null); setWorkflowBusy(false) }
  }, [pipelineId, finished, workflowApiAvailable, workflowState, refreshPipelineFromServer, isZh, setToast])

  const autoCompleteCurrentRound = useCallback(async () => {
    if (!pipelineId) { setToast({ message: isZh ? '请先启动会话。' : 'Start a session first.', type: 'info' }); return }
    if (!workflowApiAvailable) { setToast({ message: isZh ? '后端 workflow 接口不可用。' : 'Workflow API unavailable.', type: 'error' }); return }
    setWorkflowBusy(true)
    try {
      let guard = 0
      const timingPatch: StepTimingMap = {}
      const baseRound = Math.max(1, Number(workflowState?.round ?? 1))
      while (guard < 32) {
        const wf = await fetchWorkflowState(pipelineId)
        if (wf.state.is_complete || wf.state.ready_for_full_run) break
        const nextKey = wf.state.step_index >= 0 && wf.state.step_index < wf.state.step_order.length ? wf.state.step_order[wf.state.step_index] : null
        const roundId = Math.max(1, Number(wf.state.round ?? 1))
        setWorkflowExecutingStep(nextKey)
        const startedAt = Date.now()
        const adv = await advanceWorkflow(pipelineId)
        const elapsedSec = Math.max(0.01, (Date.now() - startedAt) / 1000)
        const executedStep = (adv.step || nextKey || '').trim()
        if (executedStep.length > 0) { timingPatch[timingKey(roundId, executedStep)] = elapsedSec; setLastExecutedStep(executedStep) }
        guard += 1
        if (adv.done) break
        if (executedStep === 'experience') break
        if (Math.max(1, Number(adv.state?.round ?? baseRound)) > baseRound) break
      }
      if (Object.keys(timingPatch).length > 0) setStepDurations((prev) => ({ ...prev, ...timingPatch }))
      await refreshPipelineFromServer(pipelineId, timingPatch)
      setToast({ message: isZh ? '自动推进结束，画布已同步。' : 'Auto-advance finished; canvas synced.', type: 'success' })
    } catch (e) {
      const raw = e instanceof ApiError ? e.detail : String(e)
      const parsed = parseFastApiErrorBody(raw)
      try { await refreshPipelineFromServer(pipelineId) } catch { /* ignore */ }
      setToast({ message: (isZh ? '自动推进失败：' : 'Auto-advance failed: ') + parsed.userMessage, type: 'error' })
    } finally { setWorkflowExecutingStep(null); setWorkflowBusy(false) }
  }, [pipelineId, workflowApiAvailable, workflowState, refreshPipelineFromServer, isZh, setToast])

  const runAutoModeWorkflow = useCallback(async (pid: string) => {
    if (!pid || !workflowApiAvailable) return
    setWorkflowBusy(true)
    setWorkflowExecutingStep(null)
    try {
      const timingPatch: StepTimingMap = {}
      let guard = 0
      while (guard < 96) {
        const wf = await fetchWorkflowState(pid)
        if (wf.state.ready_for_full_run || wf.state.next_action === 'run_full' || wf.state.is_complete) {
          await runPipelineFull(pid)
          if (Object.keys(timingPatch).length > 0) setStepDurations((prev) => ({ ...prev, ...timingPatch }))
          await refreshPipelineFromServer(pid, timingPatch)
          setToast({ message: isZh ? '自动模式已完成（含全量运行）。' : 'Auto mode finished (including full run).', type: 'success' })
          return
        }
        const order = wf.state.step_order ?? []
        const idx = wf.state.step_index ?? 0
        const stepKey = idx >= 0 && idx < order.length ? order[idx] : null
        const roundId = Math.max(1, Number(wf.state.round ?? 1))
        setWorkflowExecutingStep(stepKey)
        const startedAt = Date.now()
        const adv = await advanceWorkflow(pid)
        const elapsedSec = Math.max(0.01, (Date.now() - startedAt) / 1000)
        const executedStep = (adv.step || stepKey || '').trim()
        if (executedStep) { timingPatch[timingKey(roundId, executedStep)] = elapsedSec; setLastExecutedStep(executedStep) }
        guard += 1
        if (!adv.done && !adv.state?.ready_for_full_run && adv.state?.next_action !== 'run_full') continue
      }
      if (Object.keys(timingPatch).length > 0) setStepDurations((prev) => ({ ...prev, ...timingPatch }))
      await refreshPipelineFromServer(pid, timingPatch)
    } catch (e) {
      const raw = e instanceof ApiError ? e.detail : String(e)
      const parsed = parseFastApiErrorBody(raw)
      try { await refreshPipelineFromServer(pid) } catch { /* ignore */ }
      setToast({ message: (isZh ? '自动模式失败：' : 'Auto mode failed: ') + parsed.userMessage, type: 'error' })
    } finally { setWorkflowExecutingStep(null); setWorkflowBusy(false) }
  }, [workflowApiAvailable, isZh, refreshPipelineFromServer, setToast])

  const openStartModal = () => {
    const hasMeta = Boolean(upload.raw || upload.seed || upload.description)
    if (!hasMeta) { setToast({ message: isZh ? '请先上传至少一个文件' : 'Upload at least one file first', type: 'error' }); return }
    if (upload.raw && !uploadFilesRef.current.raw) { setToast({ message: isZh ? '请重新上传文件' : 'Re-upload files', type: 'error' }); return }
    if (upload.seed && !uploadFilesRef.current.seed) { setToast({ message: isZh ? '请重新上传文件' : 'Re-upload files', type: 'error' }); return }
    if (upload.description && !uploadFilesRef.current.description) { setToast({ message: isZh ? '请重新上传文件' : 'Re-upload files', type: 'error' }); return }
    setPipelineIdInput(upload.pipelineId ?? '')
    setStartModalOpen(true)
  }

  const submitStartSession = async () => {
    const pid = pipelineIdInput.trim()
    if (!pid) { setToast({ message: isZh ? '请填写 Pipeline ID' : 'Pipeline ID required', type: 'error' }); return }
    const fd = new FormData()
    fd.append('pipeline_id', pid)
    if (metaDomainInput.trim()) fd.append('domain', metaDomainInput.trim())
    if (metaTaskTypeInput.trim()) fd.append('task_type', metaTaskTypeInput.trim())
    if (metaLanguageInput.trim()) fd.append('language', metaLanguageInput.trim())
    const raw = uploadFilesRef.current.raw
    const seed = uploadFilesRef.current.seed
    const desc = uploadFilesRef.current.description
    if (raw) fd.append('raw_file', raw, raw.name)
    if (seed) fd.append('seed_file', seed, seed.name)
    if (desc) fd.append('description_file', desc, desc.name)
    setStartSubmitting(true)
    try {
      await startSessionUpload(fd)
      try { await resetWorkflowForDebug(pid) }
      catch (e) {
        if (!(e instanceof ApiError) || e.status !== 404) throw e
        try { await resetWorkflowState(pid); await rerunWorkflowFromStep(pid, 'understanding') }
        catch (e2) { if (!(e2 instanceof ApiError) || e2.status !== 404) throw e2 }
      }
      setPipelineId(pid)
      setUpload({ pipelineId: pid })
      setRows([makeEntryOnlyRow(1)])
      setActiveRowId(1)
      setFinished(false)
      setCanRunFull(false)
      setLiveArtifacts(null)
      setWorkflowState(null)
      setLastExecutedStep(null)
      setStepDurations({})
      setArtifactHistoryCount(0)
      setStartModalOpen(false)
      setActiveStep('pipeline')
      setToast({ message: isZh ? '已上传并写入清单，开始推进流程' : 'Uploaded and manifest updated', type: 'success' })
      try { await syncPreviewsFromServer(pid) } catch { /* optional */ }
      if (runMode === 'auto') void runAutoModeWorkflow(pid)
    } catch (e) {
      const detail = e instanceof ApiError ? e.detail : String(e)
      setToast({ message: `${isZh ? '上传失败' : 'Upload failed'}${detail ? `: ${detail.slice(0, 200)}` : ''}`, type: 'error' })
    } finally { setStartSubmitting(false) }
  }

  const runFullData = useCallback(async () => {
    if (!pipelineId) { setToast({ message: isZh ? '请先加载有效 pipeline' : 'Load a valid pipeline first', type: 'error' }); return }
    if (!canRunFull) { setToast({ message: isZh ? '尚未通过质检闭环，暂不可全量运行' : 'Quality loop not passed yet', type: 'error' }); return }
    setWorkflowBusy(true)
    try {
      await runPipelineFull(pipelineId)
      await refreshPipelineFromServer(pipelineId)
      setToast({ message: isZh ? '全量运行完成，画布已同步。' : 'Full run finished; canvas synced.', type: 'success' })
      setActiveStep('run')
    } catch (e) {
      const raw = e instanceof ApiError ? e.detail : String(e)
      const parsed = parseFastApiErrorBody(raw)
      setToast({ message: (isZh ? '全量运行失败：' : 'Full run failed: ') + parsed.userMessage, type: 'error' })
    } finally { setWorkflowBusy(false) }
  }, [pipelineId, canRunFull, refreshPipelineFromServer, isZh, setToast])

  const rerunFromCurrentStep = useCallback(async () => {
    if (!pipelineId) return
    const stepOrder = workflowState?.step_order ?? []
    const stepIndex = workflowState?.step_index ?? 0
    const stateMessage = (workflowState?.last_message ?? '').trim()
    const inferredByMessage = stateMessage.includes(':') ? stateMessage.split(':', 1)[0].trim() : ''
    const fallbackStep =
      stepOrder.length === 0 ? 'understanding'
      : stepIndex <= 0 ? stepOrder[0]
      : stepIndex >= stepOrder.length
        ? workflowState?.ready_for_full_run || workflowState?.next_action === 'run_full'
          ? 'quality_check' : stepOrder[stepOrder.length - 1]
        : stepOrder[stepIndex - 1]
    const targetStep =
      (lastExecutedStep && stepOrder.includes(lastExecutedStep) && lastExecutedStep) ||
      (inferredByMessage && stepOrder.includes(inferredByMessage) && inferredByMessage) ||
      fallbackStep
    setWorkflowBusy(true)
    try {
      await rerunWorkflowFromStep(pipelineId, targetStep)
      setWorkflowExecutingStep(targetStep)
      const startedAt = Date.now()
      const res = await advanceWorkflow(pipelineId)
      const elapsedSec = Math.max(0.01, (Date.now() - startedAt) / 1000)
      const executedStep = (res.step || targetStep || '').trim()
      const roundId = Math.max(1, Number(workflowState?.round ?? 1))
      const timingPatch: StepTimingMap = executedStep.length > 0 ? { [timingKey(roundId, executedStep)]: elapsedSec } : {}
      if (Object.keys(timingPatch).length > 0) setStepDurations((prev) => ({ ...prev, ...timingPatch }))
      await refreshPipelineFromServer(pipelineId, timingPatch)
      setLastExecutedStep(executedStep || targetStep)
      setToast({ message: isZh ? `已重跑步骤 ${executedStep || targetStep}。` : `Step ${executedStep || targetStep} rerun completed.`, type: 'success' })
    } catch (e) {
      const detail = e instanceof ApiError ? e.detail : String(e)
      setToast({ message: `${isZh ? '重跑失败' : 'Rerun failed'}${detail ? `: ${detail.slice(0, 180)}` : ''}`, type: 'error' })
    } finally { setWorkflowExecutingStep(null); setWorkflowBusy(false) }
  }, [pipelineId, workflowState, lastExecutedStep, refreshPipelineFromServer, isZh, setToast])

  const handleClearWorkspace = () => {
    uploadFilesRef.current = {}
    setUpload({})
    setUploadPreview({ raw: [], seed: [], description: [] })
    setPipelineId(null)
    setLiveArtifacts(null)
    setRows([])
    void reloadOperatorPool()
    setActiveRowId(1)
    setLastExecutedStep(null)
    setFinished(false)
    setActiveStep('upload')
    setToast({ message: isZh ? '已清除工作区' : 'Workspace cleared', type: 'info' })
  }

  const handleSaveSession = () => {
    const payload = { format: 'dataevolver-session-v1', savedAt: new Date().toISOString(), upload, uploadPreview, runMode, themeMode, language, llmConfig, rows, operatorPool, activeRowId, finished }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'dataevolver.session.json'
    document.body.appendChild(a); a.click(); a.remove()
    URL.revokeObjectURL(url)
    setToast({ message: isZh ? '会话已导出' : 'Session exported', type: 'success' })
  }

  const handleLoadSession = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    try {
      const text = await file.text()
      const parsed = JSON.parse(text)
      if (parsed?.format !== 'dataevolver-session-v1') throw new Error('invalid format')
      setUpload(parsed.upload ?? {})
      setUploadPreview(parsed.uploadPreview ?? { raw: [], seed: [], description: [] })
      if (parsed.runMode === 'manual' || parsed.runMode === 'auto') setRunMode(parsed.runMode)
      if (parsed.themeMode === 'light' || parsed.themeMode === 'dark') setThemeMode(parsed.themeMode)
      if (parsed.language === 'zh' || parsed.language === 'en') setLanguage(parsed.language)
      if (parsed.llmConfig) setLlmConfig(parsed.llmConfig)
      if (Array.isArray(parsed.rows)) setRows(parsed.rows)
      if (Array.isArray(parsed.operatorPool)) setOperatorPool(parsed.operatorPool)
      if (typeof parsed.activeRowId === 'number') setActiveRowId(parsed.activeRowId)
      if (typeof parsed.finished === 'boolean') setFinished(parsed.finished)
      setLastExecutedStep(null)
      const loadedPid = parsed.upload?.pipelineId
      if (typeof loadedPid === 'string' && loadedPid.trim()) setPipelineId(loadedPid.trim())
      else setPipelineId(null)
      setToast({ message: isZh ? '会话已加载' : 'Session loaded', type: 'success' })
    } catch { setToast({ message: isZh ? '会话文件格式不正确' : 'Invalid session file', type: 'error' }) }
    finally { event.target.value = '' }
  }

  const previewTabs = useMemo(() =>
    (['raw', 'seed', 'description'] as UploadFileKey[]).filter(
      (key) => Boolean(upload[key]) || uploadPreview[key].length > 0
    ), [upload, uploadPreview])

  const previewDisplayLines = useMemo(() =>
    formatPreviewLinesForDisplay(uploadPreview[previewModalTab] ?? []),
    [previewModalTab, uploadPreview])

  useEffect(() => {
    if (!previewModalOpen) return
    const onKeyDown = (e: KeyboardEvent) => { if (e.key === 'Escape') setPreviewModalOpen(false) }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [previewModalOpen])

  // ── Canvas text bundle ──
  const canvasT = {
    operatorPool: isZh ? '算子池' : 'Operator Pool',
    operatorHint: isZh ? '系统会在算子自进化阶段实时沉淀新算子。' : 'New operators are added during operator-level evolution.',
    sourceBase: isZh ? '基础' : 'Base',
    sourceEvolved: isZh ? '进化' : 'Evolved',
    pipelineCanvas: isZh ? 'Pipeline 自进化画布' : 'Pipeline Self-Evolution Canvas',
    progressHint: isZh ? '点击"推进一步"按流程自动补全区块内容。' : 'Click "Step Forward" to auto-fill blocks.',
    stepForward: isZh ? '推进一步' : 'Step Forward',
    autoBuildRound: isZh ? '自动补完整轮' : 'Auto Complete Round',
    totalTime: isZh ? '累计用时' : 'Total Time',
    totalTokens: isZh ? '累计 Token' : 'Total Tokens',
    roundTitle: isZh ? '流水线级自进化轮次' : 'Pipeline Self-Evolution Round',
    rawBlock: isZh ? 'Raw Data' : 'Raw Data',
    understanding: isZh ? '理解' : 'Understanding',
    operatorEvolution: isZh ? '算子级自进化' : 'Operator-Level Evolution',
    dagTabs: isZh ? 'DAG 编排结果' : 'DAG Orchestration Results',
    instantiation: isZh ? '实例化' : 'Instantiation',
    sampleEval: isZh ? 'Sample + LLM 评估' : 'Sample + LLM Evaluation',
    experience: isZh ? '经验总结' : 'Experience Summary',
    pending: isZh ? '待执行' : 'Pending',
    done: isZh ? '已完成' : 'Done',
    checkNeedEvolution: isZh ? '检测结果：需要算子进化' : 'Check: evolution required',
    checkPass: isZh ? '检测结果：通过' : 'Check: passed',
    llmScore: isZh ? 'LLM 评分' : 'LLM Score',
    willIterate: isZh ? '触发下一轮流水线级迭代' : 'Trigger next pipeline iteration',
    qualityOk: isZh ? '达到目标质量，结束迭代' : 'Target quality reached, stop iterations',
    runFullData: isZh ? '应用全量' : 'Apply Full Run',
    runQueuedToast: isZh ? '已启动流程推进。' : 'Evolution flow started.',
    fullDataToast: isZh ? '全量运行完成，画布已同步。' : 'Full run finished; canvas synced.',
    running: isZh ? '运行中' : 'Running',
    active: isZh ? '当前轮' : 'Active',
    autoTag: isZh ? '自动新增 Tag' : 'Auto Tag',
    backHome: isZh ? '返回主页' : 'Back Home',
  }

  // ── Upload card ──
  const renderUploadCard = (
    label: string,
    sublabel: string,
    key: UploadFileKey,
    icon: ReactNode,
    inputRef: { current: HTMLInputElement | null }
  ) => {
    const fileMeta = upload[key]
    const isDrag = dragOver === key
    return (
      <div className="notion-card slide-in" style={{ animationDelay: key === 'raw' ? '0ms' : key === 'seed' ? '60ms' : '120ms' }}>
        <div className="flex items-start gap-3 mb-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0" style={{ background: 'var(--brand-glow)', border: '1px solid var(--brand-dim)' }}>
            {icon}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium" style={{ color: 'var(--text)', fontFamily: 'Lora, serif' }}>{label}</p>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{sublabel}</p>
          </div>
          {fileMeta && (
            <button
              type="button"
              className="btn-ghost"
              onClick={() => { setPreviewModalTab(key); setPreviewModalOpen(true) }}
            >
              <Eye className="w-3.5 h-3.5" />
              {isZh ? '预览' : 'Preview'}
            </button>
          )}
        </div>
        <div
          className={`upload-zone ${fileMeta ? 'has-file' : ''} ${isDrag ? 'drag-over' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(key) }}
          onDragLeave={() => setDragOver(null)}
          onDrop={handleDrop(key)}
          onClick={() => inputRef.current?.click()}
        >
          <input ref={inputRef} type="file" className="hidden" onChange={handleFileSelect(key)} />
          {fileMeta ? (
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 min-w-0">
                <FileJson2 className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--brand)' }} />
                <div className="min-w-0">
                  <p className="text-sm truncate" style={{ color: 'var(--text)' }}>{fileMeta.name}</p>
                  <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                    {'rows' in fileMeta && fileMeta.rows !== undefined ? `${fileMeta.rows} rows` : ''}
                    {fileMeta.size != null ? ` · ${formatSize(fileMeta.size)}` : ''}
                  </p>
                </div>
              </div>
              <span className="pill pill-brand flex-shrink-0">
                <Check className="w-3 h-3" />
                {isZh ? '已上传' : 'Uploaded'}
              </span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-1.5">
              <UploadCloud className="w-6 h-6" style={{ color: 'var(--text-muted)' }} />
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {isZh ? '拖拽或点击上传' : 'Drag or click to upload'}
              </p>
              <p className="text-[11px]" style={{ color: 'var(--text-placeholder)' }}>JSON · JSONL · TXT · CSV</p>
            </div>
          )}
        </div>
      </div>
    )
  }

  // ── Step nav item ──
  const renderStepNav = (step: typeof STEPS_ZH[0], index: number) => {
    const done = stepsDone.has(step.id as AppStep)
    const active = activeStep === step.id
    return (
      <button
        key={step.id}
        type="button"
        className={`step-nav-item w-full ${active ? 'active' : ''} ${done ? 'done' : ''}`}
        onClick={() => setActiveStep(step.id as AppStep)}
      >
        <div className={`step-num ${active ? 'active' : ''}`}>
          {done ? <Check className="w-3 h-3" style={{ color: 'var(--step-done-color)' }} /> : <span>{index + 1}</span>}
        </div>
        <div className="flex-1 text-left min-w-0">
          <p className="text-[13px] font-medium leading-tight truncate">{step.label}</p>
          <p className="text-[11px] mt-0.5 truncate" style={{ color: 'var(--text-muted)' }}>{step.desc}</p>
        </div>
        {active && <ChevronRight className="w-3.5 h-3.5 flex-shrink-0" style={{ color: 'var(--brand)' }} />}
      </button>
    )
  }

  return (
    <div className="min-h-screen relative z-10" style={{ background: 'var(--bg-deep)' }}>
      <div
        className="mx-auto flex flex-col"
        style={{
          maxWidth: '1600px',
          height: '100vh',
          background: 'var(--bg-panel)',
          boxShadow: 'var(--shell-shadow)',
        }}
      >
        {/* ── Topbar ── */}
        <header className="topbar" style={{ borderBottom: '1px solid var(--border-soft)' }}>
          {/* Logo */}
          <div className="flex items-center gap-2.5 flex-shrink-0">
            <div className="w-8 h-8 rounded-xl flex items-center justify-center" style={{ background: 'var(--brand-glow)', border: '1px solid var(--brand-dim)' }}>
              <Sparkles className="w-4 h-4" style={{ color: 'var(--brand)' }} />
            </div>
            <div>
              <p className="text-sm font-semibold leading-none" style={{ fontFamily: 'Lora, serif', color: 'var(--text)' }}>
                DataEvolver 2.0
              </p>
              <p className="text-[10px] leading-none mt-0.5" style={{ color: 'var(--text-muted)' }}>
                Multimodal Data Preparation
              </p>
            </div>
          </div>

          {/* Pipeline ID + status */}
          {pipelineId && (
            <>
              <div className="w-px h-5 mx-1 flex-shrink-0" style={{ background: 'var(--border)' }} />
              <div className="flex items-center gap-2">
                <Database className="w-3.5 h-3.5 flex-shrink-0" style={{ color: 'var(--text-muted)' }} />
                <span className="text-sm font-mono" style={{ color: 'var(--text-dim)' }}>{pipelineId}</span>
                {workflowBusy && (
                  <span className="pill pill-brand">
                    <div className="spinner" style={{ width: 10, height: 10 }} />
                    {workflowExecutingStep ? (isZh ? `执行 ${workflowExecutingStep}` : workflowExecutingStep) : (isZh ? '运行中' : 'Running')}
                  </span>
                )}
                {!workflowBusy && workflowState?.ready_for_full_run && (
                  <span className="pill pill-success">
                    <Check className="w-3 h-3" />
                    {isZh ? '可全量执行' : 'Ready for full run'}
                  </span>
                )}
              </div>
            </>
          )}

          {/* Data type pills */}
          {(upload.raw || upload.seed) && (
            <div className="flex items-center gap-1.5 ml-1">
              {detectedTypes.includes('image') && <span className="pill pill-image"><Image className="w-3 h-3" />{isZh ? '图像' : 'Image'}</span>}
              {detectedTypes.includes('doc') && <span className="pill pill-doc"><FileText className="w-3 h-3" />{isZh ? '文档' : 'Document'}</span>}
              {detectedTypes.includes('text') && !detectedTypes.includes('image') && !detectedTypes.includes('doc') && (
                <span className="pill pill-info"><FileJson2 className="w-3 h-3" />{isZh ? '文本' : 'Text'}</span>
              )}
            </div>
          )}

          {/* Spacer */}
          <div className="flex-1" />

          {/* Metrics */}
          {totalMetrics.tokens > 0 && (
            <div className="hidden md:flex items-center gap-2">
              <span className="metric-chip">
                <Zap className="w-3 h-3" style={{ color: 'var(--amber)' }} />
                {formatTokens(totalMetrics.tokens)} tokens
              </span>
              <span className="metric-chip">
                {formatSec(totalMetrics.sec)}
              </span>
            </div>
          )}

          {/* Run mode */}
          <div className="flex items-center gap-1 p-0.5 rounded-lg" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <button type="button" className={`run-mode-tab ${runMode === 'manual' ? 'active' : ''}`} onClick={() => setRunMode('manual')}>
              {isZh ? '手动' : 'Manual'}
            </button>
            <button type="button" className={`run-mode-tab ${runMode === 'auto' ? 'active' : ''}`} onClick={() => setRunMode('auto')}>
              {isZh ? '自动' : 'Auto'}
            </button>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-1.5">
            <button type="button" className="btn-ghost has-tooltip relative" onClick={handleSaveSession} title="">
              <Save className="w-4 h-4" />
              <span className="tooltip-text">{isZh ? '导出会话' : 'Export session'}</span>
            </button>
            <button type="button" className="btn-ghost has-tooltip relative" onClick={() => sessionInputRef.current?.click()} title="">
              <Upload className="w-4 h-4" />
              <span className="tooltip-text">{isZh ? '加载会话' : 'Load session'}</span>
            </button>
            <button type="button" className="btn-ghost has-tooltip relative" onClick={() => setClearConfirmOpen(true)} title="" style={{ color: 'var(--error)' }}>
              <Trash2 className="w-4 h-4" />
              <span className="tooltip-text">{isZh ? '清除工作区' : 'Clear workspace'}</span>
            </button>
            <div className="w-px h-5 mx-0.5" style={{ background: 'var(--border)' }} />
            <button type="button" className="btn-ghost" onClick={() => { setSettingsTab('general'); setSettingsOpen(true) }}>
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* ── Body: sidebar + main ── */}
        <div className="flex flex-1 min-h-0">
          {/* Sidebar */}
          <nav className="sidebar">
            <div className="flex flex-col h-full">
              {/* Step navigation */}
              <div className="p-3 flex-shrink-0">
                <p className="text-[11px] font-semibold uppercase tracking-wider mb-2 px-2" style={{ color: 'var(--text-muted)' }}>
                  {isZh ? '工作流程' : 'Workflow'}
                </p>
                <div className="space-y-0.5">
                  {steps.map((step, i) => renderStepNav(step, i))}
                </div>
              </div>

              {/* Progress bar (when pipeline running) */}
              {pipelineId && stepProgress > 0 && (
                <div className="px-4 py-2 flex-shrink-0">
                  <div className="flex items-center justify-between mb-1.5">
                    <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                      {isZh ? `步骤 ${workflowState?.step_index ?? 0} / ${workflowState?.step_order?.length ?? 7}` : `Step ${workflowState?.step_index ?? 0} / ${workflowState?.step_order?.length ?? 7}`}
                    </p>
                    <p className="text-[11px] font-mono" style={{ color: 'var(--brand)' }}>{stepProgress}%</p>
                  </div>
                  <div className="step-progress">
                    <div className="step-progress-fill" style={{ width: `${stepProgress}%` }} />
                  </div>
                  {workflowState?.round && workflowState.round > 1 && (
                    <p className="text-[11px] mt-1.5" style={{ color: 'var(--text-muted)' }}>
                      {isZh ? `第 ${workflowState.round} 轮迭代` : `Round ${workflowState.round}`}
                    </p>
                  )}
                </div>
              )}

              <div className="divider mx-4" />

              {/* Upload section in sidebar (compact) */}
              <div className="px-3 flex-shrink-0">
                <p className="text-[11px] font-semibold uppercase tracking-wider mb-2 px-2" style={{ color: 'var(--text-muted)' }}>
                  {isZh ? '数据文件' : 'Data Files'}
                </p>
                {(['raw', 'seed', 'description'] as UploadFileKey[]).map((key) => {
                  const fileMeta = upload[key]
                  const labels: Record<UploadFileKey, string> = {
                    raw: isZh ? 'Raw Data' : 'Raw Data',
                    seed: isZh ? 'Seed Data' : 'Seed Data',
                    description: isZh ? '任务描述' : 'Description',
                  }
                  return (
                    <div key={key} className="flex items-center gap-2 px-2 py-1.5 rounded-md mb-0.5 cursor-pointer hover:bg-[var(--bg-hover)] transition-colors" onClick={() => { setActiveStep('upload') }}>
                      <div className="w-5 h-5 rounded flex items-center justify-center flex-shrink-0" style={{ background: fileMeta ? 'var(--brand-glow)' : 'var(--bg-card)', border: `1px solid ${fileMeta ? 'var(--brand-dim)' : 'var(--border)'}` }}>
                        {fileMeta
                          ? <Check className="w-2.5 h-2.5" style={{ color: 'var(--brand)' }} />
                          : <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--border)', display: 'block' }} />
                        }
                      </div>
                      <span className="text-[12px] flex-1 truncate" style={{ color: fileMeta ? 'var(--text-dim)' : 'var(--text-muted)' }}>
                        {fileMeta ? fileMeta.name : labels[key]}
                      </span>
                      {key === 'description' && (
                        <span className="text-[10px]" style={{ color: 'var(--text-placeholder)' }}>
                          {isZh ? '可选' : 'opt'}
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>

              <div className="divider mx-4" />

              {/* Start button */}
              <div className="px-3 pb-3">
                <button
                  type="button"
                  className="btn-primary w-full"
                  onClick={openStartModal}
                >
                  <Play className="w-3.5 h-3.5" />
                  {isZh ? '启动 Pipeline' : 'Start Pipeline'}
                </button>
                {pipelineId && canRunFull && (
                  <button
                    type="button"
                    className="w-full mt-2 h-8 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
                    style={{ background: 'var(--success-dim)', color: 'var(--success)', border: '1px solid var(--success)' }}
                    onClick={() => void runFullData()}
                    disabled={workflowBusy}
                  >
                    {workflowBusy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Zap className="w-3.5 h-3.5" />}
                    {isZh ? '应用全量执行' : 'Apply Full Run'}
                  </button>
                )}
              </div>

              {/* Spacer */}
              <div className="flex-1" />

              {/* Operator count */}
              {operatorPool.length > 0 && (
                <div className="px-4 py-3 border-t" style={{ borderColor: 'var(--border-soft)' }}>
                  <div className="flex items-center justify-between">
                    <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{isZh ? '算子池' : 'Operator Pool'}</p>
                    <span className="pill pill-brand text-[10px]">{operatorPool.length}</span>
                  </div>
                </div>
              )}
            </div>
          </nav>

          {/* Main content area */}
          <main className="flex-1 min-w-0 overflow-auto" style={{ background: 'var(--bg-deep)' }}>
            {/* Step 1: Data Setup */}
            {activeStep === 'upload' && (
              <div className="p-6 max-w-2xl mx-auto">
                <div className="mb-6">
                  <h1 className="text-2xl mb-1" style={{ fontFamily: 'Lora, serif', color: 'var(--text)' }}>
                    {isZh ? '数据准备' : 'Data Setup'}
                  </h1>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                    {isZh
                      ? '上传原始数据（Raw）和高质量示例（Seed），系统将自动学习目标数据格式并生成 Pipeline。'
                      : 'Upload raw data and high-quality examples (seed). The system learns the target format and generates a pipeline automatically.'}
                  </p>
                </div>

                {/* Modality guide */}
                <div className="notion-card mb-4" style={{ background: 'var(--brand-glow)', borderColor: 'var(--brand-dim)' }}>
                  <p className="text-xs font-medium mb-2" style={{ color: 'var(--brand-text)' }}>
                    {isZh ? '✦ 支持的数据模态' : '✦ Supported modalities'}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <span className="pill pill-info"><FileJson2 className="w-3 h-3" />{isZh ? '纯文本 / JSONL' : 'Text / JSONL'}</span>
                    <span className="pill pill-image"><Image className="w-3 h-3" />{isZh ? '图像（image_path）' : 'Image (image_path)'}</span>
                    <span className="pill pill-doc"><FileText className="w-3 h-3" />{isZh ? '文档 PDF（pdf_path）' : 'Document PDF (pdf_path)'}</span>
                    <span className="pill pill-warn"><FileJson2 className="w-3 h-3" />{isZh ? '图文混合 QA' : 'Multimodal QA'}</span>
                  </div>
                </div>

                <div className="space-y-3">
                  {renderUploadCard(
                    isZh ? 'Raw Data（必填）' : 'Raw Data (required)',
                    isZh ? '待处理的原始训练数据，质量较低或格式不完整' : 'Raw training data with low quality or incomplete format',
                    'raw',
                    <Database className="w-4 h-4" style={{ color: 'var(--brand)' }} />,
                    rawInputRef
                  )}
                  {renderUploadCard(
                    isZh ? 'Seed Data（必填）' : 'Seed Data (required)',
                    isZh ? '少量高质量示例，展示期望的输出格式和质量' : 'A few high-quality examples showing expected output format',
                    'seed',
                    <Sparkles className="w-4 h-4" style={{ color: 'var(--brand)' }} />,
                    seedInputRef
                  )}
                  {renderUploadCard(
                    isZh ? '任务描述（可选）' : 'Description (optional)',
                    isZh ? '用自然语言描述数据准备目标，帮助系统更好理解任务' : 'Natural language description of the data preparation goal',
                    'description',
                    <FileText className="w-4 h-4" style={{ color: 'var(--brand)' }} />,
                    descInputRef
                  )}
                </div>

                {(upload.raw || upload.seed) && (
                  <button
                    type="button"
                    className="btn-primary w-full mt-5 h-11 text-sm"
                    onClick={openStartModal}
                  >
                    <Play className="w-4 h-4" />
                    {isZh ? '启动 Pipeline 自进化' : 'Start Pipeline Evolution'}
                    <ChevronRight className="w-4 h-4 ml-auto" />
                  </button>
                )}
              </div>
            )}

            {/* Step 2: Pipeline Evolution */}
            {activeStep === 'pipeline' && (
              <div className="h-full">
                <EvolutionCanvas
                  t={canvasT}
                  rows={rows}
                  operatorPool={operatorPool}
                  totalMetrics={totalMetrics}
                  finished={finished}
                  pipelineId={pipelineId}
                  liveArtifacts={liveArtifacts}
                  workflowBusy={workflowBusy}
                  workflowExecutingStep={workflowExecutingStep}
                  canRunFull={canRunFull}
                  workflowState={workflowState}
                  artifactHistoryCount={artifactHistoryCount}
                  uploadPresence={{
                    raw: Boolean(upload.raw || uploadPreview.raw.length),
                    seed: Boolean(upload.seed || uploadPreview.seed.length),
                    description: Boolean(upload.description || uploadPreview.description.length),
                  }}
                  onStepForward={stepForward}
                  onAutoCompleteRound={autoCompleteCurrentRound}
                  onSelectDagTab={selectDagTab}
                  onRunFullData={runFullData}
                  onRerunFromCurrentStep={rerunFromCurrentStep}
                />
              </div>
            )}

            {/* Step 3: Full Run */}
            {activeStep === 'run' && (
              <div className="p-6 max-w-2xl mx-auto">
                <div className="mb-6">
                  <h1 className="text-2xl mb-1" style={{ fontFamily: 'Lora, serif', color: 'var(--text)' }}>
                    {isZh ? '全量执行' : 'Full Run'}
                  </h1>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                    {isZh ? '将生成的 pipeline 应用到全部数据。' : 'Apply the generated pipeline to the full dataset.'}
                  </p>
                </div>

                {canRunFull ? (
                  <div className="notion-card" style={{ background: 'var(--success-dim)', borderColor: 'var(--success)', borderWidth: 1 }}>
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0" style={{ background: 'var(--success-dim)', border: '1px solid var(--success)' }}>
                        <Check className="w-5 h-5" style={{ color: 'var(--success)' }} />
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-semibold" style={{ color: 'var(--success)' }}>
                          {isZh ? 'Pipeline 质检通过，可执行全量' : 'Pipeline passed quality check'}
                        </p>
                        <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                          {isZh ? `Pipeline ID: ${pipelineId}` : `Pipeline ID: ${pipelineId}`}
                        </p>
                        {workflowState?.round && workflowState.round > 1 && (
                          <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                            {isZh ? `经过 ${workflowState.round} 轮迭代优化` : `After ${workflowState.round} rounds of optimization`}
                          </p>
                        )}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="btn-primary w-full mt-4 h-11"
                      onClick={() => void runFullData()}
                      disabled={workflowBusy}
                    >
                      {workflowBusy
                        ? <><Loader2 className="w-4 h-4 animate-spin" />{isZh ? '执行中...' : 'Running...'}</>
                        : <><Zap className="w-4 h-4" />{isZh ? '开始全量执行' : 'Start Full Run'}</>
                      }
                    </button>
                  </div>
                ) : (
                  <div className="notion-card text-center py-12">
                    <div className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-3" style={{ background: 'var(--bg-hover)', border: '1px solid var(--border)' }}>
                      <RotateCcw className="w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                    </div>
                    <p className="text-sm font-medium" style={{ color: 'var(--text-dim)' }}>
                      {isZh ? '请先完成 Pipeline 自进化' : 'Complete pipeline evolution first'}
                    </p>
                    <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                      {isZh ? '在"Pipeline 自进化"步骤中推进流程至质检通过' : 'Advance the workflow until quality check passes'}
                    </p>
                    <button type="button" className="btn-secondary mt-4" onClick={() => setActiveStep('pipeline')}>
                      {isZh ? '前往 Pipeline 画布' : 'Go to Pipeline Canvas'}
                    </button>
                  </div>
                )}
              </div>
            )}
          </main>
        </div>
      </div>

      {/* ── Hidden inputs ── */}
      <input ref={sessionInputRef} type="file" accept=".json" className="hidden" onChange={handleLoadSession} />

      {/* ── Preview Modal ── */}
      {previewModalOpen && (
        <div className="modal-backdrop" onClick={() => setPreviewModalOpen(false)}>
          <div className="modal-panel w-[min(96vw,1100px)] h-[82vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between gap-3 px-5 py-3.5 border-b" style={{ borderColor: 'var(--border-soft)' }}>
              <div>
                <p className="text-sm font-semibold" style={{ color: 'var(--text)', fontFamily: 'Lora, serif' }}>
                  {isZh ? '数据预览' : 'Data Preview'}
                </p>
                <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  {isZh ? '展示上传文件的前几行内容' : 'Shows the first lines of uploaded files'}
                </p>
              </div>
              <button type="button" className="btn-ghost" onClick={() => setPreviewModalOpen(false)}>
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="px-5 pt-3 flex gap-2">
              {previewTabs.map((key) => (
                <button
                  key={`preview-tab-${key}`}
                  type="button"
                  onClick={() => setPreviewModalTab(key)}
                  className="h-8 px-3 rounded-md text-xs border transition-colors"
                  style={{
                    borderColor: previewModalTab === key ? 'var(--brand)' : 'var(--border)',
                    background: previewModalTab === key ? 'var(--brand-glow)' : 'var(--bg-card)',
                    color: previewModalTab === key ? 'var(--brand-text)' : 'var(--text-dim)',
                  }}
                >
                  {key === 'raw' ? 'Raw Data' : key === 'seed' ? 'Seed Data' : (isZh ? '任务描述' : 'Description')}
                </button>
              ))}
            </div>
            <div className="px-5 pb-5 pt-3 flex-1 min-h-0">
              <div className="h-full rounded-xl border overflow-hidden" style={{ borderColor: 'var(--code-border)', background: 'var(--code-bg)' }}>
                <div className="px-4 py-2 border-b text-xs" style={{ borderColor: 'var(--code-border)', background: 'var(--bg-card)', color: 'var(--text-dim)' }}>
                  {upload[previewModalTab]?.name ?? (isZh ? '暂无文件' : 'No file')}
                </div>
                <div className="preview-lines-scroll h-full overflow-auto px-4 py-3">
                  {(previewDisplayLines.length > 0 ? previewDisplayLines : [isZh ? '暂无预览内容' : 'No preview available']).map((line, idx) => (
                    <p key={`preview-modal-line-${previewModalTab}-${idx}`} className="m-0 text-xs font-mono leading-relaxed break-words whitespace-pre-wrap mb-1.5 last:mb-0" style={{ color: 'var(--code-text)' }}>
                      {renderLineWithJsonKeyHighlight(line, `preview-modal-${previewModalTab}-${idx}`)}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Settings Modal ── */}
      {settingsOpen && (
        <div className="modal-backdrop" onClick={() => setSettingsOpen(false)}>
          <div className="modal-panel w-full max-w-3xl h-[68vh] grid" style={{ gridTemplateColumns: '200px 1fr' }} onClick={(e) => e.stopPropagation()}>
            {/* Settings sidebar */}
            <div className="border-r p-4 flex flex-col gap-1" style={{ borderColor: 'var(--border-soft)', background: 'var(--bg-card)' }}>
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm font-semibold" style={{ fontFamily: 'Lora, serif', color: 'var(--text)' }}>
                  {isZh ? '设置' : 'Settings'}
                </p>
                <button type="button" className="btn-ghost" onClick={() => setSettingsOpen(false)}>
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
              {[
                { id: 'general', label: isZh ? '常规' : 'General' },
                { id: 'model', label: isZh ? '模型配置' : 'Model Config' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  className={`step-nav-item ${settingsTab === tab.id ? 'active' : ''}`}
                  onClick={() => setSettingsTab(tab.id as SettingsTab)}
                >
                  <span className="text-sm">{tab.label}</span>
                </button>
              ))}
            </div>

            {/* Settings content */}
            <div className="p-5 overflow-auto">
              {settingsTab === 'general' ? (
                <div className="space-y-5 max-w-md">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                      {isZh ? '外观' : 'Appearance'}
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        { value: 'light', label: isZh ? '☀️ 浅色' : '☀️ Light' },
                        { value: 'dark', label: isZh ? '🌙 深色' : '🌙 Dark' },
                      ].map((opt) => (
                        <button key={opt.value} type="button" onClick={() => setThemeMode(opt.value as 'light' | 'dark')}
                          className="h-10 rounded-lg border text-sm transition-colors"
                          style={{ borderColor: themeMode === opt.value ? 'var(--brand)' : 'var(--border)', background: themeMode === opt.value ? 'var(--brand-glow)' : 'var(--bg-card)', color: themeMode === opt.value ? 'var(--text)' : 'var(--text-dim)' }}>
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wider mb-3 flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }}>
                      <Languages className="w-3.5 h-3.5" />
                      {isZh ? '语言' : 'Language'}
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        { value: 'zh', label: '🇨🇳 中文' },
                        { value: 'en', label: '🇺🇸 English' },
                      ].map((opt) => (
                        <button key={opt.value} type="button" onClick={() => setLanguage(opt.value as 'zh' | 'en')}
                          className="h-10 rounded-lg border text-sm transition-colors"
                          style={{ borderColor: language === opt.value ? 'var(--brand)' : 'var(--border)', background: language === opt.value ? 'var(--brand-glow)' : 'var(--bg-card)', color: language === opt.value ? 'var(--text)' : 'var(--text-dim)' }}>
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-4 max-w-md">
                  <div>
                    <label className="text-xs font-medium uppercase tracking-wider block mb-2" style={{ color: 'var(--text-muted)' }}>
                      {isZh ? '服务商' : 'Provider'}
                    </label>
                    <select value={providerMode} onChange={(e) => setProviderMode(e.target.value as ProviderMode)} className="notion-select">
                      <option value="openai-official">{isZh ? 'OpenAI 官网' : 'OpenAI Official'}</option>
                      <option value="third-party">{isZh ? '第三方服务商' : 'Third-party Provider'}</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium uppercase tracking-wider block mb-2" style={{ color: 'var(--text-muted)' }}>{isZh ? '模型' : 'Model'}</label>
                    <select value={llmConfig.model} onChange={(e) => setLlmConfig({ model: e.target.value })} className="notion-select">
                      {!(MODEL_OPTIONS as readonly string[]).includes(llmConfig.model) && <option value={llmConfig.model}>{llmConfig.model}</option>}
                      {MODEL_OPTIONS.map((m) => <option key={m} value={m}>{m}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium uppercase tracking-wider block mb-2" style={{ color: 'var(--text-muted)' }}>API Key</label>
                    <input type="password" value={apiKeyDraft} onChange={(e) => setApiKeyDraft(e.target.value)} className="notion-input" placeholder="sk-..." />
                    {apiKeyDraft && <p className="text-[11px] mt-1" style={{ color: 'var(--text-muted)' }}>{isZh ? 'API Key 已保存在本地' : 'API Key stored locally'}</p>}
                    {!apiKeyDraft && serverKeyHint?.configured && (
                      <p className="text-[11px] mt-1" style={{ color: 'var(--text-muted)' }}>
                        {isZh ? '服务端已保存密钥' : 'Server has a key configured'}
                        {serverKeyHint.masked ? <span className="font-mono"> {serverKeyHint.masked}</span> : null}
                      </p>
                    )}
                  </div>
                  {providerMode === 'third-party' && (
                    <div>
                      <label className="text-xs font-medium uppercase tracking-wider block mb-2" style={{ color: 'var(--text-muted)' }}>Base URL</label>
                      <input value={baseUrlDraft} onChange={(e) => setBaseUrlDraft(e.target.value)} className="notion-input" placeholder="https://api.example.com/v1" />
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-xs font-medium uppercase tracking-wider block mb-2" style={{ color: 'var(--text-muted)' }}>Temperature</label>
                      <div className="flex items-center gap-2">
                        <input type="range" min={0} max={2} step={0.1} value={llmConfig.temperature} onChange={(e) => setLlmConfig({ temperature: Number(e.target.value) || 0 })} className="flex-1 accent-[var(--brand)]" />
                        <input type="number" min={0} max={2} step={0.1} value={llmConfig.temperature} onChange={(e) => setLlmConfig({ temperature: Number(e.target.value) || 0 })} className="notion-input w-16 text-center px-1" />
                      </div>
                    </div>
                    <div>
                      <label className="text-xs font-medium uppercase tracking-wider block mb-2" style={{ color: 'var(--text-muted)' }}>Max Tokens</label>
                      <div className="flex items-center gap-2">
                        <input type="range" min={256} max={8192} step={256} value={llmConfig.maxTokens} onChange={(e) => setLlmConfig({ maxTokens: Number(e.target.value) || 256 })} className="flex-1 accent-[var(--brand)]" />
                        <input type="number" min={256} max={8192} step={256} value={llmConfig.maxTokens} onChange={(e) => setLlmConfig({ maxTokens: Number(e.target.value) || 256 })} className="notion-input w-20 text-center px-1" />
                      </div>
                    </div>
                  </div>
                  <button type="button" onClick={() => void handleSaveLlmConfig()} disabled={configSaving} className="btn-secondary flex items-center gap-2">
                    {configSaving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                    {configSaving ? (isZh ? '保存中...' : 'Saving...') : (isZh ? '保存到项目' : 'Save to project')}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Start Session Modal ── */}
      {startModalOpen && (
        <div className="modal-backdrop" onClick={() => setStartModalOpen(false)}>
          <div className="modal-panel w-full max-w-sm p-5" onClick={(e) => e.stopPropagation()}>
            <p className="text-base font-semibold mb-1" style={{ fontFamily: 'Lora, serif', color: 'var(--text)' }}>
              {isZh ? '启动 Pipeline' : 'Start Pipeline'}
            </p>
            <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
              {isZh ? '设置 Pipeline ID 并写入清单，启动后进入流程推进。' : 'Set a pipeline ID, then advance the workflow.'}
            </p>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-dim)' }}>
                  Pipeline ID <span style={{ color: 'var(--error)' }}>*</span>
                </label>
                <input type="text" value={pipelineIdInput} onChange={(e) => setPipelineIdInput(e.target.value)} placeholder={isZh ? '例如 my_pipeline_001' : 'e.g. my_pipeline_001'} className="notion-input" autoComplete="off" />
              </div>
              {[
                { key: 'domain', value: metaDomainInput, setter: setMetaDomainInput, label: isZh ? '领域（可选）' : 'Domain (optional)' },
                { key: 'task', value: metaTaskTypeInput, setter: setMetaTaskTypeInput, label: isZh ? '任务类型（可选）' : 'Task type (optional)' },
                { key: 'lang', value: metaLanguageInput, setter: setMetaLanguageInput, label: isZh ? '语言（可选）' : 'Language (optional)' },
              ].map((field) => (
                <div key={field.key}>
                  <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-dim)' }}>{field.label}</label>
                  <input type="text" value={field.value} onChange={(e) => field.setter(e.target.value)} className="notion-input" />
                </div>
              ))}
            </div>
            <div className="flex gap-2 mt-5">
              <button type="button" className="btn-secondary flex-1" disabled={startSubmitting} onClick={() => setStartModalOpen(false)}>
                {isZh ? '取消' : 'Cancel'}
              </button>
              <button type="button" className="btn-primary flex-1" disabled={startSubmitting} onClick={() => void submitStartSession()}>
                {startSubmitting
                  ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />{isZh ? '上传中...' : 'Uploading...'}</>
                  : <><Play className="w-3.5 h-3.5" />{isZh ? '确认并启动' : 'Confirm & Start'}</>
                }
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Clear Confirm Modal ── */}
      {clearConfirmOpen && (
        <div className="modal-backdrop" onClick={() => setClearConfirmOpen(false)}>
          <div className="modal-panel w-full max-w-sm p-5" onClick={(e) => e.stopPropagation()}>
            <p className="text-base font-semibold mb-2" style={{ fontFamily: 'Lora, serif', color: 'var(--text)' }}>
              {isZh ? '清除工作区' : 'Clear Workspace'}
            </p>
            <p className="text-sm mb-5" style={{ color: 'var(--text-dim)' }}>
              {isZh ? '确认清除本轮上传文件与中间工作区内容？此操作不可撤销。' : 'Clear all uploaded files and workspace data? This cannot be undone.'}
            </p>
            <div className="flex gap-2">
              <button type="button" className="btn-secondary flex-1" onClick={() => setClearConfirmOpen(false)}>
                {isZh ? '取消' : 'Cancel'}
              </button>
              <button type="button" className="flex-1 h-9 rounded-lg text-sm font-medium" style={{ background: 'var(--error)', color: 'white' }}
                onClick={() => { setClearConfirmOpen(false); handleClearWorkspace() }}>
                {isZh ? '确认清除' : 'Confirm Clear'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}