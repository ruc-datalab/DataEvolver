import { ArrowRight, Database, FileText, Image, Layers, Sparkles, Zap } from 'lucide-react'
import { useAppStore } from '../stores/appStore'

export function LandingPage({ onEnter }: { onEnter: () => void }) {
  const themeMode = useAppStore((s) => s.themeMode)
  const language = useAppStore((s) => s.language)
  const isZh = language === 'zh'
  const isDark = themeMode === 'dark'

  const features = isZh ? [
    {
      icon: <Database className="w-5 h-5" />,
      title: '多模态数据支持',
      desc: '支持文本、图像、PDF文档等多种数据模态，自动识别数据类型并选择合适算子',
      color: 'var(--brand)',
      bg: 'var(--brand-glow)',
      border: 'var(--brand-dim)',
    },
    {
      icon: <Sparkles className="w-5 h-5" />,
      title: 'Pipeline 自进化',
      desc: '基于 seed 数据自动理解目标格式，通过多轮 LLM 迭代生成并优化数据准备 pipeline',
      color: 'var(--amber)',
      bg: 'var(--amber-glow)',
      border: 'var(--amber-dim)',
    },
    {
      icon: <Layers className="w-5 h-5" />,
      title: '算子级自进化',
      desc: '支持图像处理、文档解析、OCR清洗、VLM问答等专用算子，可自动扩展算子库',
      color: 'var(--mm-image)',
      bg: 'var(--mm-image-dim)',
      border: 'var(--mm-image-dim)',
    },
    {
      icon: <Zap className="w-5 h-5" />,
      title: 'Pilot LLM 评估',
      desc: '每轮试运行后由 LLM Judge 从语义、格式、多样性等维度打分，达标后自动终止迭代',
      color: 'var(--success)',
      bg: 'var(--success-dim)',
      border: 'var(--success-dim)',
    },
  ] : [
    {
      icon: <Database className="w-5 h-5" />,
      title: 'Multimodal Support',
      desc: 'Handles text, images, PDF documents and more. Automatically detects data type and selects appropriate operators.',
      color: 'var(--brand)',
      bg: 'var(--brand-glow)',
      border: 'var(--brand-dim)',
    },
    {
      icon: <Sparkles className="w-5 h-5" />,
      title: 'Pipeline Self-Evolution',
      desc: 'Learns target format from seed data, then generates and refines a data preparation pipeline through LLM iterations.',
      color: 'var(--amber)',
      bg: 'var(--amber-glow)',
      border: 'var(--amber-dim)',
    },
    {
      icon: <Layers className="w-5 h-5" />,
      title: 'Operator Evolution',
      desc: 'Built-in operators for image processing, document parsing, OCR cleaning, VLM QA, and more — with auto-expansion.',
      color: 'var(--mm-image)',
      bg: 'var(--mm-image-dim)',
      border: 'var(--mm-image-dim)',
    },
    {
      icon: <Zap className="w-5 h-5" />,
      title: 'Pilot LLM Judge',
      desc: 'After each trial run, an LLM Judge scores the output on semantic quality, format, diversity, and noise. Stops when target score is reached.',
      color: 'var(--success)',
      bg: 'var(--success-dim)',
      border: 'var(--success-dim)',
    },
  ]

  const scenarios = isZh ? [
    { icon: <Image className="w-4 h-4" />, label: '图像隐私保护', desc: '人脸打码 + 高质量 VLM 问答生成', tag: 'pill-image' },
    { icon: <FileText className="w-4 h-4" />, label: '文档问答准备', desc: 'PDF 解析 → chunks → DocQA 训练数据', tag: 'pill-doc' },
    { icon: <Database className="w-4 h-4" />, label: '电商商品描述', desc: '商品图片 + 多样化问答对生成', tag: 'pill-brand' },
  ] : [
    { icon: <Image className="w-4 h-4" />, label: 'Image Privacy', desc: 'Face blurring + high-quality VLM QA generation', tag: 'pill-image' },
    { icon: <FileText className="w-4 h-4" />, label: 'Document QA', desc: 'PDF → chunks → DocQA training data', tag: 'pill-doc' },
    { icon: <Database className="w-4 h-4" />, label: 'Product Descriptions', desc: 'Product images + diverse QA pair generation', tag: 'pill-brand' },
  ]

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--bg-deep)', color: 'var(--text)' }}>
      {/* Top nav */}
      <header className="flex items-center justify-between px-8 py-4 border-b" style={{ borderColor: 'var(--border-soft)', background: 'var(--bg-panel)' }}>
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl flex items-center justify-center" style={{ background: 'var(--brand-glow)', border: '1px solid var(--brand-dim)' }}>
            <Sparkles className="w-4 h-4" style={{ color: 'var(--brand)' }} />
          </div>
          <div>
            <p className="text-sm font-semibold leading-none" style={{ fontFamily: 'Lora, serif', color: 'var(--text)' }}>DataEvolver 2.0</p>
            <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>Multimodal Data Preparation</p>
          </div>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={onEnter}
        >
          {isZh ? '进入工作台' : 'Open Workspace'}
          <ArrowRight className="w-4 h-4" />
        </button>
      </header>

      {/* Hero */}
      <section className="flex flex-col items-center text-center px-6 pt-20 pb-16">
        <div className="pill pill-brand mb-5">
          <Sparkles className="w-3 h-3" />
          {isZh ? 'DataEvolver 2.0 · 多模态原生版本' : 'DataEvolver 2.0 · Multimodal Native'}
        </div>

        <h1 className="text-5xl leading-tight mb-5 max-w-3xl" style={{ fontFamily: 'Lora, serif', color: 'var(--text)', letterSpacing: '-0.03em' }}>
          {isZh
            ? <>自动化<span style={{ color: 'var(--brand)' }}>多模态</span>数据准备</>
            : <>Automated <span style={{ color: 'var(--brand)' }}>Multimodal</span> Data Preparation</>
          }
        </h1>

        <p className="text-lg max-w-2xl mb-8 leading-relaxed" style={{ color: 'var(--text-muted)' }}>
          {isZh
            ? '上传原始数据和少量高质量示例，DataEvolver 2.0 自动生成、进化并评估数据准备 pipeline，支持图像、文档、文本等多种模态。'
            : 'Upload raw data and a few high-quality examples. DataEvolver 2.0 automatically generates, evolves, and evaluates data preparation pipelines for images, documents, and text.'
          }
        </p>

        <div className="flex items-center gap-3 flex-wrap justify-center">
          <button type="button" className="btn-primary h-11 px-6 text-sm" onClick={onEnter}>
            <Play className="w-4 h-4" />
            {isZh ? '立即开始' : 'Get Started'}
          </button>
          <a
            href="https://github.com/modelscope/AgentEvolver"
            target="_blank"
            rel="noreferrer"
            className="btn-secondary h-11 px-6 text-sm"
          >
            {isZh ? '查看论文' : 'Read Paper'}
          </a>
        </div>

        {/* Data type badges */}
        <div className="flex items-center gap-2 mt-8 flex-wrap justify-center">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{isZh ? '支持：' : 'Supports:'}</span>
          <span className="pill pill-info"><Database className="w-3 h-3" />{isZh ? '纯文本 JSONL' : 'Text JSONL'}</span>
          <span className="pill pill-image"><Image className="w-3 h-3" />{isZh ? '图像数据' : 'Image Data'}</span>
          <span className="pill pill-doc"><FileText className="w-3 h-3" />{isZh ? 'PDF 文档' : 'PDF Documents'}</span>
          <span className="pill pill-warn"><Layers className="w-3 h-3" />{isZh ? '图文混合' : 'Multimodal QA'}</span>
        </div>
      </section>

      {/* Feature cards */}
      <section className="px-8 pb-16 max-w-5xl mx-auto w-full">
        <div className="grid grid-cols-2 gap-4">
          {features.map((f, i) => (
            <div key={i} className="notion-card slide-in" style={{ animationDelay: `${i * 80}ms` }}>
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0" style={{ background: f.bg, border: `1px solid ${f.border}`, color: f.color }}>
                  {f.icon}
                </div>
                <div>
                  <p className="text-sm font-semibold mb-1" style={{ color: 'var(--text)', fontFamily: 'Lora, serif' }}>{f.title}</p>
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>{f.desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Scenarios */}
      <section className="px-8 pb-16 max-w-5xl mx-auto w-full">
        <p className="text-xs font-semibold uppercase tracking-wider mb-4" style={{ color: 'var(--text-muted)' }}>
          {isZh ? '典型应用场景' : 'Example Scenarios'}
        </p>
        <div className="grid grid-cols-3 gap-3">
          {scenarios.map((s, i) => (
            <div key={i} className="notion-card flex items-start gap-3">
              <div className={`pill ${s.tag} flex-shrink-0 mt-0.5`}>{s.icon}</div>
              <div>
                <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>{s.label}</p>
                <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="px-8 pb-20 max-w-5xl mx-auto w-full">
        <p className="text-xs font-semibold uppercase tracking-wider mb-5" style={{ color: 'var(--text-muted)' }}>
          {isZh ? '工作流程' : 'How It Works'}
        </p>
        <div className="flex items-start gap-3">
          {(isZh ? [
            { n: '01', t: '上传数据', d: '上传 Raw Data 和 Seed Data，可选任务描述文件' },
            { n: '02', t: '理解 & 编排', d: 'LLM 分析数据差距，自动生成 Pipeline DAG' },
            { n: '03', t: '实例化 & 试运行', d: '生成算子代码，采样执行，Pilot Judge 打分' },
            { n: '04', t: '迭代优化', d: '未达标则写入经验，回流下一轮理解，直到通过' },
            { n: '05', t: '全量执行', d: '质检通过后对全量数据执行 pipeline，导出结果' },
          ] : [
            { n: '01', t: 'Upload Data', d: 'Upload Raw Data and Seed Data, optional description' },
            { n: '02', t: 'Understand & Orchestrate', d: 'LLM analyzes data gap, generates pipeline DAG' },
            { n: '03', t: 'Instantiate & Trial Run', d: 'Generates operator code, samples data, Pilot Judge scores' },
            { n: '04', t: 'Iterate', d: 'If score too low, writes experience and loops back to understanding' },
            { n: '05', t: 'Full Run', d: 'After quality check passes, apply pipeline to all data' },
          ]).map((step, i, arr) => (
            <div key={i} className="flex items-start gap-2 flex-1">
              <div className="flex flex-col items-center">
                <div className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 font-mono text-xs font-bold" style={{ background: 'var(--brand-glow)', border: '1px solid var(--brand-dim)', color: 'var(--brand-text)' }}>
                  {step.n}
                </div>
                {i < arr.length - 1 && (
                  <div className="w-px flex-1 mt-1" style={{ background: 'var(--border-soft)', minHeight: 20 }} />
                )}
              </div>
              <div className="pb-4">
                <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>{step.t}</p>
                <p className="text-xs mt-0.5 leading-relaxed" style={{ color: 'var(--text-muted)' }}>{step.d}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="border-t py-12 text-center" style={{ borderColor: 'var(--border-soft)', background: isDark ? 'var(--bg-panel)' : 'var(--bg-card)' }}>
        <p className="text-xl font-semibold mb-2" style={{ fontFamily: 'Lora, serif', color: 'var(--text)' }}>
          {isZh ? '准备好了吗？' : 'Ready to start?'}
        </p>
        <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
          {isZh ? '上传你的数据，让系统自动完成 pipeline 设计。' : 'Upload your data and let the system design the pipeline automatically.'}
        </p>
        <button type="button" className="btn-primary h-11 px-8 text-sm" onClick={onEnter}>
          {isZh ? '进入工作台' : 'Open Workspace'}
          <ArrowRight className="w-4 h-4" />
        </button>
      </section>

      <footer className="py-4 text-center border-t" style={{ borderColor: 'var(--border-soft)' }}>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          DataEvolver 2.0 · Multimodal Data Preparation for VLM SFT · Built with FastAPI + React
        </p>
      </footer>
    </div>
  )
}

// Play icon inline to avoid import issue
function Play({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5v14l11-7z" />
    </svg>
  )
}