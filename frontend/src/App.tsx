import React, { useState, useEffect, useRef } from 'react';
import {
  Sparkles,
  Bot,
  User,
  Send,
  Plus,
  Trash2,
  Copy,
  Check,
  Download,
  RefreshCw,
  Cpu,
  Layers,
  ChevronDown,
  ChevronUp,
  PanelLeftClose,
  PanelLeft,
  Columns3,
  X,
  Clock,
  Settings2,
  AlertCircle
} from 'lucide-react';

interface HardwareInfo {
  vram: {
    total_mb: number;
    used_mb: number;
    free_mb: number;
    gpu_name: string;
  };
  current_loaded_model: string | null;
}

interface ModelInfo {
  name: string;
  available: boolean;
  size_mb: number;
  path: string;
}

interface SystemStatus {
  hardware: HardwareInfo;
  models: {
    gemma: ModelInfo;
    flan_t5: ModelInfo;
    humanizer: ModelInfo;
  };
  model1?: {
    provider: 'gemini' | 'gemma';
    name: string;
    cloud: boolean;
    configured: boolean;
  };
  currently_loaded_model: string | null;
}

interface StageOutput {
  stage: string;
  model: string;
  content: string;
  inference_time_sec: number;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  stages?: {
    gemma?: StageOutput;
    flan_t5?: StageOutput;
    humanizer?: StageOutput;
    totalTime?: number;
  };
  isLoading?: boolean;
  loadingStep?: string;
  activeStageIndex?: 1 | 2 | 3;
  elapsedSec?: number;
  error?: string;
}

interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  messages: ChatMessage[];
}

const DEFAULT_SUGGESTIONS = [
  {
    title: 'Explain Quantum Computing',
    desc: 'Break it down into simple concepts and everyday analogies',
    prompt: 'Explain the principles of quantum computing in simple terms, highlighting qubits, superposition, and practical applications.'
  },
  {
    title: 'Rise of Artificial Intelligence',
    desc: 'Generate an insightful overview of modern AI development',
    prompt: 'Write an engaging piece on the recent rapid rise of artificial intelligence, its impact on daily life, and what the future holds.'
  },
  {
    title: 'Habits for Peak Focus',
    desc: 'Proven routines and science-backed techniques',
    prompt: 'What are 5 actionable, science-backed habits to maximize mental focus, energy, and daily productivity?'
  },
  {
    title: 'Space Exploration Challenges',
    desc: 'Key hurdles in reaching Mars and beyond',
    prompt: 'What are the biggest engineering and biological hurdles humanity must overcome to establish a permanent presence on Mars?'
  }
];

/** Render the predictable Markdown emitted by the local models without
 * injecting HTML from model output. */
function renderInline(text: string) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={index} className="content-inline-code">{part.slice(1, -1)}</code>;
    }
    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}

function FormattedContent({ content }: { content: string }) {
  const blocks: React.ReactNode[] = [];
  let paragraphLines: string[] = [];
  let listItems: string[] = [];
  let listType: 'ul' | 'ol' | null = null;

  const flushParagraph = () => {
    if (paragraphLines.length) {
      blocks.push(<p key={`p-${blocks.length}`} className="content-p">{renderInline(paragraphLines.join(' '))}</p>);
      paragraphLines = [];
    }
  };
  const flushList = () => {
    if (!listType || !listItems.length) return;
    const List = listType;
    blocks.push(
      <List key={`list-${blocks.length}`} className={`content-list ${listType === 'ol' ? 'ordered' : ''}`}>
        {listItems.map((item, index) => <li key={index}>{renderInline(item)}</li>)}
      </List>
    );
    listItems = [];
    listType = null;
  };

  for (const line of content.replace(/\r\n/g, '\n').split('\n')) {
    const trimmed = line.trim();
    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    const bullet = trimmed.match(/^[-*+]\s+(.+)$/);
    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);

    if (!trimmed) {
      flushParagraph();
      flushList();
    } else if (heading) {
      flushParagraph();
      flushList();
      const Heading = heading[1].length === 1 ? 'h2' : heading[1].length === 2 ? 'h3' : 'h4';
      blocks.push(<Heading key={`h-${blocks.length}`} className={`content-heading level-${heading[1].length}`}>{renderInline(heading[2])}</Heading>);
    } else if (bullet || ordered) {
      flushParagraph();
      const nextListType = bullet ? 'ul' : 'ol';
      if (listType && listType !== nextListType) flushList();
      listType = nextListType;
      listItems.push((bullet || ordered)![1]);
    } else if (trimmed.startsWith('> ')) {
      flushParagraph();
      flushList();
      blocks.push(<blockquote key={`quote-${blocks.length}`} className="content-quote">{renderInline(trimmed.slice(2))}</blockquote>);
    } else if (/^(-{3,}|\*{3,})$/.test(trimmed)) {
      flushParagraph();
      flushList();
      blocks.push(<hr key={`rule-${blocks.length}`} className="content-rule" />);
    } else {
      paragraphLines.push(trimmed);
    }
  }
  flushParagraph();
  flushList();
  return <div className="message-text-body">{blocks}</div>;
}

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    try {
      const saved = localStorage.getItem('localai_chat_sessions');
      if (saved) {
        const parsed: ChatSession[] = JSON.parse(saved);
        // Clear any stuck loading state from previous runs
        return parsed.map((s) => ({
          ...s,
          messages: s.messages.map((m) =>
            m.isLoading ? { ...m, isLoading: false, error: 'Previous request was canceled or reset.' } : m
          )
        }));
      }
    } catch (e) {
      console.error(e);
    }
    return [
      {
        id: 'session-1',
        title: 'New Conversation',
        createdAt: new Date().toISOString(),
        messages: []
      }
    ];
  });

  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    return sessions[0]?.id || 'session-1';
  });

  const [inputPrompt, setInputPrompt] = useState('');
  const [contentType, setContentType] = useState<'Chat' | 'Blog' | 'Article' | 'Essay' | 'Social Media'>('Chat');
  const [contentLength, setContentLength] = useState<'Short' | 'Medium' | 'Long'>('Medium');
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const stage1ModelName = status?.model1?.name || 'Gemma 3 4B';

  // Expanded reasoning/pipeline stage drawers per message
  const [expandedStages, setExpandedStages] = useState<Record<string, boolean>>({});
  const [activeStageTab, setActiveStageTab] = useState<Record<string, 'gemma' | 'flan' | 'humanizer'>>({});

  // Compare Modal State
  const [compareModalMsg, setCompareModalMsg] = useState<ChatMessage | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Active session helper
  const currentSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];

  // Persist sessions
  useEffect(() => {
    try {
      localStorage.setItem('localai_chat_sessions', JSON.stringify(sessions));
    } catch (e) {
      console.error(e);
    }
  }, [sessions]);

  // Fetch Hardware & Model Status
  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (e) {
      console.error('Failed to fetch hardware status:', e);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession?.messages, isGenerating]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [inputPrompt]);

  const handleNewChat = () => {
    const newSession: ChatSession = {
      id: `session-${Date.now()}`,
      title: 'New Conversation',
      createdAt: new Date().toISOString(),
      messages: []
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (sessions.length <= 1) {
      const reset: ChatSession = {
        id: `session-${Date.now()}`,
        title: 'New Conversation',
        createdAt: new Date().toISOString(),
        messages: []
      };
      setSessions([reset]);
      setActiveSessionId(reset.id);
      return;
    }
    const filtered = sessions.filter((s) => s.id !== id);
    setSessions(filtered);
    if (activeSessionId === id) {
      setActiveSessionId(filtered[0].id);
    }
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleDownload = (text: string, filename: string) => {
    const blob = new Blob([text], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `${filename}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleUnloadVRAM = async () => {
    try {
      await fetch('/api/unload', { method: 'POST' });
      await fetchStatus();
    } catch (e) {
      console.error('Error unloading VRAM:', e);
    }
  };

  const toggleStages = (msgId: string) => {
    setExpandedStages((prev) => ({
      ...prev,
      [msgId]: !prev[msgId]
    }));
    if (!activeStageTab[msgId]) {
      setActiveStageTab((prev) => ({ ...prev, [msgId]: 'gemma' }));
    }
  };

  const updateAssistant = (msgId: string, updates: Partial<ChatMessage>) => {
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId) {
          return {
            ...s,
            messages: s.messages.map((m) => (m.id === msgId ? { ...m, ...updates } : m))
          };
        }
        return s;
      })
    );
  };

  const handleSendMessage = async (promptToSend?: string) => {
    const prompt = (promptToSend || inputPrompt).trim();
    if (!prompt || isGenerating) return;

    setInputPrompt('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    const userMsgId = `msg-${Date.now()}-u`;
    const assistantMsgId = `msg-${Date.now()}-a`;

    const userMessage: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: prompt,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const assistantPlaceholder: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isLoading: true,
      activeStageIndex: 1,
      loadingStep: `Stage 1/3: ${stage1ModelName} is generating the initial draft...`
    };

    // Update active session with user message and placeholder
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId) {
          const isFirstMessage = s.messages.length === 0;
          return {
            ...s,
            title: isFirstMessage ? prompt.slice(0, 32) + (prompt.length > 32 ? '...' : '') : s.title,
            messages: [...s.messages, userMessage, assistantPlaceholder]
          };
        }
        return s;
      })
    );

    setIsGenerating(true);
    const pipelineStart = Date.now();

    try {
      // STAGE 1: configured cloud Gemini or local Gemma provider
      updateAssistant(assistantMsgId, {
        isLoading: true,
        activeStageIndex: 1,
        loadingStep: `Stage 1/3: ${stage1ModelName} is generating initial content...`
      });

      const gemmaRes = await fetch('/api/generate/gemma', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          content_type: contentType,
          content_length: contentLength
        })
      });
      const gemmaData = await gemmaRes.json();
      if (!gemmaRes.ok) throw new Error(gemmaData.detail || 'Stage 1 failed');

      // STAGE 2: FLAN-T5 Paraphraser
      updateAssistant(assistantMsgId, {
        isLoading: true,
        activeStageIndex: 2,
        loadingStep: `Stage 2/3: FLAN-T5 Paraphraser is restructuring sentences... (${stage1ModelName} done in ${gemmaData.inference_time_sec}s)`,
        stages: { gemma: gemmaData }
      });

      const flanRes = await fetch('/api/generate/flan-t5', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: gemmaData.content })
      });
      const flanData = await flanRes.json();
      if (!flanRes.ok) throw new Error(flanData.detail || 'Stage 2 (FLAN-T5) failed');

      // STAGE 3: Humanizer 12B
      updateAssistant(assistantMsgId, {
        isLoading: true,
        activeStageIndex: 3,
        loadingStep: `Stage 3/3: Humanizer 12B is naturally polishing tone & style... (FLAN-T5 done in ${flanData.inference_time_sec}s)`,
        stages: { gemma: gemmaData, flan_t5: flanData }
      });

      const humanizerRes = await fetch('/api/generate/humanizer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: flanData.content })
      });
      const humanizerData = await humanizerRes.json();
      if (!humanizerRes.ok) throw new Error(humanizerData.detail || 'Stage 3 (Humanizer) failed');

      const totalSec = parseFloat(((Date.now() - pipelineStart) / 1000).toFixed(2));

      // Pipeline Complete!
      updateAssistant(assistantMsgId, {
        isLoading: false,
        activeStageIndex: undefined,
        loadingStep: undefined,
        content: humanizerData.content,
        stages: {
          gemma: gemmaData,
          flan_t5: flanData,
          humanizer: humanizerData,
          totalTime: totalSec
        }
      });
    } catch (err: any) {
      updateAssistant(assistantMsgId, {
        isLoading: false,
        activeStageIndex: undefined,
        loadingStep: undefined,
        error: err.message || 'Error occurred during generation'
      });
    } finally {
      setIsGenerating(false);
      fetchStatus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chatgpt-container">
      {/* LEFT SIDEBAR (ChatGPT Style) */}
      <aside className={`chatgpt-sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={handleNewChat} title="Start New Chat">
            <Plus size={18} />
            <span>New chat</span>
          </button>
          <button
            className="sidebar-toggle-btn"
            onClick={() => setSidebarOpen(false)}
            title="Close sidebar"
          >
            <PanelLeftClose size={18} />
          </button>
        </div>

        {/* Conversation List */}
        <div className="sidebar-sessions">
          <div className="sidebar-section-title">Recent Conversations</div>
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`session-item ${s.id === activeSessionId ? 'active' : ''}`}
              onClick={() => setActiveSessionId(s.id)}
            >
              <div className="session-title-text" title={s.title}>
                {s.title}
              </div>
              <button
                className="session-delete-btn"
                onClick={(e) => handleDeleteSession(s.id, e)}
                title="Delete chat"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        {/* Hardware Status Card */}
        <div className="sidebar-hardware-card">
          <div className="hardware-header">
            <div className="flex-row items-center gap-2">
              <Cpu size={15} className="text-cyan" />
              <span className="hw-name">{status?.hardware?.vram?.gpu_name || 'RTX 3050 Laptop GPU'}</span>
            </div>
            <button
              onClick={handleUnloadVRAM}
              className="vram-clean-btn"
              title="Unload model and free GPU VRAM"
            >
              Free VRAM
            </button>
          </div>

          <div className="vram-stat-bar">
            <div className="vram-labels">
              <span>VRAM Usage</span>
              <span>
                {status?.hardware?.vram?.used_mb ?? 0} / {status?.hardware?.vram?.total_mb ?? 6144} MB
              </span>
            </div>
            <div className="vram-progress-bg">
              <div
                className="vram-progress-fill"
                style={{
                  width: `${Math.min(
                    100,
                    (((status?.hardware?.vram?.used_mb ?? 0) / (status?.hardware?.vram?.total_mb ?? 6144)) * 100)
                  )}%`
                }}
              />
            </div>
          </div>

          <div className="active-model-indicator">
            <span className="dot online" />
            <span className="text-xs text-muted">
              {status?.currently_loaded_model ? `Loaded: ${status.currently_loaded_model}` : 'VRAM Clean (On-demand)'}
            </span>
          </div>
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <main className="chatgpt-main">
        {/* Top Navbar */}
        <header className="chatgpt-topbar">
          <div className="flex-row items-center gap-3">
            {!sidebarOpen && (
              <button
                className="icon-btn"
                onClick={() => setSidebarOpen(true)}
                title="Open sidebar"
              >
                <PanelLeft size={18} />
              </button>
            )}

            <div className="model-badge-group">
              <div className="pipeline-pill">
                <Sparkles size={15} className="text-indigo" />
                <span className="pill-title">LocalAI Pipeline</span>
                <span className="pill-arrow">→</span>
                <span className="pill-step">{stage1ModelName}</span>
                <span className="pill-arrow">→</span>
                <span className="pill-step">FLAN-T5</span>
                <span className="pill-arrow">→</span>
                <span className="pill-step">Humanizer 12B</span>
              </div>
            </div>
          </div>

          <div className="topbar-actions">
            {/* Settings dropdown toggle */}
            <div className="settings-wrapper">
              <button
                className={`topbar-btn ${showSettings ? 'active' : ''}`}
                onClick={() => setShowSettings(!showSettings)}
                title="Adjust Generation Style & Length"
              >
                <Settings2 size={16} />
                <span>{contentType} ({contentLength})</span>
              </button>

              {showSettings && (
                <div className="settings-popup">
                  <div className="popup-title">Generation Settings</div>
                  <div className="popup-section">
                    <label>Response Style / Type</label>
                    <div className="pills-grid">
                      {(['Chat', 'Blog', 'Article', 'Essay', 'Social Media'] as const).map((t) => (
                        <button
                          key={t}
                          className={`pill-option ${contentType === t ? 'selected' : ''}`}
                          onClick={() => setContentType(t)}
                        >
                          {t}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="popup-section">
                    <label>Response Length</label>
                    <div className="pills-grid">
                      {(['Short', 'Medium', 'Long'] as const).map((l) => (
                        <button
                          key={l}
                          className={`pill-option ${contentLength === l ? 'selected' : ''}`}
                          onClick={() => setContentLength(l)}
                        >
                          {l}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            <button
              className="topbar-btn text-muted hover-red"
              onClick={() => {
                setSessions((prev) =>
                  prev.map((s) => (s.id === activeSessionId ? { ...s, messages: [] } : s))
                );
              }}
              title="Clear messages in this chat"
            >
              <Trash2 size={16} />
              <span>Clear</span>
            </button>
          </div>
        </header>

        {/* Chat Feed */}
        <div className="chat-messages-container">
          {currentSession?.messages?.length === 0 ? (
            /* ChatGPT Welcome Hero */
            <div className="chatgpt-hero">
              <div className="hero-icon-ring">
                <Sparkles size={36} className="text-gradient" />
              </div>
              <h1 className="hero-title">What would you like to explore?</h1>
              <p className="hero-subtitle">
                Ask any question or request content. Your answer is generated by <strong>{stage1ModelName}</strong>,
                restructured by <strong>FLAN-T5 Paraphraser</strong>, and naturally polished by <strong>Humanizer 12B</strong>.
                {status?.model1?.cloud ? ' Stage 1 runs through Gemini cloud.' : ' Running 100% locally on your RTX 3050 GPU.'}
              </p>

              {/* Suggestions Grid */}
              <div className="suggestions-grid">
                {DEFAULT_SUGGESTIONS.map((sug, idx) => (
                  <div
                    key={idx}
                    className="suggestion-card"
                    onClick={() => handleSendMessage(sug.prompt)}
                  >
                    <div className="sug-title">{sug.title}</div>
                    <div className="sug-desc">{sug.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            /* Message Thread */
            <div className="message-stream">
              {currentSession.messages.map((msg) => (
                <div key={msg.id} className={`message-row ${msg.role}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? (
                      <div className="avatar user">
                        <User size={18} />
                      </div>
                    ) : (
                      <div className="avatar assistant">
                        <Bot size={18} />
                      </div>
                    )}
                  </div>

                  <div className="message-bubble-wrapper">
                    <div className="message-author">
                      <span className="author-name">
                        {msg.role === 'user' ? 'You' : `LocalAI (${stage1ModelName} + FLAN-T5 + Humanizer)`}
                      </span>
                      <span className="msg-time">{msg.timestamp}</span>
                    </div>

                    {/* Assistant In-Progress State */}
                    {msg.isLoading && (
                      <div className="assistant-loading-box">
                        <div className="loading-spinner-row">
                          <RefreshCw size={18} className="animate-spin text-cyan" />
                          <span className="loading-label">{msg.loadingStep || 'Executing 3-stage pipeline...'}</span>
                        </div>
                        <div className="loading-steps-pills">
                          <span className={`step-pill ${msg.activeStageIndex === 1 ? 'active pulse' : msg.stages?.gemma ? 'done' : ''}`}>
                            {msg.stages?.gemma ? `✓ ${msg.stages.gemma.model} (${msg.stages.gemma.inference_time_sec}s)` : `1. ${stage1ModelName}`}
                          </span>
                          <span className="step-arrow">→</span>
                          <span className={`step-pill ${msg.activeStageIndex === 2 ? 'active pulse' : msg.stages?.flan_t5 ? 'done' : ''}`}>
                            {msg.stages?.flan_t5 ? `✓ FLAN-T5 (${msg.stages.flan_t5.inference_time_sec}s)` : '2. FLAN-T5 Paraphraser'}
                          </span>
                          <span className="step-arrow">→</span>
                          <span className={`step-pill ${msg.activeStageIndex === 3 ? 'active pulse' : ''}`}>
                            3. Humanizer 12B
                          </span>
                        </div>
                        <button
                          className="cancel-btn"
                          onClick={() => {
                            setIsGenerating(false);
                            updateAssistant(msg.id, { isLoading: false, error: 'Canceled by user.' });
                            handleUnloadVRAM();
                          }}
                        >
                          Cancel / Reset
                        </button>
                      </div>
                    )}

                    {/* Assistant Error State */}
                    {msg.error && (
                      <div className="message-error-box">
                        <AlertCircle size={18} className="text-rose" />
                        <div>
                          <div className="error-title">Generation Error</div>
                          <div className="error-desc">{msg.error}</div>
                        </div>
                      </div>
                    )}

                    {/* Pipeline Thought Accordion (ChatGPT / Reasoning Style) */}
                    {msg.stages && (
                      <div className="stages-accordion">
                        <button
                          className="accordion-header"
                          onClick={() => toggleStages(msg.id)}
                        >
                          <div className="flex-row items-center gap-2">
                            <Layers size={14} className="text-cyan" />
                            <span className="font-semibold text-xs">
                              Pipeline Trace: 3 Stages Completed
                            </span>
                            {msg.stages.totalTime && (
                              <span className="time-badge">
                                <Clock size={11} />
                                {msg.stages.totalTime}s total
                              </span>
                            )}
                          </div>
                          {expandedStages[msg.id] ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </button>

                        {expandedStages[msg.id] && (
                          <div className="accordion-content">
                            <div className="accordion-tabs">
                              <button
                                className={`acc-tab ${activeStageTab[msg.id] === 'gemma' ? 'active' : ''}`}
                                onClick={() => setActiveStageTab((p) => ({ ...p, [msg.id]: 'gemma' }))}
                              >
                                {msg.stages.gemma?.model || stage1ModelName} ({msg.stages.gemma?.inference_time_sec}s)
                              </button>
                              <button
                                className={`acc-tab ${activeStageTab[msg.id] === 'flan' ? 'active' : ''}`}
                                onClick={() => setActiveStageTab((p) => ({ ...p, [msg.id]: 'flan' }))}
                              >
                                FLAN-T5 ({msg.stages.flan_t5?.inference_time_sec}s)
                              </button>
                              <button
                                className={`acc-tab ${activeStageTab[msg.id] === 'humanizer' ? 'active' : ''}`}
                                onClick={() => setActiveStageTab((p) => ({ ...p, [msg.id]: 'humanizer' }))}
                              >
                                Humanizer 12B ({msg.stages.humanizer?.inference_time_sec}s)
                              </button>
                            </div>

                            <div className="stage-text-preview">
                              {activeStageTab[msg.id] === 'gemma' && (
                                <pre className="stage-raw-content">{msg.stages.gemma?.content}</pre>
                              )}
                              {activeStageTab[msg.id] === 'flan' && (
                                <pre className="stage-raw-content">{msg.stages.flan_t5?.content}</pre>
                              )}
                              {activeStageTab[msg.id] === 'humanizer' && (
                                <pre className="stage-raw-content">{msg.stages.humanizer?.content}</pre>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Final Response Content */}
                    {msg.content && (
                      <FormattedContent content={msg.content} />
                    )}

                    {/* Message Action Bar */}
                    {msg.role === 'assistant' && msg.content && (
                      <div className="message-action-bar">
                        <button
                          className="msg-action-btn"
                          onClick={() => handleCopy(msg.content, msg.id)}
                          title="Copy final response"
                        >
                          {copiedId === msg.id ? <Check size={14} className="text-emerald" /> : <Copy size={14} />}
                          <span>{copiedId === msg.id ? 'Copied' : 'Copy'}</span>
                        </button>

                        <button
                          className="msg-action-btn"
                          onClick={() => handleDownload(msg.content, `localai-response-${msg.id}`)}
                          title="Download Markdown"
                        >
                          <Download size={14} />
                          <span>Save .md</span>
                        </button>

                        {msg.stages && (
                          <button
                            className="msg-action-btn"
                            onClick={() => setCompareModalMsg(msg)}
                            title="Compare Stage 1, FLAN-T5, and Humanizer side by side"
                          >
                            <Columns3 size={14} />
                            <span>Compare Stages</span>
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* BOTTOM STICKY INPUT BAR (ChatGPT Style) */}
        <div className="chatgpt-input-container">
          <div className="input-box-wrapper">
            <textarea
              ref={textareaRef}
              rows={1}
              className="chatgpt-textarea"
              placeholder="Ask anything or request content..."
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isGenerating}
            />

            <button
              className={`send-button ${inputPrompt.trim() && !isGenerating ? 'active' : ''}`}
              onClick={() => handleSendMessage()}
              disabled={!inputPrompt.trim() || isGenerating}
              title="Send prompt (Enter)"
            >
              {isGenerating ? (
                <RefreshCw size={17} className="animate-spin text-muted" />
              ) : (
                <Send size={17} />
              )}
            </button>
          </div>

          <div className="chat-footer-disclaimer">
            <span>Runs 100% locally through <strong>llama.cpp</strong> on <strong>NVIDIA RTX 3050 (6GB)</strong>. Zero cloud APIs. AI drafts still need fact-checking, citations, and your own review before publishing or submitting.</span>
          </div>
        </div>
      </main>

      {/* COMPARE STAGES MODAL */}
      {compareModalMsg && compareModalMsg.stages && (
        <div className="modal-backdrop" onClick={() => setCompareModalMsg(null)}>
          <div className="compare-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="flex-row items-center gap-2">
                <Columns3 size={20} className="text-cyan" />
                <h2 className="modal-title">Stage-by-Stage Content Evolution</h2>
              </div>
              <button
                className="modal-close-btn"
                onClick={() => setCompareModalMsg(null)}
              >
                <X size={18} />
              </button>
            </div>

            <div className="compare-grid">
              {/* Stage 1 Column */}
              <div className="compare-column">
                <div className="col-header gemma">
                  <div>
                    <div className="col-title">Stage 1: {compareModalMsg.stages.gemma?.model || stage1ModelName}</div>
                    <div className="col-subtitle">Initial Draft Generation</div>
                  </div>
                  <span className="badge-time">{compareModalMsg.stages.gemma?.inference_time_sec}s</span>
                </div>
                <div className="col-body">
                  {compareModalMsg.stages.gemma?.content}
                </div>
              </div>

              {/* FLAN-T5 Column */}
              <div className="compare-column">
                <div className="col-header flan">
                  <div>
                    <div className="col-title">Stage 2: FLAN-T5</div>
                    <div className="col-subtitle">Restructured & Paraphrased</div>
                  </div>
                  <span className="badge-time">{compareModalMsg.stages.flan_t5?.inference_time_sec}s</span>
                </div>
                <div className="col-body">
                  {compareModalMsg.stages.flan_t5?.content}
                </div>
              </div>

              {/* Humanizer Column */}
              <div className="compare-column">
                <div className="col-header humanizer">
                  <div>
                    <div className="col-title">Stage 3: Humanizer 12B</div>
                    <div className="col-subtitle">Final Polished Content</div>
                  </div>
                  <span className="badge-time">{compareModalMsg.stages.humanizer?.inference_time_sec}s</span>
                </div>
                <div className="col-body">
                  {compareModalMsg.stages.humanizer?.content}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
