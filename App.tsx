
import React, { useState, useCallback } from 'react';
import { PromptMode, ShoeType, PromptResult } from './types';
import { generateSoraPrompt } from './services/geminiService';
import PromptDisplay from './components/PromptDisplay';

const SHOE_TYPES: ShoeType[] = ['sneaker', 'runner', 'leather', 'casual', 'sandals', 'boots', 'luxury'];
const TONES = ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"];

const App: React.FC = () => {
  const [idea, setIdea] = useState('');
  const [shoeType, setShoeType] = useState<ShoeType>('sneaker');
  const [mode, setMode] = useState<PromptMode>(PromptMode.CAMEO);
  const [tone, setTone] = useState(TONES[0]);
  const [count, setCount] = useState(5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<PromptResult[]>([]);
  const [activeTab, setActiveTab] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!idea.trim()) return;
    setLoading(true);
    setError(null);
    setResults([]);
    
    try {
      const promises = Array.from({ length: count }).map(() => 
        generateSoraPrompt(idea, shoeType, mode, tone)
      );
      const data = await Promise.all(promises);
      setResults(data);
      setActiveTab(0);
    } catch (err) {
      setError("Lỗi hệ thống studio. Vui lòng kiểm tra API Key.");
    } finally {
      setLoading(false);
    }
  };

  const resetMemory = () => {
    setResults([]);
    setError(null);
    // Trong bản web này, logic chống trùng được xử lý bởi tính ngẫu nhiên cao của Gemini 3 Flash
    alert("Đã reset bộ nhớ studio!");
  };

  return (
    <div className="min-h-screen p-4 md:p-10 flex flex-col items-center bg-[#020617] text-slate-200">
      <header className="max-w-4xl w-full text-center mb-12 animate-in fade-in slide-in-from-top-4 duration-700">
        <div className="inline-flex items-center gap-2 mb-4 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-bold uppercase tracking-widest">
          Director Edition • Timeline Lock 10s
        </div>
        <h1 className="text-5xl md:text-7xl font-black mb-4 tracking-tighter leading-none">
          SORA <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-500">STUDIO</span> PRO
        </h1>
        <p className="text-slate-500 text-sm md:text-base font-medium">
          @phuongnghi18091991 Voice Sync • TikTok Shop Safe
        </p>
      </header>

      <main className="max-w-7xl w-full grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left: Configuration */}
        <section className="lg:col-span-5 space-y-6">
          <div className="glass-panel p-8 rounded-[2rem] space-y-8">
            <div className="space-y-3">
              <label className="text-[10px] font-black uppercase tracking-widest text-slate-500 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
                Tên giày / Ý tưởng chủ đạo
              </label>
              <textarea 
                className="w-full h-32 bg-slate-900/50 border border-slate-800 rounded-2xl p-4 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all resize-none text-slate-200 placeholder:text-slate-700"
                placeholder="Ví dụ: Giày Sneaker trắng cổ cao, chất liệu da lộn..."
                value={idea}
                onChange={(e) => setIdea(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-3">
                <label className="text-[10px] font-black uppercase tracking-widest text-slate-500">Loại giày</label>
                <select 
                  className="w-full bg-slate-900/50 border border-slate-800 rounded-xl p-4 outline-none capitalize cursor-pointer hover:bg-slate-800 transition-colors"
                  value={shoeType}
                  onChange={(e) => setShoeType(e.target.value as ShoeType)}
                >
                  {SHOE_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
              <div className="space-y-3">
                <label className="text-[10px] font-black uppercase tracking-widest text-slate-500">Tone thoại</label>
                <select 
                  className="w-full bg-slate-900/50 border border-slate-800 rounded-xl p-4 outline-none cursor-pointer hover:bg-slate-800 transition-colors"
                  value={tone}
                  onChange={(e) => setTone(e.target.value)}
                >
                  {TONES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
            </div>

            <div className="space-y-3">
              <label className="text-[10px] font-black uppercase tracking-widest text-slate-500 flex justify-between">
                <span>Số lượng prompt</span>
                <span className="text-blue-400 font-mono">{count}</span>
              </label>
              <input 
                type="range" min="1" max="10" step="1"
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                value={count}
                onChange={(e) => setCount(parseInt(e.target.value))}
              />
            </div>

            <div className="space-y-3">
              <label className="text-[10px] font-black uppercase tracking-widest text-slate-500">Loại Prompt</label>
              <div className="grid grid-cols-2 gap-3">
                {Object.values(PromptMode).map(m => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`py-3 rounded-xl text-[11px] font-black transition-all border ${mode === m ? 'bg-blue-600 border-blue-400 text-white shadow-[0_0_20px_rgba(37,99,235,0.3)]' : 'bg-slate-900/50 border-slate-800 text-slate-500 hover:border-slate-600'}`}
                  >
                    {m.split(' (')[0]}
                  </button>
                ))}
              </div>
            </div>

            <div className="pt-4 space-y-4">
              <button 
                disabled={loading || !idea}
                onClick={handleGenerate}
                className="w-full py-5 pro-gradient rounded-2xl font-black text-white hover:shadow-[0_0_30px_rgba(59,130,246,0.4)] active:scale-95 transition-all disabled:opacity-30 flex items-center justify-center gap-3 uppercase tracking-widest"
              >
                {loading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Đang sinh prompt...
                  </>
                ) : `🎬 SINH ${count} PROMPT`}
              </button>
              
              <button 
                onClick={resetMemory}
                className="w-full py-3 bg-slate-900/30 border border-slate-800 rounded-xl text-[10px] font-bold text-slate-500 hover:text-slate-300 hover:border-slate-700 transition-all uppercase tracking-widest"
              >
                ♻️ Reset chống trùng
              </button>
            </div>
          </div>
        </section>

        {/* Right: Results Display */}
        <section className="lg:col-span-7 min-h-[600px] flex flex-col">
          {results.length > 0 ? (
            <div className="space-y-6 flex flex-col h-full">
              <div className="flex flex-wrap gap-2 p-1 bg-slate-900/50 border border-slate-800 rounded-2xl w-fit">
                {results.map((_, idx) => (
                  <button
                    key={idx}
                    onClick={() => setActiveTab(idx)}
                    className={`px-5 py-2.5 rounded-xl text-xs font-black transition-all ${activeTab === idx ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:bg-slate-800'}`}
                  >
                    #{idx + 1}
                  </button>
                ))}
              </div>
              
              <div className="flex-1">
                <PromptDisplay result={results[activeTab]} />
              </div>
            </div>
          ) : (
            <div className="glass-panel flex-1 rounded-[2rem] border-dashed border-2 border-slate-800/50 flex flex-col items-center justify-center text-slate-700 group">
              <div className="w-20 h-20 rounded-full bg-slate-900/50 flex items-center justify-center mb-6 border border-slate-800 group-hover:scale-110 transition-transform duration-500">
                <svg className="w-10 h-10 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <p className="font-bold text-lg tracking-tight">Studio Engine Idle</p>
              <p className="text-xs opacity-50 mt-1 uppercase tracking-widest">Tải lên ý tưởng để bắt đầu kịch bản</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default App;
