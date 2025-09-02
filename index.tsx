import React, { useState, useEffect, useRef, FormEvent, FC, ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Content } from '@google/genai';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const MODEL_NAME = 'gemini-2.5-pro';

// --- 第一段階：初期応答生成用 ---
const INITIAL_SYSTEM_INSTRUCTION =  `あなたは、高度な専門知識を持つAIアシスタントです。ユーザーの質問に対し、初回応答として、網羅的で精度の高い、論理的な回答を生成することがあなたのタスクです。
以下の指示に厳密に従ってください:
1.  **正確性と厳密性を最優先**してください。情報が不確かな場合や推測に基づく場合は、その旨を明確に記述してください。
2.  回答は**多角的な視点**から構成し、単純な二項対立に陥ることを避けてください。
3.  提案する視点のうち、**少なくとも1つは、主流の意見に対する対抗的・批判的な視点**を必ず含めてください。
4.  定量的な分析が可能な場合は、**具体的な数値や数式を用いて**説明してください。

注意: この応答は、他のAIエージェントが利用するための中間生成物であり、ユーザーには直接表示されません。冗長な表現を避け、核心的な情報に焦点を当てて簡潔に記述してください。`;

// --- 第二段階：深掘り分析用（5つのモード）---
const DEEP_DIVE_INSTRUCTIONS = {
  strategy: `あなたは「戦略比較」を専門とするAIアナリストです。与えられた情報に基づき、以下のタスクを厳密に実行してください。
1. 目的達成のための、互いに異なる具体的な戦略を3つ提案する。
2. 各戦略について、「メリット」「デメリット」「実行の前提条件」を明確に整理してリスト化する。
3. 3つの戦略を最も効果が高いと思われる順にランク付けし、その順位付けの根拠と、順位によるトレードオフ（例：1位は効果が高いがリスクも大きい、など）を明示する。`,

  challenge: `あなたは「前提・思考挑戦」を専門とするAIクリティークです。与えられた情報や提案に対し、以下のタスクを厳密に実行してください。
1. 議論の「盲点」や見落とされている論点を、鋭く複数指摘する。
2. 暗黙的に設定されている、あるいは疑われていない重要な「前提」を特定し、それに挑戦する問いを立てる（例：「そもそも、この目標設定は正しいのか？」）。
3. 主流の意見からは見落とされがちな、全く異なる視点やオルタナティブな解釈を提示する。`,

  expert: `あなたは「専門家向け圧縮」を専門とするAIコンサルタントです。与えられた情報を、高度な知的水準を持つ専門家（経営者、研究者など）が、時間的制約の厳しい意思決定の場で利用するために再構成してください。
1. 冗長な説明や表面的な分析をすべて排除し、議論の核心となる深い論点のみに絞り込む。
2. 情報を箇条書きや短いパラグラフで再構成し、30秒で全体像が把握できるようにする。
3. 専門用語の使用を厭わず、知的誠実さを保ち、簡潔かつ鋭いアウトプットを生成する。`,

  insights: `あなたは「ハイレベル洞察」を専門とするAIストラテジストです。与えられた情報全体を俯瞰し、表層的な事象の背後にある本質的な示唆を抽出してください。
1. これまでのやり取りの中から、最も重要だと思われる「パターン」「インプリケーション」「原理原則」を3つ抽出する。
2. それぞれの洞察が、「短期的な成果」と「長期的な戦略的ポジショニング」の両方に対してどのような意味を持つかを解説する。
3. 抽象的なレベルにとどまらず、具体的なアクションに繋がりうるような示唆を提供する。`,

  checklist: `あなたは「成功チェックリスト」の作成を専門とするAIプランナーです。与えられた計画や目標に対し、その成否を分けるであろう要素を網羅した、専門家レベルのチェックリストを作成してください。
1. 成功のために絶対不可欠だと思われる条件を、具体的なチェック項目としてリストアップする。
2. 各チェック項目がなぜ重要なのか、その理由を簡潔に説明する。
3. 計画が各条件をどの程度満たしているかを評価し、不足している点については具体的な改善アクションを提示する。特に、成否に直結するクリティカルな要素を重点的に扱う。`
};

// --- 第三段階：最終統合用 ---
const SYNTHESIZER_SYSTEM_INSTRUCTION = `あなたは、最高の統合能力を持つマスターAIです。あなたの最重要目標は、ユーザーの質問に対する最終的で完璧な回答を記述することです。
あなたには、ユーザーの質問と、5つの異なる専門的観点（戦略比較、前提挑戦、専門家向け圧縮、ハイレベル洞察、成功チェックリスト）から深掘りされた分析結果が与えられます。
これらの多様な分析結果を批判的に吟味し、それぞれの長所を組み合わせ、矛盾点を解消し、一貫した論理構造を持つ、単一の最終回答を構築してください。
あなたの出力が、ユーザーが目にする唯一の完成品です。分析プロセスを説明するのではなく、完成された回答そのものを生成してください。`;

interface Message {
  role: 'user' | 'model';
  parts: { text: string }[];
}

const CodeBlock: FC<{ children?: ReactNode }> = ({ children }) => {
  const [copied, setCopied] = useState(false);
  const textToCopy = String(children).replace(/\n$/, '');

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  return (
    <div className="code-block-wrapper">
      <pre><code>{children}</code></pre>
      <button onClick={handleCopy} className="copy-button" aria-label="Copy code">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
          {copied ? (
            <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
          ) : (
            <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-5zm0 16H8V7h11v14z"/>
          )}
        </svg>
        {copied ? 'Copied!' : 'Copy'}
      </button>
    </div>
  );
};

const LoadingIndicator: FC<{ status: string; time: number }> = ({ status, time }) => (
  <div className="loading-animation">
    <div className="loading-header">
      <span className="loading-status">{status}</span>
      <span className="timer-display">{(time / 1000).toFixed(1)}s</span>
    </div>
    {/* statusに応じてプログレスバーの数を変更 */}
    <div className={`progress-bars-container ${status.startsWith('Deepening') ? 'deep-dive' : 'initial'}`}>
      <div className="progress-bar"></div>
      <div className="progress-bar"></div>
      <div className="progress-bar"></div>
      <div className="progress-bar"></div>
      {/* 深掘り分析の時だけ5本目のバーを表示 */}
      {status.startsWith('Deepening') && <div className="progress-bar"></div>}
    </div>
  </div>
);

const App: FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingStatus, setLoadingStatus] = useState<string>('');
  const [timer, setTimer] = useState<number>(0);
  const messageListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [messages, isLoading]);
  
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isLoading) {
      interval = setInterval(() => {
        setTimer(prevTime => prevTime + 100);
      }, 100);
    } else {
      setTimer(0);
    }
    return () => clearInterval(interval);
  }, [isLoading]);


  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const userInput = formData.get('userInput') as string;
    event.currentTarget.reset();
    if (!userInput.trim()) return;

    const userMessage: Message = { role: 'user', parts: [{ text: userInput }] };
    const currentMessages = [...messages, userMessage];
    setMessages(currentMessages);
    setIsLoading(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
      
      const mainChatHistory: Content[] = currentMessages.slice(0, -1).map(msg => ({
        role: msg.role,
        parts: msg.parts,
      }));
      const currentUserTurn: Content = { role: 'user', parts: [{ text: userInput }] };

      // === STEP 1: Initial Responses (Concurrent) ===
      setLoadingStatus('Initializing agents...');
      const initialAgentPromises = Array(4).fill(0).map(() => 
        ai.models.generateContent({
          model: MODEL_NAME,
          contents: [...mainChatHistory, currentUserTurn],
          config: { systemInstruction: INITIAL_SYSTEM_INSTRUCTION },
        })
      );
      const initialResponses = await Promise.all(initialAgentPromises);
      const initialAnswers = initialResponses.map(res => res.text);
      const combinedInitialContext = `以下は、4つのエージェントによって生成された初期応答です。これらの多様な視点を、後続の分析の基礎情報としてください。\n\n--- 初期応答 ---\n1. ${initialAnswers[0]}\n\n2. ${initialAnswers[1]}\n\n3. ${initialAnswers[2]}\n\n4. ${initialAnswers[3]}\n---`;


      // === STEP 2: Deep Dive Analysis (Concurrent) ===
      setLoadingStatus('Deepening analysis...');
      const deepDiveTurn: Content = { role: 'user', parts: [{ text: `${userInput}\n\n---INTERNAL CONTEXT---\n${combinedInitialContext}` }] };
      
      const deepDivePromises = Object.values(DEEP_DIVE_INSTRUCTIONS).map(instruction => 
        ai.models.generateContent({
          model: MODEL_NAME,
          contents: [...mainChatHistory, deepDiveTurn],
          config: { systemInstruction: instruction },
        })
      );
      const deepDiveResponses = await Promise.all(deepDivePromises);
      const deepDiveAnswers = deepDiveResponses.map(res => res.text);

      // === STEP 3: Final Synthesis ===
      setLoadingStatus('Synthesizing final response...');
      const synthesizerContext = `ユーザーの質問に対し、5つの専門エージェントが以下の通り多角的な分析を行いました。これらの分析を統合し、最高の最終回答を作成してください。

--- 分析結果 ---
1.  **戦略比較**:
    "${deepDiveAnswers[0]}"

2.  **前提・思考挑戦**:
    "${deepDiveAnswers[1]}"

3.  **専門家向け圧縮**:
    "${deepDiveAnswers[2]}"

4.  **ハイレベル洞察**:
    "${deepDiveAnswers[3]}"

5.  **成功チェックリスト**:
    "${deepDiveAnswers[4]}"
---`;
      const synthesizerTurn: Content = { role: 'user', parts: [{ text: `${userInput}\n\n---INTERNAL CONTEXT---\n${synthesizerContext}` }] };

      const finalResult = await ai.models.generateContent({
        model: MODEL_NAME,
        contents: [...mainChatHistory, synthesizerTurn],
        config: { systemInstruction: SYNTHESIZER_SYSTEM_INSTRUCTION },
      });
      
      setIsLoading(false);

      const finalResponseText = finalResult.text;
      const finalMessage: Message = { role: 'model', parts: [{ text: finalResponseText }] };
      setMessages(prev => [...prev, finalMessage]);

    } catch (error) {
      console.error('Error sending message to agents:', error);
      setIsLoading(false);
      setMessages(prev => [...prev, { role: 'model', parts: [{ text: 'Sorry, I encountered an error. Please try again.' }] }]);
    }
  };

  return (
    <div className="chat-container">
      <header>
        <h1>Multi-Agent Chat</h1>
      </header>
      <div className="message-list" ref={messageListRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
             {msg.role === 'model' && <span className="agent-label">Synthesizer Agent</span>}
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code(props) {
                  const {children, className, ...rest} = props
                  return <CodeBlock>{String(children)}</CodeBlock>
                }
              }}
            >
              {msg.parts[0].text}
            </ReactMarkdown>
          </div>
        ))}
        {isLoading && <LoadingIndicator status={loadingStatus} time={timer} />}
      </div>
      <form className="input-area" onSubmit={handleSubmit}>
        <input
          type="text"
          name="userInput"
          placeholder="Ask the agents..."
          aria-label="User input"
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading} aria-label="Send message">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
        </button>
      </form>
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
