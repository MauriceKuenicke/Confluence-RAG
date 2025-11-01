import logging
import os
import sys

import alembic.config
import anyio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastembed.rerank.cross_encoder import TextCrossEncoder
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import models

from .qdrant import QdrantConfluencePages

encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
reranker = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')


logger = logging.getLogger('uvicorn.error')

app = FastAPI(title="RAG Backend API",
              swagger_ui_parameters={"defaultModelsExpandDepth": -1},
              docs_url="/api/docs",
              )

app.add_middleware(
    CORSMiddleware,  # NOQA
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthcheck")
def healthcheck():
    return {"Status": "Everything OK."}


html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Research Agent</title>
<style>
  /* CSS Reset & Base Styles */
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: #000000;
    color: #F5F5F7;
    font-size: 16px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  /* Support for reduced motion */
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }

  /* Main Container */
  .app-container {
    max-width: 960px;
    margin: 0 auto;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background-color: #1C1C1E;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.1);
  }

  /* Header */
  .header {
    padding: 24px 32px;
    border-bottom: 1px solid #2C2C2E;
    background-color: #1C1C1E;
  }

  .header h1 {
    font-size: 24px;
    font-weight: 600;
    color: #F5F5F7;
    letter-spacing: -0.02em;
  }

  /* Reasoning Panel - Sticky at Top */
  .reasoning-panel {
    position: sticky;
    top: 0;
    z-index: 100;
    background-color: #1C1C1E;
    border-bottom: 1px solid #2C2C2E;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  }

  .reasoning-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 32px;
    cursor: pointer;
    user-select: none;
  }

  .reasoning-header:focus {
    outline: 2px solid #0A84FF;
    outline-offset: -2px;
  }

  .reasoning-title-section {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .reasoning-title {
    font-size: 15px;
    font-weight: 600;
    color: #F5F5F7;
  }

  .chevron {
    display: inline-block;
    width: 20px;
    height: 20px;
    transition: transform 150ms cubic-bezier(0.4, 0, 0.2, 1);
    color: #98989D;
  }

  .chevron.expanded {
    transform: rotate(180deg);
  }

  .reasoning-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 500;
    color: #0A84FF;
    background: transparent;
    border: 1px solid #3A3A3C;
    border-radius: 8px;
    cursor: pointer;
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
  }

  .action-btn:hover {
    background-color: #2C2C2E;
    border-color: #0A84FF;
  }

  .action-btn:focus {
    outline: 2px solid #0A84FF;
    outline-offset: 2px;
  }

  .action-btn:active {
    transform: scale(0.98);
  }

  /* Reasoning Content Area */
  .reasoning-content {
    transition: max-height 200ms cubic-bezier(0.4, 0, 0.2, 1);
    background-color: #1C1C1E;
  }

  .reasoning-content.expanded {
    max-height: 400px;
    overflow-y: auto;
    border-top: 1px solid #2C2C2E;
  }

  /* Custom Scrollbar */
  .reasoning-content::-webkit-scrollbar {
    width: 8px;
  }

  .reasoning-content::-webkit-scrollbar-track {
    background: transparent;
  }

  .reasoning-content::-webkit-scrollbar-thumb {
    background: #3A3A3C;
    border-radius: 4px;
  }

  .reasoning-content::-webkit-scrollbar-thumb:hover {
    background: #48484A;
  }

  /* Latest Reasoning (Collapsed State) */
  .reasoning-latest {
    padding: 12px 32px;
    font-size: 14px;
    color: #98989D;
    line-height: 1.6;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }

  /* Reasoning History (Expanded State) */
  .reasoning-history {
    padding: 16px 32px 20px;
  }

  .reasoning-entry {
    padding: 12px 16px;
    margin-bottom: 8px;
    background-color: #2C2C2E;
    border-radius: 10px;
    border: 1px solid #3A3A3C;
    animation: fadeInPulse 200ms ease-out;
  }

  @keyframes fadeInPulse {
    0% {
      opacity: 0;
      transform: translateY(-4px);
    }
    50% {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .reasoning-entry:last-child {
    margin-bottom: 0;
  }

  .reasoning-entry-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }

  .reasoning-entry-badge {
    display: inline-block;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: #0A84FF;
    background-color: #1C3A52;
    border-radius: 6px;
    letter-spacing: 0.02em;
  }

  .reasoning-entry-time {
    font-size: 12px;
    color: #98989D;
  }

  .reasoning-entry-text {
    font-size: 14px;
    color: #F5F5F7;
    line-height: 1.5;
    word-wrap: break-word;
  }

  .connection-banner {
    padding: 8px 16px;
    font-size: 13px;
    text-align: center;
    border-radius: 8px;
    margin-bottom: 8px;
  }

  .connection-banner.connected {
    background-color: #1C3A2E;
    color: #4CD964;
  }

  .connection-banner.disconnected,
  .connection-banner.error {
    background-color: #3A1C1C;
    color: #FF453A;
  }

  /* Main Content Area */
  .main-content {
    flex: 1;
    padding: 32px;
    overflow-y: auto;
    background-color: #1C1C1E;
  }

  /* Result Box */
  .result-section {
    margin-top: 24px;
  }

  .result-section-title {
    font-size: 18px;
    font-weight: 600;
    color: #F5F5F7;
    margin-bottom: 16px;
    letter-spacing: -0.01em;
  }

  .result-box {
    position: relative;
    background-color: #2C2C2E;
    border: 1px solid #3A3A3C;
    border-radius: 12px;
    padding: 20px;
    min-height: 120px;
    font-size: 14px;
    line-height: 1.6;
    color: #F5F5F7;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  }

  .result-box.empty {
    color: #98989D;
    font-style: italic;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .result-box pre {
    margin: 0;
    font-family: "SF Mono", Menlo, Monaco, "Courier New", monospace;
    font-size: 13px;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  /* Markdown Content Styling */
  .result-content {
    line-height: 1.7;
  }

  .result-content h1,
  .result-content h2,
  .result-content h3,
  .result-content h4,
  .result-content h5,
  .result-content h6 {
    margin-top: 24px;
    margin-bottom: 12px;
    font-weight: 600;
    line-height: 1.3;
    color: #F5F5F7;
  }

  .result-content h1 {
    font-size: 28px;
    border-bottom: 1px solid #3A3A3C;
    padding-bottom: 8px;
  }

  .result-content h2 {
    font-size: 24px;
    border-bottom: 1px solid #3A3A3C;
    padding-bottom: 6px;
  }

  .result-content h3 {
    font-size: 20px;
  }

  .result-content h4 {
    font-size: 18px;
  }

  .result-content h5,
  .result-content h6 {
    font-size: 16px;
  }

  .result-content p {
    margin-bottom: 12px;
    color: #F5F5F7;
  }

  .result-content a {
    color: #0A84FF;
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 150ms ease;
  }

  .result-content a:hover {
    border-bottom-color: #0A84FF;
  }

  .result-content ul,
  .result-content ol {
    margin-bottom: 12px;
    padding-left: 24px;
  }

  .result-content li {
    margin-bottom: 6px;
  }

  .result-content code {
    font-family: "SF Mono", Menlo, Monaco, "Courier New", monospace;
    font-size: 13px;
    background-color: #2C2C2E;
    border: 1px solid #3A3A3C;
    border-radius: 4px;
    padding: 2px 6px;
    color: #FF9F0A;
  }

  .result-content pre {
    background-color: #2C2C2E;
    border: 1px solid #3A3A3C;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    overflow-x: auto;
  }

  .result-content pre code {
    background: transparent;
    border: none;
    padding: 0;
    color: #F5F5F7;
  }

  .result-content blockquote {
    margin: 12px 0;
    padding: 12px 16px;
    border-left: 4px solid #0A84FF;
    background-color: #2C2C2E;
    color: #98989D;
    font-style: italic;
  }

  .result-content table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 12px;
  }

  .result-content table th,
  .result-content table td {
    padding: 8px 12px;
    border: 1px solid #3A3A3C;
    text-align: left;
  }

  .result-content table th {
    background-color: #2C2C2E;
    font-weight: 600;
    color: #F5F5F7;
  }

  .result-content table td {
    background-color: #1C1C1E;
  }

  .result-content hr {
    border: none;
    border-top: 1px solid #3A3A3C;
    margin: 20px 0;
  }

  .result-content img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    margin: 12px 0;
  }

  .result-content strong {
    font-weight: 600;
    color: #F5F5F7;
  }

  .result-content em {
    font-style: italic;
  }

  .result-copy-btn {
    position: absolute;
    top: 12px;
    right: 12px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 500;
    color: #0A84FF;
    background-color: #1C1C1E;
    border: 1px solid #3A3A3C;
    border-radius: 8px;
    cursor: pointer;
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
  }

  .result-copy-btn:hover {
    background-color: #2C2C2E;
    border-color: #0A84FF;
  }

  .result-copy-btn:active {
    transform: scale(0.98);
  }

  /* Input Form */
  .input-form {
    padding: 20px 32px;
    border-top: 1px solid #2C2C2E;
    background-color: #1C1C1E;
  }

  .input-wrapper {
    display: flex;
    gap: 12px;
    max-width: 800px;
    margin: 0 auto;
  }

  .input-field {
    flex: 1;
    padding: 12px 16px;
    font-size: 15px;
    font-family: inherit;
    color: #F5F5F7;
    background-color: #2C2C2E;
    border: 1px solid #3A3A3C;
    border-radius: 10px;
    outline: none;
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
  }

  .input-field:focus {
    border-color: #0A84FF;
    box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.2);
  }

  .submit-btn {
    padding: 12px 28px;
    font-size: 15px;
    font-weight: 600;
    color: #FFFFFF;
    background-color: #0A84FF;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    white-space: nowrap;
  }

  .submit-btn:hover {
    background-color: #0070E0;
  }

  .submit-btn:active {
    transform: scale(0.98);
  }

  .submit-btn:focus {
    outline: 2px solid #0A84FF;
    outline-offset: 2px;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .app-container {
      box-shadow: none;
    }

    .header {
      padding: 20px 20px;
    }

    .header h1 {
      font-size: 20px;
    }

    .reasoning-header {
      padding: 14px 20px;
    }

    .reasoning-latest,
    .reasoning-history {
      padding-left: 20px;
      padding-right: 20px;
    }

    .main-content {
      padding: 20px;
    }

    .input-form {
      padding: 16px 20px;
    }

    .input-wrapper {
      flex-direction: column;
      gap: 10px;
    }

    .submit-btn {
      width: 100%;
    }

    .action-btn {
      font-size: 12px;
      padding: 5px 10px;
    }
  }
</style>
</head>
<body>
  <div class="app-container">
    <!-- Header -->
    <header class="header">
      <h1>Confluence Search</h1>
    </header>

    <!-- Sticky Reasoning Panel -->
    <div class="reasoning-panel">
      <div 
        class="reasoning-header" 
        role="button" 
        tabindex="0" 
        aria-expanded="false" 
        aria-controls="reasoningContent"
        id="reasoningHeader"
      >
        <div class="reasoning-title-section">
          <span class="reasoning-title" id="reasoningTitle">Latest reasoning</span>
          <svg class="chevron" id="chevron" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
          </svg>
        </div>
        <div class="reasoning-actions" id="reasoningActions">
          <button class="action-btn" id="clearBtn" aria-label="Clear session">Clear</button>
        </div>
      </div>
      
      <div class="reasoning-content" id="reasoningContent" aria-live="polite" aria-atomic="false">
        <!-- Collapsed: shows latest -->
        <div class="reasoning-latest" id="reasoningLatest"></div>
        <!-- Expanded: shows full history -->
        <div class="reasoning-history" id="reasoningHistory" style="display: none;"></div>
      </div>
    </div>

    <!-- Main Content -->
    <main class="main-content">
      <!-- Final Result Section -->
      <section class="result-section">
        <h2 class="result-section-title">Results</h2>
        <div class="result-box empty" id="resultBox">
          <span>No results yet. Send a query to get started.</span>
          <button class="result-copy-btn" id="resultCopyBtn" style="display: none;">Copy</button>
        </div>
      </section>
    </main>

    <!-- Input Form -->
    <form class="input-form" id="inputForm">
      <div class="input-wrapper">
        <input 
          type="text" 
          class="input-field" 
          id="queryInput" 
          placeholder="Ask a question..." 
          required 
          autocomplete="off"
        />
        <button type="submit" class="submit-btn">Send</button>
      </div>
    </form>
  </div>

<script>
  // State Management
  let ws = null;
  let connectionState = 'connecting';
  let reasoningHistory = [];
  let isReasoningExpanded = false;
  let finalResult = null;
  let reasoningIdCounter = 0;

  // DOM Elements
  const reasoningHeader = document.getElementById('reasoningHeader');
  const reasoningTitle = document.getElementById('reasoningTitle');
  const chevron = document.getElementById('chevron');
  const reasoningContent = document.getElementById('reasoningContent');
  const reasoningLatest = document.getElementById('reasoningLatest');
  const reasoningHistory_el = document.getElementById('reasoningHistory');
  const clearBtn = document.getElementById('clearBtn');
  const resultBox = document.getElementById('resultBox');
  const resultCopyBtn = document.getElementById('resultCopyBtn');
  const inputForm = document.getElementById('inputForm');
  const queryInput = document.getElementById('queryInput');

  // Utility: Format timestamp
  function formatTime(date) {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit',
      hour12: false 
    });
  }

  // Utility: Copy to clipboard
  async function copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (err) {
      console.error('Failed to copy:', err);
      return false;
    }
  }

  // Initialize WebSocket
  function initWebSocket() {
    const wsUrl = `ws://localhost:8005/ws`;
    
    ws = new WebSocket(wsUrl);
    console.log('Connecting to:', wsUrl);

    ws.onopen = () => {
      connectionState = 'open';
      addConnectionBanner('Connected', 'connected');
    };

    ws.onclose = () => {
      connectionState = 'closed';
      addConnectionBanner('Disconnected', 'disconnected');
    };

    ws.onerror = () => {
      connectionState = 'error';
      addConnectionBanner('Connection Error', 'error');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'info') {
          addReasoningEntry(data.message);
        } else if (data.type === 'results') {
          displayResult(data.data);
          addReasoningEntry('✓ Result generated', 'result');
        }
      } catch (e) {
        console.error('Error parsing message:', e);
      }
    };
  }

  // Add connection banner to reasoning history
  function addConnectionBanner(message, status) {
    const banner = document.createElement('div');
    banner.className = `connection-banner ${status}`;
    banner.textContent = message;
    reasoningHistory_el.insertBefore(banner, reasoningHistory_el.firstChild);
  }

  // Add reasoning entry
  function addReasoningEntry(text, type = 'info') {
    const entry = {
      id: ++reasoningIdCounter,
      text: text,
      timestamp: new Date(),
      type: type
    };
    
    reasoningHistory.push(entry);
    
    // Update latest (collapsed view)
    reasoningLatest.textContent = text;
    
    // Add subtle pulse animation
    reasoningLatest.style.animation = 'none';
    setTimeout(() => {
      reasoningLatest.style.animation = 'fadeInPulse 200ms ease-out';
    }, 10);
    
    // Update full history (expanded view)
    const entryEl = createReasoningEntryElement(entry);
    reasoningHistory_el.appendChild(entryEl);
    
    // Auto-scroll to bottom when expanded
    if (isReasoningExpanded) {
      reasoningContent.scrollTop = reasoningContent.scrollHeight;
    }
  }

  // Create reasoning entry DOM element
  function createReasoningEntryElement(entry) {
    const entryDiv = document.createElement('div');
    entryDiv.className = 'reasoning-entry';
    
    const header = document.createElement('div');
    header.className = 'reasoning-entry-header';
    
    const badge = document.createElement('span');
    badge.className = 'reasoning-entry-badge';
    badge.textContent = entry.type === 'result' ? 'Result' : 'Info';
    
    const time = document.createElement('span');
    time.className = 'reasoning-entry-time';
    time.textContent = formatTime(entry.timestamp);
    
    header.appendChild(badge);
    header.appendChild(time);
    
    const textDiv = document.createElement('div');
    textDiv.className = 'reasoning-entry-text';
    textDiv.textContent = entry.text;
    
    entryDiv.appendChild(header);
    entryDiv.appendChild(textDiv);
    
    return entryDiv;
  }

  // Toggle reasoning panel
  function toggleReasoning() {
    isReasoningExpanded = !isReasoningExpanded;
    
    if (isReasoningExpanded) {
      reasoningTitle.textContent = 'Full reasoning history';
      chevron.classList.add('expanded');
      reasoningContent.classList.add('expanded');
      reasoningLatest.style.display = 'none';
      reasoningHistory_el.style.display = 'block';
      reasoningHeader.setAttribute('aria-expanded', 'true');
      
      // Scroll to bottom
      setTimeout(() => {
        reasoningContent.scrollTop = reasoningContent.scrollHeight;
      }, 210);
    } else {
      reasoningTitle.textContent = 'Latest reasoning';
      chevron.classList.remove('expanded');
      reasoningContent.classList.remove('expanded');
      reasoningLatest.style.display = 'block';
      reasoningHistory_el.style.display = 'none';
      reasoningHeader.setAttribute('aria-expanded', 'false');
    }
  }

  // Display final result
  function displayResult(data) {
    finalResult = data;
    resultBox.classList.remove('empty');
    resultBox.innerHTML = '';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'result-content';
    
    if (typeof data === 'string') {
      // Parse and render markdown
      contentDiv.innerHTML = marked.parse(data);
    } else {
      // For non-string data, display as formatted JSON
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(data, null, 2);
      contentDiv.appendChild(pre);
    }
    
    resultBox.appendChild(contentDiv);
    
    resultCopyBtn.style.display = 'block';
    resultBox.appendChild(resultCopyBtn);
    
    // Scroll result into view
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Clear session
  clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (confirm('Clear all reasoning history and results?')) {
      reasoningHistory = [];
      reasoningIdCounter = 0;
      finalResult = null;
      reasoningLatest.textContent = '';
      reasoningHistory_el.innerHTML = '';
      resultBox.classList.add('empty');
      resultBox.innerHTML = '<span>No results yet. Send a query to get started.</span>';
      resultCopyBtn.style.display = 'none';
    }
  });

  // Copy result
  resultCopyBtn.addEventListener('click', async () => {
    const text = typeof finalResult === 'string' 
      ? finalResult 
      : JSON.stringify(finalResult, null, 2);
    
    const success = await copyToClipboard(text);
    if (success) {
      const originalText = resultCopyBtn.textContent;
      resultCopyBtn.textContent = 'Copied!';
      setTimeout(() => {
        resultCopyBtn.textContent = originalText;
      }, 1500);
    }
  });

  // Toggle reasoning panel on click
  reasoningHeader.addEventListener('click', toggleReasoning);

  // Keyboard navigation for reasoning header
  reasoningHeader.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggleReasoning();
    }
  });

  // Handle form submission
  inputForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const query = queryInput.value.trim();
    if (!query || connectionState !== 'open') return;
    
    ws.send(query);
    queryInput.value = '';
    
    // Clear previous results
    resultBox.classList.add('empty');
    resultBox.innerHTML = '<span>Processing your query...</span>';
    resultCopyBtn.style.display = 'none';
  });

  // Initialize on page load
  initWebSocket();
</script>
<script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
</body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            client = OpenAI(api_key=os.environ.get("APP__LLM_API_KEY"))
            query = await websocket.receive_text()

            # Send immediate progress messages
            await websocket.send_json({"type": "info", "message": f"The user is asking: {query}"})
            await websocket.send_json({"type": "info", "message": "I'll search through our Confluence system..."})

            vector_store = await anyio.to_thread.run_sync(QdrantConfluencePages)

            def _do_query():
                return vector_store.client.query_points(
                    collection_name=vector_store.collection,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    prefetch=[
                        models.Prefetch(
                            query=models.Document(text=query, model="sentence-transformers/all-MiniLM-L6-v2"),
                            using="dense_content",
                            limit=20,
                        ),
                        models.Prefetch(
                            query=models.Document(text=query, model="Qdrant/bm25"),
                            using="sparse_content",
                            limit=20,
                        ),
                        models.Prefetch(
                            query=models.Document(text=query, model="Qdrant/bm25"),
                            using="sparse_title",
                            limit=20,
                        ),
                    ],
                    query_filter=None,
                    limit=7,
                ).points

            search_result = await anyio.to_thread.run_sync(_do_query)

            metadata = [point.payload for point in search_result]
            pages = sorted({x["page_title"] for x in metadata})
            await websocket.send_json({"type": "info", "message": f"Looking into: {pages}"})
            description_hits = [point.payload["chunk_content"] for point in search_result]

            def _rerank():
                new_scores = list(reranker.rerank(query, description_hits))
                ranking = [(i, score) for i, score in enumerate(new_scores)]
                ranking.sort(key=lambda x: x[1], reverse=True)
                return [metadata[i] for i, _ in ranking]

            found_docs = await anyio.to_thread.run_sync(_rerank)

            class ExtendedQueries(BaseModel):
                queries: list[str]

            response = client.responses.parse(
                model="gpt-5-nano-2025-08-07",
                input=[
                    {"role": "system", "content": """You are a helpful assistant that improves search queries by generating related
                     terms and phrases. Given a user query, generate 4 to 6 concise keyword phrases or 
                     synonyms that capture the key concepts and related ideas relevant for improving search retrieval."""},
                    {
                        "role": "user",
                        "content": f"The user query: {query}",
                    }
                ],
                text_format=ExtendedQueries
            )
            subqueries = response.output_parsed
            print(subqueries)

            await websocket.send_json({"type": "info", "message": "Results found. Generating response..."})
            response = client.responses.create(
                model="gpt-5",
                instructions="""You are ACMA Assistant, a helpful, accurate internal chatbot for ACMA employees. You answer questions using ACMA’s internal Confluence documentation. Your top priorities are accuracy, clarity, and respecting confidentiality.

Core Rules
- Use only the provided context to answer. Do not invent facts or rely on outside knowledge unless explicitly told to.
- If the context does not contain an answer, say so clearly and propose next steps (e.g., where to look or what clarifying info is needed).
- Always cite your sources from the provided context. Prefer human-friendly titles and URLs.
- Be concise but complete. Use bullet points and headings where appropriate.
- Never reveal hidden chain-of-thought or internal reasoning. Summaries are fine, but do not show step-by-step internal deliberations.
- Treat all content as confidential. Do not include sensitive data unless shown in the provided context and relevant to the user’s question.

Answer Style
- Tone: professional, friendly, and precise.
- Format answers in Markdown.
- Start with a brief, direct answer. Follow with details and a Sources section.
- For how-to/procedures, return clear, numbered steps.
- For code/config snippets, use fenced code blocks with language hints when possible.
- If the question is ambiguous, ask one short clarifying question before proceeding, unless a useful partial answer is possible.

Citations
- Inline: When referencing specific details, include a short citation number like [1].
- End section: Include a “Sources” section listing the sources as well as the citation number you used. Deduplicate near-duplicates; prefer the latest versions.
- If multiple chunks are from the same page, merge them into one citation.

Conflict / Quality Handling
- If sources conflict, call it out and explain which is likely more reliable (e.g., newer version, official guideline).
- Prefer the most recent or clearly authoritative documents (e.g., “Standards”, “Runbooks”, “Architecture docs”).
- If the context looks outdated or versioned, indicate the version you used if visible.

When Context Is Insufficient
- State: “I don’t have enough information in the provided context to answer confidently.”
- Suggest: where to look (space/page names, teams, or keywords) or ask a clarifying question.

Security & Safety
- Do not share secrets, credentials, or PII beyond what is shown in the provided context and required to answer.
- If asked to perform actions outside reading/summarizing context (e.g., execute code), explain limitations.

Expected Output Structure (Markdown)
- Title (optional if short)
- Short Answer
- Details (bullets or sections)
- If applicable: Steps, Examples, or Tables
- Sources (with links)
- Optional: Next steps or clarifications
""",
                input=f"""
                    User Query:
                    {{{query}}}
                    
                    Context (use only this to answer):
                    {{{[{"content": x["chunk_content"], "source": x["page_url"], "page_title": x["page_title"]} for x in found_docs]}}}
                    - The context may include multiple entries like:
                      - content: "…chunk text…"
                      - source: "https://confluence.acma.local/space/page#anchor"
                      - page_title: "Page Title" """
            )
            await websocket.send_json({"type": "results", "data": response.output_text})

    except WebSocketDisconnect:
        # Client closed the connection; nothing to do.
        print('Client disconnected')
        return

if "pytest" not in sys.modules:
    alembic.config.main(argv=["--raiseerr", "upgrade", "head"])
    qdrant_wrapper = QdrantConfluencePages()
    logger.info("Creating Qdrant collection if not exist")
    qdrant_wrapper.create_collection()
