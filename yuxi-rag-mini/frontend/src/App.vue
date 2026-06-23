<template>
  <div id="app">
    <header class="header">
      <div class="header-left">
        <h1>RAG Console</h1>
      </div>
      <nav>
        <button @click="currentView = 'kb'" :class="{ active: currentView === 'kb' }">
          <span class="nav-icon">&#9638;</span> Knowledge Bases
        </button>
        <button @click="currentView = 'upload'" :class="{ active: currentView === 'upload' }">
          <span class="nav-icon">&#8686;</span> Files
        </button>
        <button @click="currentView = 'query'" :class="{ active: currentView === 'query' }">
          <span class="nav-icon">&#8981;</span> Query
        </button>
      </nav>
    </header>
    <main>
      <KnowledgeBaseView v-if="currentView === 'kb'" @select-kb="selectKb" />
      <UploadView v-if="currentView === 'upload'" :kb-id="selectedKbId" />
      <QueryView v-if="currentView === 'query'" :kb-id="selectedKbId" />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import KnowledgeBaseView from './views/KnowledgeBaseView.vue'
import UploadView from './views/UploadView.vue'
import QueryView from './views/QueryView.vue'

const currentView = ref('kb')
const selectedKbId = ref('')

function selectKb(kbId: string) {
  selectedKbId.value = kbId
  currentView.value = 'upload'
}
</script>

<style>
/* === Reset & Base === */
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --bg-primary: #0f1117;
  --bg-secondary: #1a1b26;
  --bg-card: #1e1f2e;
  --bg-hover: #252636;
  --border: #2d2d3f;
  --border-focus: #6366f1;
  --text-primary: #e5e7eb;
  --text-secondary: #9ca3af;
  --text-muted: #6b7280;
  --accent: #6366f1;
  --accent-hover: #818cf8;
  --danger: #ef4444;
  --danger-hover: #dc2626;
  --success: #22c55e;
  --warning: #f59e0b;
  --info: #3b82f6;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

#app {
  max-width: 1280px;
  margin: 0 auto;
  padding: 20px;
  min-height: 100vh;
}

/* === Header === */
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 20px;
}
.header h1 {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.02em;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

/* === Navigation === */
nav { display: flex; gap: 6px; }
nav button {
  padding: 7px 14px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 13px;
  transition: all 0.15s ease;
  display: flex;
  align-items: center;
  gap: 6px;
}
nav button:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}
nav button.active {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.nav-icon {
  font-size: 12px;
}

/* === Cards === */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 16px;
}

/* === Buttons === */
.btn {
  padding: 7px 14px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.15s ease;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover:not(:disabled) { background: var(--accent-hover); }
.btn-danger { background: var(--danger); color: #fff; }
.btn-danger:hover:not(:disabled) { background: var(--danger-hover); }
.btn-success { background: var(--success); color: #fff; }
.btn-success:hover:not(:disabled) { background: #16a34a; }
.btn-sm { padding: 4px 10px; font-size: 12px; }

/* === Form Elements === */
input, select, textarea {
  padding: 7px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 13px;
  width: 100%;
  background: var(--bg-primary);
  color: var(--text-primary);
  transition: border-color 0.15s ease;
}
input:focus, select:focus, textarea:focus {
  outline: none;
  border-color: var(--border-focus);
  box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15);
}
input::placeholder {
  color: var(--text-muted);
}
select {
  cursor: pointer;
}
label {
  display: block;
  margin-bottom: 4px;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.form-group { margin-bottom: 12px; }

/* === Tables === */
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
th {
  background: var(--bg-primary);
  font-weight: 600;
  font-size: 12px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
td {
  font-size: 13px;
  color: var(--text-primary);
}

/* === Badges === */
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.badge-success { background: rgba(34, 197, 94, 0.15); color: #4ade80; }
.badge-warning { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
.badge-danger { background: rgba(239, 68, 68, 0.15); color: #f87171; }
.badge-info { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }

/* === Empty State === */
.empty-state {
  text-align: center;
  padding: 40px;
  color: var(--text-muted);
  font-size: 14px;
}

/* === Scrollbar === */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
