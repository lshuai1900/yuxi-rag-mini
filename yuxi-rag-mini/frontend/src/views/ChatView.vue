<template>
  <div>
    <div class="card">
      <div class="card-header">
        <h2>Chat</h2>
        <span v-if="kbId" class="kb-badge">KB: {{ kbId }}</span>
      </div>
      <div v-if="!kbId" class="empty-state">Select a knowledge base to start chatting.</div>
      <div v-else>
        <!-- Search mode / top_k controls -->
        <div class="chat-controls">
          <div class="form-group" style="min-width: 160px;">
            <label>Search Mode</label>
            <select v-model="searchMode">
              <option value="hybrid">Hybrid</option>
              <option value="vector">Vector</option>
              <option value="keyword">Keyword (BM25)</option>
            </select>
          </div>
          <div class="form-group" style="min-width: 100px;">
            <label>Top K</label>
            <input type="number" v-model.number="topK" min="1" max="20" />
          </div>
          <div class="form-group" style="min-width: 140px;">
            <label>Temperature</label>
            <input type="number" v-model.number="temperature" min="0" max="1" step="0.05" />
          </div>
        </div>

        <!-- Question input -->
        <div class="query-row">
          <input
            v-model="question"
            placeholder="Ask a question about the knowledge base..."
            @keyup.enter="ask"
            class="query-input"
          />
          <button class="btn btn-primary" @click="ask" :disabled="!question.trim() || asking">
            {{ asking ? 'Thinking...' : 'Ask' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Answer -->
    <div v-if="answer || asking" class="card">
      <div class="card-header">
        <h2>Answer</h2>
        <span v-if="answerMeta" class="answer-meta">{{ answerMeta }}</span>
      </div>
      <div v-if="asking && !answer" class="empty-state">Generating answer...</div>
      <p v-else class="answer-text">{{ answer }}</p>
    </div>

    <!-- Sources -->
    <div v-if="sources.length > 0" class="card">
      <div class="card-header">
        <h2>Sources <span class="result-count">{{ sources.length }}</span></h2>
      </div>
      <div v-for="src in sources" :key="src.chunk_id || src.index" class="source-item">
        <div class="source-header">
          <div class="source-meta">
            <span class="badge badge-source">[{{ src.index }}]</span>
            <span class="source-filename">{{ src.filename || 'Unknown' }}</span>
          </div>
          <div class="source-scores">
            <span class="badge badge-score">Score: {{ (src.score ?? 0).toFixed(4) }}</span>
            <span class="badge badge-info">chunk #{{ src.chunk_index ?? 0 }}</span>
          </div>
        </div>
        <p class="source-content">{{ truncate(src.content, 400) }}</p>
        <div class="source-footer">
          chunk_id: {{ src.chunk_id }} <span v-if="src.file_id"> &middot; file_id: {{ src.file_id }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { chatWithKnowledgeBase, type Citation } from '../api/kb'

const props = defineProps<{ kbId: string }>()

const question = ref('')
const searchMode = ref<'vector' | 'keyword' | 'hybrid'>('hybrid')
const topK = ref(6)
const temperature = ref(0.2)
const asking = ref(false)

const answer = ref('')
const answerMeta = ref('')
const sources = ref<Citation[]>([])

function truncate(text: string | undefined, max: number): string {
  if (!text) return ''
  return text.length > max ? text.slice(0, max) + '...' : text
}

function getErrorMessage(e: any, fallback: string): string {
  const detail = e?.response?.data?.detail
  if (typeof detail === 'object' && detail?.message) return detail.message
  if (typeof detail === 'string') return detail
  return fallback
}

async function ask() {
  if (!question.value.trim() || !props.kbId) return
  asking.value = true
  answer.value = ''
  answerMeta.value = ''
  sources.value = []
  try {
    const res = await chatWithKnowledgeBase(props.kbId, {
      query: question.value,
      search_mode: searchMode.value,
      top_k: topK.value,
      temperature: temperature.value,
    })
    const data = res.data
    answer.value = data.answer || ''
    sources.value = data.sources || []
    answerMeta.value = `mode: ${data.search_mode} · ${sources.value.length} sources`
  } catch (e: any) {
    alert(getErrorMessage(e, 'Chat failed'))
  } finally {
    asking.value = false
  }
}
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.card-header h2 {
  margin: 0;
}
.kb-badge {
  font-size: 12px;
  padding: 2px 10px;
  border-radius: 4px;
  background: rgba(99, 102, 241, 0.15);
  color: #818cf8;
  font-family: monospace;
}
.chat-controls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
  align-items: flex-end;
}
.query-row {
  display: flex;
  gap: 12px;
}
.query-input {
  flex: 1;
}
.answer-meta {
  font-size: 12px;
  color: #9ca3af;
}
.answer-text {
  font-size: 14px;
  line-height: 1.7;
  white-space: pre-wrap;
  color: #e5e7eb;
}
.result-count {
  font-size: 13px;
  font-weight: 400;
  color: #9ca3af;
  margin-left: 8px;
}
.source-item {
  padding: 12px;
  border: 1px solid #2d2d3f;
  border-radius: 6px;
  margin-bottom: 8px;
  transition: border-color 0.2s;
}
.source-item:hover {
  border-color: #3d3d50;
}
.source-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.source-meta {
  display: flex;
  gap: 8px;
  align-items: center;
}
.source-filename {
  font-weight: 600;
  font-size: 14px;
}
.source-scores {
  display: flex;
  gap: 8px;
  align-items: center;
}
.badge-source {
  background: rgba(245, 158, 11, 0.15);
  color: #fbbf24;
}
.badge-score {
  background: rgba(99, 102, 241, 0.15);
  color: #818cf8;
}
.source-content {
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  color: #d1d5db;
}
.source-footer {
  margin-top: 8px;
  font-size: 11px;
  color: #6b7280;
  font-family: monospace;
}
</style>
