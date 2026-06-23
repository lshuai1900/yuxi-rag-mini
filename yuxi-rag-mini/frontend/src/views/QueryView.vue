<template>
  <div>
    <div class="card">
      <div class="card-header">
        <h2>Query</h2>
        <span v-if="kbId" class="kb-badge">KB: {{ kbId }}</span>
      </div>
      <div v-if="!kbId" class="empty-state">Select a knowledge base to begin querying.</div>
      <div v-else>
        <!-- Search Mode (always visible) -->
        <div class="search-mode-row">
          <div class="form-group" style="min-width: 160px;">
            <label>Search Mode</label>
            <select v-model="retrievalConfig.search_mode">
              <option value="hybrid">Hybrid</option>
              <option value="vector">Vector</option>
              <option value="keyword">Keyword (BM25)</option>
            </select>
          </div>
          <div class="form-group" style="min-width: 100px;">
            <label>Top K</label>
            <input type="number" v-model.number="retrievalConfig.final_top_k" min="1" max="100" />
          </div>
          <div class="form-group" style="min-width: 140px;">
            <label>Similarity Threshold</label>
            <input type="number" v-model.number="retrievalConfig.similarity_threshold" min="0" max="1" step="0.05" />
          </div>
          <div class="form-group toggle-group">
            <label class="toggle-label">
              <input type="checkbox" v-model="retrievalConfig.use_reranker" />
              <span>Reranker</span>
            </label>
          </div>
        </div>

        <!-- Advanced Settings Toggle -->
        <button class="advanced-toggle" @click="advancedOpen = !advancedOpen">
          <span class="toggle-icon">{{ advancedOpen ? '▾' : '▸' }}</span>
          Advanced Settings
        </button>

        <!-- Advanced Settings Panel -->
        <div v-if="advancedOpen" class="advanced-panel">
          <div class="advanced-grid">
            <div class="form-group">
              <label>BM25 Top K</label>
              <input type="number" v-model.number="retrievalConfig.bm25_top_k" min="1" max="200" />
            </div>
            <div class="form-group">
              <label>Vector Weight</label>
              <input type="number" v-model.number="retrievalConfig.vector_weight" min="0" max="1" step="0.05" />
            </div>
            <div class="form-group">
              <label>BM25 Weight</label>
              <input type="number" v-model.number="retrievalConfig.bm25_weight" min="0" max="1" step="0.05" />
            </div>
            <div class="form-group">
              <label>BM25 Drop Ratio</label>
              <input type="number" v-model.number="retrievalConfig.bm25_drop_ratio_search" min="0" max="1" step="0.05" />
            </div>
            <div class="form-group">
              <label>Recall Top K</label>
              <input type="number" v-model.number="retrievalConfig.recall_top_k" min="1" max="200" />
            </div>
          </div>
          <button class="btn btn-sm btn-ghost" @click="resetDefaults">Reset to Defaults</button>
        </div>

        <!-- Query Input -->
        <div class="query-row">
          <input v-model="queryText" placeholder="Enter your query..." @keyup.enter="search" class="query-input" />
          <button class="btn btn-primary" @click="search" :disabled="!queryText.trim() || searching">
            {{ searching ? 'Searching...' : 'Search' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Results -->
    <div v-if="results.length > 0" class="card">
      <div class="card-header">
        <h2>Results <span class="result-count">{{ results.length }}</span></h2>
        <span v-if="rerankInfo" class="rerank-info">
          Reranked: {{ rerankInfo.reranked ? 'Yes' : 'No' }}
          <span v-if="rerankInfo.reranker">({{ rerankInfo.reranker }})</span>
        </span>
      </div>
      <div v-for="(result, idx) in results" :key="idx" class="result-item">
        <div class="result-header">
          <div class="result-meta">
            <span class="result-filename">{{ result.filename || 'Unknown' }}</span>
            <span v-if="result.metadata?.page_number" class="badge badge-info">Page {{ result.metadata.page_number }}</span>
          </div>
          <div class="result-scores">
            <span class="badge badge-score">Score: {{ result.score?.toFixed(4) }}</span>
          </div>
        </div>
        <div v-if="result.score_detail" class="score-detail">
          <span>Vector: {{ result.score_detail.vector_score?.toFixed(4) }}</span>
          <span>Keyword: {{ result.score_detail.keyword_score?.toFixed(4) }}</span>
          <span>Final: {{ result.score_detail.final_score?.toFixed(4) }}</span>
          <span class="badge" :class="result.score_detail.source === 'hybrid' ? 'badge-warning' : 'badge-info'">{{ result.score_detail.source }}</span>
        </div>
        <p class="result-content">{{ result.content }}</p>
        <div class="result-footer">
          Chunk: {{ result.chunk_id }} &middot; File: {{ result.file_id }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { queryKnowledgeBase, type RetrievalConfig } from '../api/kb'

const props = defineProps<{ kbId: string }>()

const defaultConfig = (): RetrievalConfig => ({
  search_mode: 'hybrid',
  final_top_k: 10,
  similarity_threshold: 0.0,
  bm25_top_k: 20,
  vector_weight: 0.7,
  bm25_weight: 0.3,
  bm25_drop_ratio_search: 0.2,
  recall_top_k: 20,
  use_reranker: false,
})

const retrievalConfig = reactive<RetrievalConfig>(defaultConfig())
const advancedOpen = ref(false)
const queryText = ref('')
const results = ref<any[]>([])
const searching = ref(false)
const rerankInfo = ref<{ reranked: boolean; reranker: string } | null>(null)

function resetDefaults() {
  const defaults = defaultConfig()
  Object.assign(retrievalConfig, defaults)
}

async function search() {
  if (!queryText.value.trim() || !props.kbId) return
  searching.value = true
  try {
    const res = await queryKnowledgeBase(props.kbId, {
      query: queryText.value,
      retrieval_config: { ...retrievalConfig },
    })
    results.value = res.data.results || []
    rerankInfo.value = res.data.rerank || null
  } catch (e: any) {
    const detail = e?.response?.data?.detail
    let msg = 'Query failed'
    if (typeof detail === 'object' && detail?.message) msg = detail.message
    else if (typeof detail === 'string') msg = detail
    alert(msg)
  } finally {
    searching.value = false
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
.search-mode-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
  align-items: flex-end;
}
.toggle-group {
  display: flex;
  align-items: flex-end;
  padding-bottom: 4px;
}
.toggle-label {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font-size: 14px;
}
.toggle-label input[type="checkbox"] {
  width: auto;
  cursor: pointer;
}
.advanced-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  background: none;
  border: none;
  color: #818cf8;
  cursor: pointer;
  font-size: 13px;
  padding: 6px 0;
  margin-bottom: 8px;
  transition: color 0.2s;
}
.advanced-toggle:hover {
  color: #a5b4fc;
}
.toggle-icon {
  font-size: 11px;
}
.advanced-panel {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid #2d2d3f;
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 12px;
}
.advanced-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}
.btn-ghost {
  background: transparent;
  border: 1px solid #3d3d50;
  color: #9ca3af;
}
.btn-ghost:hover {
  border-color: #818cf8;
  color: #818cf8;
}
.query-row {
  display: flex;
  gap: 12px;
}
.query-input {
  flex: 1;
}
.result-count {
  font-size: 13px;
  font-weight: 400;
  color: #9ca3af;
  margin-left: 8px;
}
.rerank-info {
  font-size: 12px;
  color: #9ca3af;
}
.result-item {
  padding: 12px;
  border: 1px solid #2d2d3f;
  border-radius: 6px;
  margin-bottom: 8px;
  transition: border-color 0.2s;
}
.result-item:hover {
  border-color: #3d3d50;
}
.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.result-meta {
  display: flex;
  gap: 8px;
  align-items: center;
}
.result-filename {
  font-weight: 600;
  font-size: 14px;
}
.result-scores {
  display: flex;
  gap: 8px;
  align-items: center;
}
.badge-score {
  background: rgba(99, 102, 241, 0.15);
  color: #818cf8;
}
.score-detail {
  display: flex;
  gap: 12px;
  margin-bottom: 8px;
  font-size: 12px;
  color: #9ca3af;
}
.result-content {
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  color: #d1d5db;
}
.result-footer {
  margin-top: 8px;
  font-size: 12px;
  color: #6b7280;
  font-family: monospace;
}
</style>
