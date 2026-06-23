<template>
  <div>
    <div class="card">
      <h2 style="margin-bottom: 16px;">Query</h2>
      <div v-if="!kbId" class="empty-state">Please select a knowledge base first.</div>
      <div v-else>
        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px;">
          <div class="form-group" style="min-width: 150px;">
            <label>Search Mode</label>
            <select v-model="searchMode">
              <option value="vector">Vector Search</option>
              <option value="keyword">Keyword Search</option>
              <option value="hybrid">Hybrid Search</option>
            </select>
          </div>
          <div class="form-group" style="min-width: 80px;">
            <label>Top K</label>
            <input type="number" v-model.number="topK" min="1" max="100" />
          </div>
          <div class="form-group" style="min-width: 120px;">
            <label>Similarity Threshold</label>
            <input type="number" v-model.number="similarityThreshold" min="0" max="1" step="0.1" />
          </div>
          <div class="form-group" style="display: flex; align-items: flex-end; padding-bottom: 4px;">
            <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
              <input type="checkbox" v-model="enableRerank" />
              Enable Rerank
            </label>
          </div>
          <div class="form-group" style="display: flex; align-items: flex-end; padding-bottom: 4px;">
            <label style="display: flex; align-items: center; gap: 6px; cursor: not-allowed; opacity: 0.5;" title="GraphRAG is reserved but not implemented yet.">
              <input type="checkbox" :checked="false" disabled />
              GraphRAG (N/A)
            </label>
          </div>
        </div>
        <div style="display: flex; gap: 12px;">
          <input v-model="queryText" placeholder="Enter your query..." @keyup.enter="search" style="flex: 1" />
          <button class="btn btn-primary" @click="search" :disabled="!queryText.trim() || searching">
            {{ searching ? 'Searching...' : 'Search' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="results.length > 0" class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <h2>Results ({{ results.length }})</h2>
        <span v-if="rerankInfo" style="font-size: 12px; color: #666;">
          Reranked: {{ rerankInfo.reranked ? 'Yes' : 'No' }} ({{ rerankInfo.reranker }})
        </span>
      </div>
      <div v-for="(result, idx) in results" :key="idx" style="padding: 12px; border: 1px solid #eee; border-radius: 6px; margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <div style="display: flex; gap: 8px; align-items: center;">
            <span style="font-weight: 600;">{{ result.filename || 'Unknown' }}</span>
            <span v-if="result.metadata?.page_number" class="badge badge-info">Page {{ result.metadata.page_number }}</span>
          </div>
          <div style="display: flex; gap: 8px; align-items: center;">
            <span class="badge badge-info">Score: {{ result.score?.toFixed(4) }}</span>
          </div>
        </div>
        <!-- Score detail for hybrid search -->
        <div v-if="result.score_detail" style="display: flex; gap: 12px; margin-bottom: 8px; font-size: 12px; color: #666;">
          <span>Vector: {{ result.score_detail.vector_score?.toFixed(4) }}</span>
          <span>Keyword: {{ result.score_detail.keyword_score?.toFixed(4) }}</span>
          <span>Final: {{ result.score_detail.final_score?.toFixed(4) }}</span>
          <span class="badge" :class="result.score_detail.source === 'hybrid' ? 'badge-warning' : 'badge-info'">{{ result.score_detail.source }}</span>
        </div>
        <p style="font-size: 14px; line-height: 1.6; white-space: pre-wrap;">{{ result.content }}</p>
        <div style="margin-top: 8px; font-size: 12px; color: #999;">
          Chunk: {{ result.chunk_id }} &middot; File: {{ result.file_id }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { queryKnowledgeBase } from '../api/kb'

const props = defineProps<{ kbId: string }>()
const queryText = ref('')
const searchMode = ref<'vector' | 'keyword' | 'hybrid'>('vector')
const topK = ref(10)
const similarityThreshold = ref(0.0)
const enableRerank = ref(false)
const results = ref<any[]>([])
const searching = ref(false)
const rerankInfo = ref<{ reranked: boolean; reranker: string } | null>(null)

async function search() {
  if (!queryText.value.trim() || !props.kbId) return
  searching.value = true
  try {
    const res = await queryKnowledgeBase(props.kbId, {
      query: queryText.value,
      search_mode: searchMode.value,
      top_k: topK.value,
      similarity_threshold: similarityThreshold.value,
      enable_rerank: enableRerank.value,
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
