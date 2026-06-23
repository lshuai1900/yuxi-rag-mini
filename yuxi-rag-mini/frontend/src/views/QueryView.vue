<template>
  <div>
    <div class="card">
      <h2 style="margin-bottom: 16px;">Query</h2>
      <div v-if="!kbId" class="empty-state">Please select a knowledge base first.</div>
      <div v-else>
        <div class="form-group">
          <label>Search Mode</label>
          <select v-model="searchMode">
            <option value="vector">Vector Search</option>
            <option value="keyword">Keyword Search</option>
            <option value="hybrid">Hybrid Search</option>
          </select>
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
      <h2 style="margin-bottom: 16px;">Results ({{ results.length }})</h2>
      <div v-for="(result, idx) in results" :key="idx" style="padding: 12px; border: 1px solid #eee; border-radius: 6px; margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
          <span style="font-weight: 600;">{{ result.filename || 'Unknown' }}</span>
          <span class="badge badge-info">Score: {{ result.score?.toFixed(4) }}</span>
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
const results = ref<any[]>([])
const searching = ref(false)

async function search() {
  if (!queryText.value.trim() || !props.kbId) return
  searching.value = true
  try {
    const res = await queryKnowledgeBase(props.kbId, {
      query: queryText.value,
      search_mode: searchMode.value,
      top_k: 10,
    })
    results.value = res.data.results || []
  } catch (e: any) {
    const msg = e?.response?.data?.detail || 'Query failed'
    alert(msg)
  } finally {
    searching.value = false
  }
}
</script>
