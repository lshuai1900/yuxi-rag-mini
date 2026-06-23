<template>
  <div id="app">
    <header class="header">
      <h1>Yuxi RAG Mini</h1>
      <nav>
        <button @click="currentView = 'kb'" :class="{ active: currentView === 'kb' }">Knowledge Bases</button>
        <button @click="currentView = 'upload'" :class="{ active: currentView === 'upload' }">Upload</button>
        <button @click="currentView = 'query'" :class="{ active: currentView === 'query' }">Query</button>
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
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }
#app { max-width: 1200px; margin: 0 auto; padding: 20px; }
.header { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; background: #fff; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.header h1 { font-size: 20px; color: #1a1a2e; }
nav { display: flex; gap: 8px; }
nav button { padding: 8px 16px; border: 1px solid #ddd; border-radius: 6px; background: #fff; cursor: pointer; font-size: 14px; transition: all 0.2s; }
nav button:hover { background: #f0f0f0; }
nav button.active { background: #4361ee; color: #fff; border-color: #4361ee; }
.card { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; transition: all 0.2s; }
.btn-primary { background: #4361ee; color: #fff; }
.btn-primary:hover { background: #3a56d4; }
.btn-danger { background: #e74c3c; color: #fff; }
.btn-danger:hover { background: #c0392b; }
.btn-success { background: #27ae60; color: #fff; }
.btn-success:hover { background: #219a52; }
.btn-sm { padding: 4px 10px; font-size: 12px; }
input, select, textarea { padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; width: 100%; }
input:focus, select:focus, textarea:focus { outline: none; border-color: #4361ee; }
label { display: block; margin-bottom: 4px; font-size: 14px; font-weight: 500; }
.form-group { margin-bottom: 12px; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }
th { background: #f8f9fa; font-weight: 600; font-size: 13px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; }
.badge-success { background: #d4edda; color: #155724; }
.badge-warning { background: #fff3cd; color: #856404; }
.badge-danger { background: #f8d7da; color: #721c24; }
.badge-info { background: #d1ecf1; color: #0c5460; }
.empty-state { text-align: center; padding: 40px; color: #999; }
</style>
