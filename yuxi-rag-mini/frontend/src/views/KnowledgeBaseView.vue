<template>
  <div>
    <div class="card">
      <div class="section-header">
        <h2>Knowledge Bases</h2>
      </div>
      <div class="create-row">
        <input v-model="newKbName" placeholder="Knowledge base name" class="input-name" />
        <input v-model="newKbDesc" placeholder="Description (optional)" class="input-desc" />
        <button class="btn btn-primary" @click="createKb" :disabled="!newKbName.trim()">Create</button>
      </div>
    </div>

    <div v-if="loading" class="empty-state">Loading...</div>
    <div v-else-if="kbs.length === 0" class="empty-state">No knowledge bases. Create one above to get started.</div>
    <div v-else>
      <div v-for="kb in kbs" :key="kb.kb_id" class="card kb-card">
        <div class="kb-info">
          <h3 class="kb-name" @click="$emit('select-kb', kb.kb_id)">{{ kb.name }}</h3>
          <p class="kb-desc">{{ kb.description || 'No description' }}</p>
          <div class="kb-stats">
            <span class="stat-item"><span class="stat-label">ID</span> <span class="stat-value mono">{{ kb.kb_id }}</span></span>
            <span class="stat-divider">&middot;</span>
            <span class="stat-item"><span class="stat-label">Files</span> <span class="stat-value">{{ kb.stats?.file_count || 0 }}</span></span>
            <span class="stat-divider">&middot;</span>
            <span class="stat-item"><span class="stat-label">Chunks</span> <span class="stat-value">{{ kb.stats?.chunk_count || 0 }}</span></span>
          </div>
        </div>
        <button class="btn btn-danger btn-sm" @click="deleteKb(kb.kb_id)">Delete</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { listKnowledgeBases, createKnowledgeBase, deleteKnowledgeBase } from '../api/kb'

const emit = defineEmits<{ 'select-kb': [kbId: string] }>()

const kbs = ref<any[]>([])
const loading = ref(true)
const newKbName = ref('')
const newKbDesc = ref('')

async function loadKbs() {
  loading.value = true
  try {
    const res = await listKnowledgeBases()
    kbs.value = res.data.databases || []
  } catch (e) {
    console.error('Failed to load knowledge bases', e)
  } finally {
    loading.value = false
  }
}

async function createKb() {
  if (!newKbName.value.trim()) return
  try {
    await createKnowledgeBase({ name: newKbName.value, description: newKbDesc.value })
    newKbName.value = ''
    newKbDesc.value = ''
    await loadKbs()
  } catch (e) {
    console.error('Failed to create knowledge base', e)
  }
}

async function deleteKb(kbId: string) {
  if (!confirm('Delete this knowledge base?')) return
  try {
    await deleteKnowledgeBase(kbId)
    await loadKbs()
  } catch (e) {
    console.error('Failed to delete knowledge base', e)
  }
}

onMounted(loadKbs)
</script>

<style scoped>
.section-header {
  margin-bottom: 16px;
}
.section-header h2 {
  margin: 0;
}
.create-row {
  display: flex;
  gap: 12px;
}
.input-name {
  flex: 1;
}
.input-desc {
  flex: 2;
}
.kb-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.kb-name {
  cursor: pointer;
  color: var(--accent-hover);
  font-size: 15px;
  margin-bottom: 4px;
  transition: color 0.15s;
}
.kb-name:hover {
  color: #fff;
}
.kb-desc {
  color: var(--text-secondary);
  font-size: 13px;
  margin-bottom: 8px;
}
.kb-stats {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
}
.stat-item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.stat-label {
  color: var(--text-muted);
}
.stat-value {
  color: var(--text-secondary);
}
.stat-value.mono {
  font-family: monospace;
  font-size: 11px;
}
.stat-divider {
  color: var(--text-muted);
  margin: 0 4px;
}
</style>
