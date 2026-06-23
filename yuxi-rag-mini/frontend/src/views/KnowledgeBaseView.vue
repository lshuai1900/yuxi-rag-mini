<template>
  <div>
    <div class="card">
      <h2 style="margin-bottom: 16px;">Knowledge Bases</h2>
      <div style="display: flex; gap: 12px; margin-bottom: 16px;">
        <input v-model="newKbName" placeholder="Knowledge base name" style="flex: 1" />
        <input v-model="newKbDesc" placeholder="Description (optional)" style="flex: 2" />
        <button class="btn btn-primary" @click="createKb">Create</button>
      </div>
    </div>

    <div v-if="loading" class="empty-state">Loading...</div>
    <div v-else-if="kbs.length === 0" class="empty-state">No knowledge bases yet. Create one above.</div>
    <div v-else>
      <div v-for="kb in kbs" :key="kb.kb_id" class="card" style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <h3 style="cursor: pointer; color: #4361ee;" @click="$emit('select-kb', kb.kb_id)">{{ kb.name }}</h3>
          <p style="color: #666; font-size: 13px;">{{ kb.description || 'No description' }}</p>
          <div style="margin-top: 8px; font-size: 12px; color: #999;">
            <span>ID: {{ kb.kb_id }}</span> &middot;
            <span>Files: {{ kb.stats?.file_count || 0 }}</span> &middot;
            <span>Chunks: {{ kb.stats?.chunk_count || 0 }}</span>
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
