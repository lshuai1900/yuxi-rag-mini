<template>
  <div>
    <div class="card">
      <div class="section-header">
        <h2>Upload Files</h2>
        <span v-if="kbId" class="kb-badge">KB: {{ kbId }}</span>
      </div>
      <div v-if="!kbId" class="empty-state">Select a knowledge base to manage files.</div>
      <div v-else>
        <div class="drop-zone" @click="($refs.fileInput as any)?.click()" @dragover.prevent @drop.prevent="handleDrop">
          <div class="drop-zone-content">
            <span class="drop-icon">&#8686;</span>
            <p class="drop-text">Click or drag files here to upload</p>
            <p class="drop-hint">PDF, DOCX, MD, TXT</p>
          </div>
          <input ref="fileInput" type="file" multiple accept=".pdf,.docx,.md,.txt,.markdown" style="display: none" @change="handleFiles" />
        </div>
      </div>
    </div>

    <div class="card" v-if="kbId">
      <div class="section-header">
        <h2>Files</h2>
        <button class="btn btn-sm btn-ghost" @click="loadFiles">Refresh</button>
      </div>
      <table v-if="files.length > 0">
        <thead>
          <tr>
            <th>Filename</th>
            <th>Type</th>
            <th>Status</th>
            <th>Chunks</th>
            <th>Size</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="file in files" :key="file.file_id">
            <td class="filename-cell">{{ file.filename }}</td>
            <td><span class="badge badge-info">{{ file.file_type || '-' }}</span></td>
            <td>
              <span class="badge" :class="statusClass(file.status)">{{ statusLabel(file.status) }}</span>
              <div v-if="file.failed_reason || file.error_message" class="error-msg">
                {{ truncateError(file.failed_reason || file.error_message) }}
              </div>
            </td>
            <td>{{ file.chunk_count || 0 }}</td>
            <td>{{ formatSize(file.size) }}</td>
            <td style="display: flex; gap: 4px;">
              <button v-if="canIndex(file.status)" class="btn btn-primary btn-sm" @click="doIndex(file.file_id)" :disabled="indexing[file.file_id]">
                {{ indexing[file.file_id] ? 'Indexing...' : 'Index' }}
              </button>
              <button class="btn btn-danger btn-sm" @click="deleteFile_(file.file_id)">Delete</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty-state">No files uploaded yet.</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { uploadFile, listFiles, indexFile, deleteFile } from '../api/kb'

const props = defineProps<{ kbId: string }>()
const files = ref<any[]>([])
const indexing = ref<Record<string, boolean>>({})

async function loadFiles() {
  if (!props.kbId) return
  try {
    const res = await listFiles(props.kbId)
    files.value = Object.values(res.data)
  } catch (e: any) {
    console.error('Failed to load files', e)
  }
}

async function handleFiles(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files) return
  for (const file of Array.from(input.files)) {
    try {
      await uploadFile(props.kbId, file)
    } catch (e: any) {
      alert(getErrorMessage(e, 'Upload failed'))
    }
  }
  await loadFiles()
}

async function handleDrop(event: DragEvent) {
  const files = event.dataTransfer?.files
  if (!files) return
  for (const file of Array.from(files)) {
    try {
      await uploadFile(props.kbId, file)
    } catch (e: any) {
      alert(getErrorMessage(e, 'Upload failed'))
    }
  }
  await loadFiles()
}

function canIndex(status: string) {
  return ['uploaded', 'parsed', 'error_parsing', 'error_indexing'].includes(status)
}

async function doIndex(fileId: string) {
  indexing.value[fileId] = true
  try {
    await indexFile(props.kbId, fileId)
    await loadFiles()
  } catch (e: any) {
    alert(getErrorMessage(e, 'Index failed'))
  } finally {
    indexing.value[fileId] = false
  }
}

async function deleteFile_(fileId: string) {
  if (!confirm('Delete this file?')) return
  try {
    await deleteFile(props.kbId, fileId)
    await loadFiles()
  } catch (e: any) {
    alert(getErrorMessage(e, 'Delete failed'))
  }
}

function statusClass(status: string) {
  if (status === 'indexed') return 'badge-success'
  if (status === 'uploaded' || status === 'parsed') return 'badge-info'
  if (status === 'error_parsing' || status === 'error_indexing') return 'badge-danger'
  if (['parsing', 'indexing'].includes(status)) return 'badge-warning'
  return 'badge-warning'
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    uploaded: 'Uploaded',
    parsing: 'Parsing',
    parsed: 'Parsed',
    error_parsing: 'Parse Error',
    indexing: 'Indexing',
    indexed: 'Indexed',
    error_indexing: 'Index Error',
  }
  return labels[status] || status
}

function formatSize(bytes: number | undefined): string {
  if (!bytes) return '-'
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

function truncateError(msg: string | undefined): string {
  if (!msg) return ''
  return msg.length > 80 ? msg.substring(0, 80) + '...' : msg
}

function getErrorMessage(e: any, fallback: string): string {
  const detail = e?.response?.data?.detail
  if (typeof detail === 'object' && detail?.message) return detail.message
  if (typeof detail === 'string') return detail
  return fallback
}

watch(() => props.kbId, loadFiles, { immediate: true })
</script>

<style scoped>
.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.section-header h2 {
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
.btn-ghost {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-secondary);
}
.btn-ghost:hover {
  border-color: var(--accent);
  color: var(--accent-hover);
}
.drop-zone {
  border: 2px dashed var(--border);
  border-radius: 8px;
  padding: 30px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
}
.drop-zone:hover {
  border-color: var(--accent);
  background: rgba(99, 102, 241, 0.05);
}
.drop-zone-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}
.drop-icon {
  font-size: 24px;
  color: var(--text-muted);
}
.drop-text {
  color: var(--text-secondary);
  font-size: 14px;
}
.drop-hint {
  font-size: 12px;
  color: var(--text-muted);
}
.filename-cell {
  font-weight: 500;
}
.error-msg {
  font-size: 11px;
  color: #f87171;
  margin-top: 2px;
}
</style>
