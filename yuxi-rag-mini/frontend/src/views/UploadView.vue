<template>
  <div>
    <div class="card">
      <h2 style="margin-bottom: 16px;">Upload Files</h2>
      <div v-if="!kbId" class="empty-state">Please select a knowledge base first.</div>
      <div v-else>
        <p style="margin-bottom: 12px; color: #666;">Knowledge Base: <strong>{{ kbId }}</strong></p>
        <div style="border: 2px dashed #ddd; border-radius: 8px; padding: 30px; text-align: center; cursor: pointer;"
             @click="($refs.fileInput as any)?.click()"
             @dragover.prevent @drop.prevent="handleDrop">
          <p>Click or drag files here to upload</p>
          <p style="font-size: 12px; color: #999;">Supports PDF, Word (.docx), Markdown (.md), TXT</p>
          <input ref="fileInput" type="file" multiple accept=".pdf,.docx,.md,.txt,.markdown" style="display: none" @change="handleFiles" />
        </div>
      </div>
    </div>

    <div class="card" v-if="kbId">
      <h2 style="margin-bottom: 16px;">Files</h2>
      <button class="btn btn-sm" style="margin-bottom: 12px;" @click="loadFiles">Refresh</button>
      <table v-if="files.length > 0">
        <thead>
          <tr>
            <th>Filename</th>
            <th>Status</th>
            <th>Chunks</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="file in files" :key="file.file_id">
            <td>{{ file.filename }}</td>
            <td>
              <span class="badge" :class="statusClass(file.status)">{{ file.status }}</span>
            </td>
            <td>{{ file.chunk_count || 0 }}</td>
            <td>
              <button v-if="file.status === 'uploaded'" class="btn btn-primary btn-sm" @click="ingest(file.file_id)">Ingest</button>
              <button v-if="file.status === 'parsed'" class="btn btn-success btn-sm" @click="indexFile_(file.file_id)">Index</button>
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
import { uploadFile, listFiles, ingestFile, indexFile, deleteFile } from '../api/kb'

const props = defineProps<{ kbId: string }>()
const files = ref<any[]>([])

async function loadFiles() {
  if (!props.kbId) return
  try {
    const res = await listFiles(props.kbId)
    files.value = Object.values(res.data)
  } catch (e) {
    console.error('Failed to load files', e)
  }
}

async function handleFiles(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files) return
  for (const file of Array.from(input.files)) {
    try {
      await uploadFile(props.kbId, file)
    } catch (e) {
      console.error('Upload failed', e)
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
    } catch (e) {
      console.error('Upload failed', e)
    }
  }
  await loadFiles()
}

async function ingest(fileId: string) {
  try {
    await ingestFile(props.kbId, fileId)
    await loadFiles()
  } catch (e) {
    console.error('Ingest failed', e)
  }
}

async function indexFile_(fileId: string) {
  try {
    await indexFile(props.kbId, fileId)
    await loadFiles()
  } catch (e) {
    console.error('Index failed', e)
  }
}

async function deleteFile_(fileId: string) {
  if (!confirm('Delete this file?')) return
  try {
    await deleteFile(props.kbId, fileId)
    await loadFiles()
  } catch (e) {
    console.error('Delete failed', e)
  }
}

function statusClass(status: string) {
  if (status === 'indexed' || status === 'done') return 'badge-success'
  if (status === 'uploaded' || status === 'parsed') return 'badge-info'
  if (status?.startsWith('error')) return 'badge-danger'
  return 'badge-warning'
}

watch(() => props.kbId, loadFiles, { immediate: true })
</script>
