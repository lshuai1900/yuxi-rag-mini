<template>
  <div class="file-uploader">
    <div class="drop-zone" @click="triggerUpload" @dragover.prevent @drop.prevent="handleDrop">
      <span class="drop-icon">&#8686;</span>
      <p>Click or drag files here</p>
      <input ref="fileInput" type="file" multiple :accept="accept" style="display: none" @change="handleFiles" />
    </div>
  </div>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<{ accept?: string }>(), { accept: '.pdf,.docx,.md,.txt' })
const emit = defineEmits<{ files: [files: File[]] }>()

function triggerUpload() {
  (document.querySelector('.file-uploader input') as HTMLInputElement)?.click()
}

function handleFiles(event: Event) {
  const input = event.target as HTMLInputElement
  if (input.files) emit('files', Array.from(input.files))
}

function handleDrop(event: DragEvent) {
  const files = event.dataTransfer?.files
  if (files) emit('files', Array.from(files))
}
</script>

<style scoped>
.drop-zone {
  border: 2px dashed var(--border);
  border-radius: 8px;
  padding: 30px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  color: var(--text-secondary);
}
.drop-zone:hover {
  border-color: var(--accent);
  background: rgba(99, 102, 241, 0.05);
}
.drop-icon {
  font-size: 24px;
  color: var(--text-muted);
  display: block;
  margin-bottom: 4px;
}
</style>
