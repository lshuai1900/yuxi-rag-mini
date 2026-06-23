<template>
  <div class="file-uploader">
    <div class="drop-zone" @click="triggerUpload" @dragover.prevent @drop.prevent="handleDrop">
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
  border: 2px dashed #ddd;
  border-radius: 8px;
  padding: 30px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s;
}
.drop-zone:hover { border-color: #4361ee; }
</style>
