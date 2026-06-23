<template>
  <div>
    <h3 style="margin-bottom: 12px; font-size: 14px; color: var(--text-secondary);">Results <span style="color: var(--text-muted);">({{ results.length }})</span></h3>
    <div v-for="(result, idx) in results" :key="idx" class="chunk-item">
      <div class="chunk-header">
        <span class="chunk-source">{{ result.metadata?.source || 'Unknown' }}</span>
        <span class="badge badge-info">Score: {{ result.score?.toFixed(4) }}</span>
      </div>
      <p class="chunk-content">{{ result.content }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{ results: any[] }>()
</script>

<style scoped>
.chunk-item {
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 8px;
  transition: border-color 0.2s;
}
.chunk-item:hover {
  border-color: #3d3d50;
}
.chunk-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}
.chunk-source {
  font-weight: 600;
  font-size: 14px;
}
.chunk-content {
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  color: var(--text-primary);
}
</style>
