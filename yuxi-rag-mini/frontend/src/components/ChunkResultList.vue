<template>
  <div>
    <h3 style="margin-bottom: 12px;">Results ({{ results.length }})</h3>
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
.chunk-item { padding: 12px; border: 1px solid #eee; border-radius: 6px; margin-bottom: 8px; }
.chunk-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
.chunk-source { font-weight: 600; }
.chunk-content { font-size: 14px; line-height: 1.6; white-space: pre-wrap; }
</style>
