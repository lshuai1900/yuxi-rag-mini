<template>
  <div class="search-box">
    <input v-model="query" :placeholder="placeholder" @keyup.enter="$emit('search', query)" class="search-input" />
    <button class="btn btn-primary" @click="$emit('search', query)" :disabled="!query.trim()">Search</button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
withDefaults(defineProps<{ placeholder?: string }>(), { placeholder: 'Enter query...' })
const emit = defineEmits<{ search: [query: string] }>()
const query = ref('')
</script>

<style scoped>
.search-box {
  display: flex;
  gap: 12px;
}
.search-input {
  flex: 1;
}
</style>
