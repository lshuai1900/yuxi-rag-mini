import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

// Knowledge Base APIs
export const listKnowledgeBases = () => api.get('/kb')
export const createKnowledgeBase = (data: { name: string; description?: string; embedding_model_spec?: string }) =>
  api.post('/kb', data)
export const getKnowledgeBase = (kbId: string, includeFiles = false) =>
  api.get(`/kb/${kbId}?include_files=${includeFiles}`)
export const deleteKnowledgeBase = (kbId: string) => api.delete(`/kb/${kbId}`)

// File APIs
export const uploadFile = (kbId: string, file: File) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post(`/kb/${kbId}/files/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}
export const parseFile = (kbId: string, fileId: string) =>
  api.post(`/kb/${kbId}/files/${fileId}/parse`)
export const indexFile = (kbId: string, fileId: string) =>
  api.post(`/kb/${kbId}/files/${fileId}/index`)
export const ingestFile = (kbId: string, fileId: string) =>
  api.post(`/kb/${kbId}/files/${fileId}/ingest`)
export const listFiles = (kbId: string) => api.get(`/kb/${kbId}/files`)
export const deleteFile = (kbId: string, fileId: string) =>
  api.delete(`/kb/${kbId}/files/${fileId}`)

// Query APIs
export interface RetrievalConfig {
  search_mode?: 'vector' | 'keyword' | 'hybrid'
  final_top_k?: number
  similarity_threshold?: number
  bm25_top_k?: number
  vector_weight?: number
  bm25_weight?: number
  bm25_drop_ratio_search?: number
  recall_top_k?: number
  use_reranker?: boolean
}

export const queryKnowledgeBase = (kbId: string, data: {
  query: string
  retrieval_config?: RetrievalConfig
}) => api.post(`/kb/${kbId}/query`, data)

// Chat APIs (RAG QA: retrieve -> LLM -> answer + sources)
export interface Citation {
  index: number
  chunk_id: string
  file_id: string
  filename: string
  chunk_index: number
  score: number
  content: string
}

export interface ChatResponse {
  query: string
  answer: string
  sources: Citation[]
  search_mode: string
}

export const chatWithKnowledgeBase = (kbId: string, data: {
  query: string
  search_mode?: 'vector' | 'keyword' | 'hybrid'
  top_k?: number
  similarity_threshold?: number
  temperature?: number
  max_tokens?: number
  retrieval_config?: RetrievalConfig
}) => api.post<ChatResponse>(`/kb/${kbId}/chat`, data)

// Health
export const healthCheck = () => api.get('/health')
