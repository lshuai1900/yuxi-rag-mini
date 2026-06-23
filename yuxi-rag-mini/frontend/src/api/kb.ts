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
export const queryKnowledgeBase = (kbId: string, data: {
  query_text: string
  search_mode?: 'vector' | 'keyword' | 'hybrid'
  top_k?: number
  similarity_threshold?: number
}) => api.post(`/kb/${kbId}/query`, data)

// Health
export const healthCheck = () => api.get('/health')
