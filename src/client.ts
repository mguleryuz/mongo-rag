import mongoose from 'mongoose'
import { GoogleGenAI, Type } from '@google/genai'
import EmbeddingModel from '@/embedding.mongo'
import type { IEmbedding, IEmbeddingDocument } from '@/embedding.mongo'
import { OpenAI } from 'openai'

export type Messages =
  | {
      role: string
      content: string
    }[]
  | string

export type Memory = {
  id: string
} & Omit<IEmbedding, 'embedding'>

export type MemorySearchReturnType = {
  memories: Memory[]
  total: number
}

export type MemoryGetAllReturnType = {
  memories: Memory[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export type MemoryDeleteReturnType = {
  success: boolean
  id: string
}

export type MemoryDeleteAllReturnType = {
  success: boolean
  deleted_count: number
}

export type MemoryUsersReturnType = {
  user_ids: string[]
  agent_ids: string[]
  run_ids: string[]
  app_ids: string[]
}

export type MemoryBatchUpdateReturnType = {
  results: (
    | {
        success: boolean
        id: string
        memory: Memory
      }
    | {
        success: boolean
        id: string
        error: string
      }
  )[]
}

export type MemoryBatchDeleteReturnType = {
  results: (
    | {
        success: boolean
        id: string
      }
    | {
        success: boolean
        id: string
        error: string
      }
  )[]
}

export type MemoryFactReturnType = {
  id: string
  memory: string
  user_id?: string
  agent_id?: string
  run_id?: string
  app_id?: string
  metadata: Record<string, any>
  categories: string[]
}

export type MemoryFactsReturnType = {
  results: MemoryFactReturnType[]
}

export type MemoryAddReturnType<T> = T extends string
  ? {
      // Standard memory return type
      id: string
      content: string
      user_id?: string
      agent_id?: string
      run_id?: string
      app_id?: string
    }
  : MemoryFactsReturnType // Facts return type

export class MongoRagClient {
  private readonly openai_api_key: string
  private readonly gemini_api_key: string
  private genAI: GoogleGenAI
  private readonly embeddingDimensions: number = 1536 // Default embedding dimensions

  constructor({
    openai_api_key,
    gemini_api_key,
    embeddingDimensions = 1536,
  }: {
    openai_api_key: string
    gemini_api_key: string
    embeddingDimensions?: number
  }) {
    if (mongoose.connection.readyState !== 1) {
      throw new Error(
        'MongoDB is not connected, please connect first, then initialize the MongoRagClient'
      )
    }

    this.openai_api_key = openai_api_key
    this.gemini_api_key = gemini_api_key
    this.embeddingDimensions = embeddingDimensions
    this.genAI = new GoogleGenAI({ apiKey: this.gemini_api_key })

    // Ensure the vector search index exists
    this.ensureVectorIndex()
  }

  /**
   * Checks if the vector search index exists and creates it if not
   */
  private async ensureVectorIndex(): Promise<void> {
    try {
      // Get the database from the connection
      const db = mongoose.connection.db
      if (!db) {
        throw new Error('MongoDB database connection not available')
      }

      // Get the collection from the model
      const collection = db.collection(EmbeddingModel.collection.name)

      // Check if search index exists
      const searchIndexes = await collection
        .aggregate([{ $listSearchIndexes: {} }])
        .toArray()

      // Check if our mongo_rag_vector_index exists
      const vectorIndexExists = searchIndexes.some(
        (index) => index.name === 'mongo_rag_vector_index'
      )

      if (!vectorIndexExists) {
        console.log('Vector search index does not exist. Creating index...')

        // Define the index configuration
        const indexConfig = {
          name: 'mongo_rag_vector_index',
          type: 'vectorSearch',
          definition: {
            fields: [
              {
                type: 'vector',
                path: 'embedding',
                numDimensions: this.embeddingDimensions,
                similarity: 'cosine',
              },
            ],
          },
        }

        // Create the search index
        await collection.createSearchIndex(indexConfig)
        console.log('Vector search index created successfully!')
      } else {
        console.log('Vector search index already exists.')
      }
    } catch (error: unknown) {
      console.error('Error while ensuring vector index:', error)
      const errorMessage =
        error instanceof Error ? error.message : String(error)
      throw new Error(`Failed to create vector search index: ${errorMessage}`)
    }
  }

  // Generate embedding using Gemini
  private async generateEmbedding(text: string): Promise<number[]> {
    // Note: Replace this with actual Gemini embedding API once available
    // For now, we can use the Gemini model for embeddings or keep OpenAI as a fallback

    // Example implementation using Gemini (once embedding API is available):
    // const embeddingModel = this.genAI.getEmbeddingModel();
    // const result = await embeddingModel.embedContent(text);
    // return result.embedding;

    // Fallback to OpenAI for now
    const openai = new OpenAI({ apiKey: this.openai_api_key })
    const response = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
      dimensions: this.embeddingDimensions,
    })
    return response.data[0].embedding
  }

  /**
   * Check if input is a conversation (array of messages with role/content)
   */
  private isConversation(messages: Messages): messages is {
    role: string
    content: string
  }[] {
    return (
      Array.isArray(messages) &&
      messages.length > 0 &&
      messages.every(
        (msg) => typeof msg === 'object' && 'role' in msg && 'content' in msg
      )
    )
  }

  /**
   * Extract facts from conversation messages using AI
   * @param messages - Array of message objects
   * @returns Array of extracted facts about the user
   */
  private async extractFactsFromConversation(
    messages: {
      role: string
      content: string
    }[]
  ): Promise<string[]> {
    const conversationText = messages
      .map((msg) => `${msg.role}: ${msg.content}`)
      .join('\n')

    try {
      const response = await this.genAI.models.generateContent({
        model: 'gemini-2.0-flash-lite',
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: `Extract distinct factual statements about the user from this conversation. Focus on preferences, traits, beliefs, or any other relevant information about the user.\n\nConversation:\n${conversationText}`,
              },
            ],
          },
        ],
        config: {
          temperature: 0.2,
          topK: 40,
          topP: 0.95,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              facts: {
                type: Type.ARRAY,
                description: 'Array of facts about the user in third person',
                items: {
                  type: Type.STRING,
                },
              },
            },
            required: ['facts'],
          },
        },
      })

      if (response && response.text) {
        const resultText = response.text
        const parsedResult = JSON.parse(resultText)

        if (parsedResult.facts && Array.isArray(parsedResult.facts)) {
          return parsedResult.facts
        }
      }

      return []
    } catch (error) {
      console.error('Error calling Gemini for fact extraction:', error)

      return []
    }
  }

  /**
   * Add a memory
   * @param messages - String content or array of message objects
   * @param options - Configuration options including identifiers and metadata
   * @returns Either a single memory or extracted facts depending on input format
   */
  async add<T extends Messages>(
    messages: T,
    options: {
      user_id?: string
      agent_id?: string
      run_id?: string
      app_id?: string
      metadata?: Record<string, any>
      categories?: string[]
      expiration_date?: string | Date
    } = {}
  ): Promise<MemoryAddReturnType<T>> {
    // Automatically detect if input is a conversation and extract facts
    if (this.isConversation(messages)) {
      const facts = await this.extractFactsFromConversation(messages)

      const results: MemoryFactReturnType[] = []

      for (const fact of facts) {
        const embedding = await this.generateEmbedding(fact)

        // Process expiration date if provided
        let expirationDate: Date | undefined = undefined
        if (options.expiration_date) {
          expirationDate =
            typeof options.expiration_date === 'string'
              ? new Date(options.expiration_date)
              : options.expiration_date
        }

        const memory = new EmbeddingModel({
          content: fact,
          embedding,
          user_id: options.user_id,
          agent_id: options.agent_id,
          run_id: options.run_id,
          app_id: options.app_id,
          metadata: options.metadata || {},
          categories: options.categories || [],
          expiration_date: expirationDate,
        })

        await memory.save()

        results.push({
          id: memory._id.toString(),
          memory: fact,
          user_id: options.user_id,
          agent_id: options.agent_id,
          run_id: options.run_id,
          app_id: options.app_id,
          metadata: options.metadata || {},
          categories: options.categories || [],
        })
      }

      return { results } as MemoryAddReturnType<T>
    }

    // Original behavior for non-conversation input
    const embedding = await this.generateEmbedding(messages)

    // Process expiration date if provided
    let expirationDate: Date | undefined = undefined
    if (options.expiration_date) {
      expirationDate =
        typeof options.expiration_date === 'string'
          ? new Date(options.expiration_date)
          : options.expiration_date
    }

    const memory = new EmbeddingModel({
      content: messages,
      embedding,
      user_id: options.user_id,
      agent_id: options.agent_id,
      run_id: options.run_id,
      app_id: options.app_id,
      metadata: options.metadata || {},
      categories: options.categories || [],
      expiration_date: expirationDate,
    })

    await memory.save()

    return {
      id: memory._id.toString(),
      content: messages,
      user_id: options.user_id,
      agent_id: options.agent_id,
      run_id: options.run_id,
      app_id: options.app_id,
    } as MemoryAddReturnType<T>
  }

  /**
   * Search for similar memories
   * @param query - Search query text
   * @param options - Search configuration options
   */
  async search(
    query: string,
    options: {
      user_id?: string
      agent_id?: string
      run_id?: string
      app_id?: string
      categories?: string[]
      metadata?: Record<string, any>
      threshold?: number
      limit?: number
      filters?: Record<string, any>
    } = {}
  ): Promise<MemorySearchReturnType> {
    const embedding = await this.generateEmbedding(query)

    let results: any[]

    // Check if custom filters are provided
    if (options.filters) {
      // Convert filters to MongoDB query format
      const mongoFilters = this.processFilters(options.filters)

      // Perform search with vectorSearch
      results = await EmbeddingModel.findSimilar(embedding, {
        limit: options.limit,
        ...mongoFilters,
      })
    } else {
      // Standard search with individual parameters
      results = await EmbeddingModel.findSimilar(embedding, {
        user_id: options.user_id,
        agent_id: options.agent_id,
        run_id: options.run_id,
        app_id: options.app_id,
        categories: options.categories,
        metadata: options.metadata,
        limit: options.limit,
      })
    }

    const formattedResults = results.map((r) => ({
      id: r._id.toString(),
      content: r.content,
      user_id: r.user_id,
      agent_id: r.agent_id,
      run_id: r.run_id,
      app_id: r.app_id,
      metadata: r.metadata,
      categories: r.categories,
      created_at: r.created_at,
      updated_at: r.updated_at,
    }))

    return {
      memories: formattedResults,
      total: formattedResults.length,
    }
  }

  /**
   * Get all memories
   * @param options - Options for filtering memories
   */
  async getAll(
    options: {
      user_id?: string
      agent_id?: string
      run_id?: string
      app_id?: string
      categories?: string[]
      keywords?: string
      page?: number
      page_size?: number
      filters?: Record<string, any>
    } = {}
  ): Promise<MemoryGetAllReturnType> {
    const page = options.page || 1
    const pageSize = options.page_size || 100
    const skip = (page - 1) * pageSize

    let query: Record<string, any> = {}

    // Handle custom filters if provided
    if (options.filters) {
      query = this.processFilters(options.filters)
    } else {
      // Standard query filters
      if (options.user_id) query.user_id = options.user_id
      if (options.agent_id) query.agent_id = options.agent_id
      if (options.run_id) query.run_id = options.run_id
      if (options.app_id) query.app_id = options.app_id

      if (options.categories && options.categories.length > 0) {
        query.categories = { $in: options.categories }
      }

      // Handle keyword search
      if (options.keywords) {
        query.content = { $regex: options.keywords, $options: 'i' }
      }
    }

    // Get count for pagination
    const total = await EmbeddingModel.countDocuments(query)

    // Get memories with pagination
    const memories = await EmbeddingModel.find(query)
      .sort({ created_at: -1 })
      .skip(skip)
      .limit(pageSize)

    return {
      memories: memories.map((m) => this.formatMemory(m)),
      total,
      page,
      page_size: pageSize,
      total_pages: Math.ceil(total / pageSize),
    }
  }

  /**
   * Get a specific memory by ID
   * @param memoryId - The ID of the memory to retrieve
   */
  async get(memoryId: string): Promise<Memory> {
    const memory = await EmbeddingModel.findById(memoryId)
    if (!memory) {
      throw new Error(`Memory with ID ${memoryId} not found`)
    }

    return this.formatMemory(memory)
  }

  /**
   * Update a memory
   * @param memoryId - The ID of the memory to update
   * @param content - New content for the memory
   * @param options - Additional update options
   */
  async update(
    memoryId: string,
    content: string,
    options: {
      metadata?: Record<string, any>
      categories?: string[]
      expiration_date?: string | Date
    } = {}
  ): Promise<Memory> {
    const memory = await EmbeddingModel.findById(memoryId)
    if (!memory) {
      throw new Error(`Memory with ID ${memoryId} not found`)
    }

    // Generate new embedding for updated content
    const embedding = await this.generateEmbedding(content)

    // Process expiration date if provided
    if (options.expiration_date) {
      memory.expiration_date =
        typeof options.expiration_date === 'string'
          ? new Date(options.expiration_date)
          : options.expiration_date
    }

    // Update memory
    memory.content = content
    memory.embedding = embedding

    if (options.metadata) {
      memory.metadata = { ...memory.metadata, ...options.metadata }
    }

    if (options.categories) {
      memory.categories = options.categories
    }

    await memory.save()

    return this.formatMemory(memory)
  }

  /**
   * Delete a memory
   * @param memoryId - The ID of the memory to delete
   */
  async delete(memoryId: string): Promise<MemoryDeleteReturnType> {
    const result = await EmbeddingModel.findByIdAndDelete(memoryId)
    if (!result) {
      throw new Error(`Memory with ID ${memoryId} not found`)
    }

    return { success: true, id: memoryId }
  }

  /**
   * Delete all memories matching the criteria
   * @param options - Criteria for memories to delete
   */
  async deleteAll(
    options: {
      user_id?: string
      agent_id?: string
      run_id?: string
      app_id?: string
    } = {}
  ): Promise<MemoryDeleteAllReturnType> {
    const query: Record<string, any> = {}
    if (options.user_id) query.user_id = options.user_id
    if (options.agent_id) query.agent_id = options.agent_id
    if (options.run_id) query.run_id = options.run_id
    if (options.app_id) query.app_id = options.app_id

    const result = await EmbeddingModel.deleteMany(query)

    return {
      success: true,
      deleted_count: result.deletedCount,
    }
  }

  /**
   * Delete users, agents, or runs
   * @param options - Criteria for users/agents/runs to delete
   */
  async delete_users(
    options: {
      user_id?: string
      agent_id?: string
      run_id?: string
      app_id?: string
    } = {}
  ): Promise<{ success: boolean; deleted_count: number }> {
    return this.deleteAll(options)
  }

  /**
   * Get all unique users, agents, and runs in the system
   */
  async users(): Promise<MemoryUsersReturnType> {
    // Group by each ID type and get unique values
    const userIdAgg = await EmbeddingModel.aggregate([
      { $match: { user_id: { $exists: true, $ne: null } } },
      { $group: { _id: '$user_id' } },
    ])

    const agentIdAgg = await EmbeddingModel.aggregate([
      { $match: { agent_id: { $exists: true, $ne: null } } },
      { $group: { _id: '$agent_id' } },
    ])

    const runIdAgg = await EmbeddingModel.aggregate([
      { $match: { run_id: { $exists: true, $ne: null } } },
      { $group: { _id: '$run_id' } },
    ])

    const appIdAgg = await EmbeddingModel.aggregate([
      { $match: { app_id: { $exists: true, $ne: null } } },
      { $group: { _id: '$app_id' } },
    ])

    // Extract values
    const user_ids = userIdAgg.map((u) => u._id)
    const agent_ids = agentIdAgg.map((a) => a._id)
    const run_ids = runIdAgg.map((r) => r._id)
    const app_ids = appIdAgg.map((a) => a._id)

    return { user_ids, agent_ids, run_ids, app_ids }
  }

  /**
   * Batch update multiple memories
   * @param updates - Array of memory updates
   */
  async batchUpdate(
    updates: {
      memory_id: string
      text: string
      metadata?: Record<string, any>
      categories?: string[]
    }[]
  ): Promise<MemoryBatchUpdateReturnType> {
    const results = []

    // Process each update sequentially
    for (const update of updates) {
      try {
        const result = await this.update(update.memory_id, update.text, {
          metadata: update.metadata,
          categories: update.categories,
        })
        results.push({ success: true, id: update.memory_id, memory: result })
      } catch (error) {
        results.push({
          success: false,
          id: update.memory_id,
          error: String(error),
        })
      }
    }

    return { results }
  }

  /**
   * Batch delete multiple memories
   * @param deletes - Array of memory IDs to delete
   */
  async batchDelete(
    deletes: { memory_id: string }[]
  ): Promise<MemoryBatchDeleteReturnType> {
    const results = []

    // Process each delete sequentially
    for (const del of deletes) {
      try {
        await this.delete(del.memory_id)
        results.push({ success: true, id: del.memory_id })
      } catch (error) {
        results.push({
          success: false,
          id: del.memory_id,
          error: String(error),
        })
      }
    }

    return { results }
  }

  /**
   * Reset the client (for testing purposes)
   */
  async reset() {
    if (process.env.NODE_ENV === 'test') {
      await EmbeddingModel.deleteMany({})
      return { success: true, message: 'All memories deleted' }
    } else {
      throw new Error('Reset is only allowed in test mode')
    }
  }

  // Helper methods
  private formatMemory(memory: IEmbeddingDocument) {
    return {
      id: memory._id.toString(),
      content: memory.content,
      user_id: memory.user_id,
      agent_id: memory.agent_id,
      run_id: memory.run_id,
      app_id: memory.app_id,
      metadata: memory.metadata,
      categories: memory.categories,
      expiration_date: memory.expiration_date,
      created_at: memory.created_at,
      updated_at: memory.updated_at,
    }
  }

  private processFilters(filters: any): Record<string, any> {
    // Handle complex filters with logical operators
    if (filters.AND) {
      return { $and: filters.AND.map((f: any) => this.processFilters(f)) }
    }

    if (filters.OR) {
      return { $or: filters.OR.map((f: any) => this.processFilters(f)) }
    }

    // Process field-specific filters
    const result: Record<string, any> = {}

    Object.entries(filters).forEach(([key, value]: [string, any]) => {
      if (key === 'metadata' && typeof value === 'object') {
        // Handle metadata filters
        Object.entries(value).forEach(([metaKey, metaValue]) => {
          result[`metadata.${metaKey}`] = metaValue
        })
      } else if (typeof value === 'object' && value !== null) {
        // Handle operators like in, gte, lte, etc.
        const conditions: Record<string, any> = {}

        Object.entries(value).forEach(([op, opValue]) => {
          switch (op) {
            case 'in':
              conditions.$in = opValue
              break
            case 'gte':
              conditions.$gte = opValue
              break
            case 'lte':
              conditions.$lte = opValue
              break
            case 'gt':
              conditions.$gt = opValue
              break
            case 'lt':
              conditions.$lt = opValue
              break
            case 'ne':
              conditions.$ne = opValue
              break
            case 'contains':
              conditions.$regex = opValue
              conditions.$options = ''
              break
            case 'icontains':
              conditions.$regex = opValue
              conditions.$options = 'i'
              break
            default:
              conditions[op] = opValue
          }
        })

        result[key] = conditions
      } else {
        // Direct value assignment
        result[key] = value
      }
    })

    return result
  }
}
