import { Schema, model, Document, Model } from 'mongoose'

// Interface for the Embedding document
export interface IEmbedding {
  content: string
  embedding: number[]
  user_id?: string
  agent_id?: string
  run_id?: string
  app_id?: string
  metadata?: Record<string, any>
  categories?: string[]
  expiration_date?: Date
  created_at: Date
  updated_at: Date
}

export interface IEmbeddingDocument extends IEmbedding, Document {
  _id: Schema.Types.ObjectId
}

// Define static methods for the model
export interface IEmbeddingModel extends Model<IEmbeddingDocument> {
  findSimilar(
    embedding: number[],
    options?: {
      user_id?: string
      agent_id?: string
      run_id?: string
      app_id?: string
      categories?: string[]
      metadata?: Record<string, any>
      threshold?: number
      limit?: number
      numCandidates?: number
    }
  ): Promise<IEmbeddingDocument[]>
}

// Schema definition
const EmbeddingSchema = new Schema<IEmbeddingDocument>(
  {
    content: {
      type: String,
      required: true,
      index: true,
    },
    embedding: {
      type: [Number],
      required: true,
    },
    user_id: {
      type: String,
      index: true,
    },
    agent_id: {
      type: String,
      index: true,
    },
    run_id: {
      type: String,
      index: true,
    },
    app_id: {
      type: String,
      index: true,
    },
    metadata: {
      type: Schema.Types.Mixed,
      default: {},
    },
    categories: {
      type: [String],
      default: [],
      index: true,
    },
    expiration_date: {
      type: Date,
    },
    created_at: {
      type: Date,
      default: Date.now,
      index: true,
    },
    updated_at: {
      type: Date,
      default: Date.now,
      index: true,
    },
  },
  {
    timestamps: { createdAt: 'created_at', updatedAt: 'updated_at' },
  }
)

// Create indexes for efficient querying
EmbeddingSchema.index({ user_id: 1, agent_id: 1 })
EmbeddingSchema.index({ user_id: 1, run_id: 1 })
EmbeddingSchema.index({ expiration_date: 1 }, { expireAfterSeconds: 0 })

// Static method to find similar embeddings using MongoDB's vector search
EmbeddingSchema.statics.findSimilar = function (
  embedding: number[],
  options: {
    user_id?: string
    agent_id?: string
    run_id?: string
    app_id?: string
    categories?: string[]
    metadata?: Record<string, any>
    threshold?: number
    limit?: number
    numCandidates?: number
  } = {}
) {
  const { limit = 10, numCandidates = 100, ...filters } = options

  // Build the filter based on provided options
  const filterConditions: any = {}

  if (filters.user_id) filterConditions.user_id = filters.user_id
  if (filters.agent_id) filterConditions.agent_id = filters.agent_id
  if (filters.run_id) filterConditions.run_id = filters.run_id
  if (filters.app_id) filterConditions.app_id = filters.app_id

  if (filters.categories && filters.categories.length > 0) {
    filterConditions.categories = { $in: filters.categories }
  }

  if (filters.metadata) {
    Object.entries(filters.metadata).forEach(([key, value]) => {
      filterConditions[`metadata.${key}`] = value
    })
  }

  // Add expiration date filter to exclude expired memories
  filterConditions.$or = [
    { expiration_date: { $exists: false } },
    { expiration_date: { $gt: new Date() } },
  ]

  // Initialize the pipeline with $vectorSearch as the first stage
  const pipeline: any[] = [
    // $vectorSearch must be the first stage
    {
      $vectorSearch: {
        index: 'mongo_rag_vector_index',
        queryVector: embedding,
        path: 'embedding',
        numCandidates,
        limit: Math.max(limit * 10, 100), // Get more candidates for post-filtering
      },
    },
  ]

  // Add a $match stage after $vectorSearch if we have filter conditions
  if (Object.keys(filterConditions).length > 0) {
    pipeline.push({ $match: filterConditions })
  }

  // Add a $limit stage to ensure we don't exceed the desired limit after filtering
  pipeline.push({ $limit: limit })

  // Project stage to include relevant fields and score
  pipeline.push({
    $project: {
      _id: 1,
      content: 1,
      user_id: 1,
      agent_id: 1,
      run_id: 1,
      app_id: 1,
      metadata: 1,
      categories: 1,
      expiration_date: 1,
      created_at: 1,
      updated_at: 1,
      similarity: { $meta: 'vectorSearchScore' },
    },
  })

  // Perform vector similarity search
  return this.aggregate(pipeline)
}

// Create and export the model with proper typing
const EmbeddingModel = model<IEmbeddingDocument, IEmbeddingModel>(
  'embeddings',
  EmbeddingSchema
)

export default EmbeddingModel
