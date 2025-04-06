import { expect, describe, beforeAll, afterAll, it } from 'bun:test'
import { writeLog } from '../scripts'
import mongoose from 'mongoose'
import { MongoRagClient } from '@/client'
import type { Message } from '@/client'
import { logger } from './logger'

const MONGO_URI = process.env.MONGO_URI
const OPENAI_API_KEY = process.env.OPENAI_API_KEY

describe('#BASIC_EMBEDDING', () => {
  // Connect to MongoDB before all tests
  beforeAll(async () => {
    if (!MONGO_URI) {
      logger.error('MONGO_URI is missing from environment variables')
      throw new Error('MONGO_URI is required for this test')
    }

    try {
      logger.info(`Connecting to MongoDB at ${MONGO_URI}`)
      await mongoose.connect(MONGO_URI)
      logger.info('MongoDB connection successful')
    } catch (error) {
      logger.error('Failed to connect to MongoDB:', error)
      throw error
    }
  })

  // Disconnect after all tests
  afterAll(async () => {
    logger.info('Closing MongoDB connection')
    await mongoose.connection.close()
    logger.info('MongoDB connection closed')
  })

  describe('MongoRagClient memory operations', () => {
    it('should add and retrieve different types of memories', async () => {
      if (!OPENAI_API_KEY) {
        logger.error('OPENAI_API_KEY is missing from environment variables')
        throw new Error('OPENAI_API_KEY is required for this test')
      }

      // Ensure MongoDB is connected before creating the client
      expect(mongoose.connection.readyState).toBe(1) // 1 = connected
      logger.info(
        'MongoDB connection verified, readyState:',
        mongoose.connection.readyState
      )

      // Initialize client
      logger.info('Initializing MongoRagClient with embeddingDimensions=1536')
      const client = new MongoRagClient({
        openai_api_key: OPENAI_API_KEY,
        embeddingDimensions: 1536,
      })

      try {
        // Reset the database to ensure clean state
        logger.info('Resetting database to ensure clean state')
        await client.reset()
        logger.debug('Database reset complete')

        // Test adding different types of memories
        logger.info('Starting memory creation tests')

        // 1. Add memory as string
        logger.debug('Adding string memory with categories: test, string')
        const stringMemory = await client.add(
          'This is a test memory as a string',
          { user_id: 'test_user', categories: ['test', 'string'] }
        )
        logger.info('Created string memory with ID:', stringMemory.id)
        expect(stringMemory.id).toBeDefined()
        expect(stringMemory.content).toBe('This is a test memory as a string')

        // 2. Add memory as messages
        const messages: Message[] = [
          { role: 'user', content: 'I like pizza with extra cheese' },
          { role: 'assistant', content: "That's a great choice!" },
        ]
        logger.debug('Adding message-based memory with metadata', {
          message_count: messages.length,
          metadata: { food_type: 'pizza' },
        })
        const messageMemory = await client.add(messages, {
          user_id: 'test_user',
          agent_id: 'test_agent',
          categories: ['test', 'food'],
          metadata: { food_type: 'pizza' },
        })
        logger.info('Created message memory with ID:', messageMemory.id)
        expect(messageMemory.id).toBeDefined()

        // 3. Add memory with expiration
        const expirationDate = new Date()
        expirationDate.setDate(expirationDate.getDate() + 1) // Expires tomorrow
        logger.debug('Adding memory with expiration date', {
          expiration: expirationDate,
        })

        const expiringMemory = await client.add(
          'This memory will expire tomorrow',
          {
            user_id: 'test_user',
            expiration_date: expirationDate,
          }
        )
        logger.info('Created expiring memory with ID:', expiringMemory.id)
        expect(expiringMemory.id).toBeDefined()

        // Test retrieving memories
        logger.info('Starting memory retrieval tests')

        // 1. Get by ID
        logger.debug('Retrieving memory by ID:', stringMemory.id)
        const retrievedMemory = await client.get(stringMemory.id)
        logger.info('Retrieved memory by ID, content:', retrievedMemory.content)
        expect(retrievedMemory.content).toBe(
          'This is a test memory as a string'
        )
        expect(retrievedMemory.user_id).toBe('test_user')

        // 2. Get all memories for a user
        logger.debug('Retrieving all memories for user_id: test_user')
        const allMemories = await client.getAll({ user_id: 'test_user' })
        logger.info('Retrieved all memories', {
          count: allMemories.total,
          page: allMemories.page,
          total_pages: allMemories.total_pages,
        })
        expect(allMemories.total).toBe(3)
        expect(allMemories.memories.length).toBe(3)

        // 3. Get memories by category
        logger.debug('Retrieving memories by category: food')
        const foodMemories = await client.getAll({
          user_id: 'test_user',
          categories: ['food'],
        })
        logger.info('Retrieved food category memories', {
          count: foodMemories.total,
          metadata: foodMemories.memories[0]?.metadata,
        })
        expect(foodMemories.total).toBe(1)
        expect(foodMemories.memories[0]?.metadata?.food_type).toBe('pizza')

        // 4. Search for similar memories
        logger.debug(
          'Searching for memories similar to: "I enjoy cheese pizza"'
        )
        const searchResults = await client.search('I enjoy cheese pizza', {
          user_id: 'test_user',
        })

        // Log search results
        logger.info('Search results', {
          count: searchResults.memories.length,
          top_match: searchResults.memories[0]?.content,
          top_match_score: searchResults.memories[0]?.score,
        })

        // The pizza memory should be in the results
        const hasPizzaMemory = searchResults.memories.some((memory) =>
          memory.content.includes('pizza')
        )
        logger.debug('Check if pizza memory was found in search results', {
          hasPizzaMemory,
        })
        expect(hasPizzaMemory).toBe(true)

        // Log summary of test results
        const summary = {
          added_memories: 3,
          retrieved_memories: allMemories.total,
          search_results: searchResults.memories.length,
        }

        logger.info('Test summary', summary)
        writeLog({
          content: summary,
          label: 'MEMORY_OPERATIONS_TEST',
          format: 'json',
        })
      } finally {
        // Clean up - ensure this happens even if test fails
        logger.info('Cleaning up - resetting database')
        await client.reset()
        logger.info('Test cleanup complete')
      }
    })
  })
})
