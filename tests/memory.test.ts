import { expect, describe, it, beforeAll, afterAll } from 'bun:test'
import { writeLog } from '../scripts'

import mongoose from 'mongoose'
import { MongoRagClient } from '@/client'
import type { MemoryFactsReturnType } from '@/client'
import { logger } from './logger'

const MONGO_URI = process.env.MONGO_URI
const OPENAI_API_KEY = process.env.OPENAI_API_KEY
const GEMINI_API_KEY = process.env.GEMINI_API_KEY

describe('Memory Conversation Facts Extraction', () => {
  let mongoRagClient: MongoRagClient

  // Connect to database before tests
  beforeAll(async () => {
    if (!MONGO_URI) {
      logger.error('MONGO_URI is missing from environment variables')
      throw new Error('MONGO_URI is required for this test')
    }

    try {
      logger.info(`Connecting to MongoDB at ${MONGO_URI}`)
      await mongoose.connect(MONGO_URI)
      logger.info('MongoDB connection successful')

      if (!OPENAI_API_KEY) {
        throw new Error('OpenAI API key is not defined')
      }

      if (!GEMINI_API_KEY) {
        throw new Error('Gemini API key is not defined')
      }

      mongoRagClient = new MongoRagClient({
        gemini_api_key: GEMINI_API_KEY,
      })
    } catch (error) {
      logger.error('Failed to connect to MongoDB:', error)
      throw error
    }
  })

  // Disconnect after tests
  afterAll(async () => {
    logger.info('Closing MongoDB connection')
    await mongoose.connection.close()
    logger.info('MongoDB connection closed')
  })

  it(
    'should extract facts from conversation and make them searchable',
    async () => {
      // Sample conversation between user and assistant
      const conversation = [
        {
          role: 'user',
          content:
            "I'm planning to watch a movie tonight. Any recommendations?",
        },
        {
          role: 'assistant',
          content: 'How about a thriller movie? They can be quite engaging.',
        },
        {
          role: 'user',
          content:
            "I'm not a big fan of thriller movies but I love sci-fi movies.",
        },
        {
          role: 'assistant',
          content:
            "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future.",
        },
      ]

      try {
        logger.info('Starting to add conversation')
        // Save the conversation
        const result = await mongoRagClient.add(conversation, {
          user_id: 'test-user-123',
          agent_id: 'movie-agent',
          metadata: { category: 'movie_preferences' },
        })
        logger.info('Conversation added successfully')

        // Log the result
        writeLog({
          content: result,
          label: 'conversation-facts-extraction-result',
          format: 'json',
        })

        // Expect result to have a structure with extracted facts
        expect(result).toBeDefined()

        // Verify result is in the MemoryFactsReturnType format
        expect('results' in result).toBe(true)

        const factsResult = result as MemoryFactsReturnType
        expect(Array.isArray(factsResult.results)).toBe(true)
        expect(factsResult.results.length).toBeGreaterThan(0)

        logger.info(
          `Extracted ${factsResult.results.length} facts from conversation`
        )
        factsResult.results.forEach((fact, index) => {
          logger.info(`Fact ${index + 1}: ${fact.memory}`)
        })

        logger.info('Starting semantic search for movie preferences')
        // Search for facts about movie preferences
        const searchResult = await mongoRagClient.search(
          'What kind of movies does the user like?',
          {
            user_id: 'test-user-123',
          }
        )
        logger.info(
          `Found ${searchResult.memories.length} memories about movie preferences`
        )

        // Log the search result
        writeLog({
          content: searchResult,
          label: 'conversation-facts-search-result',
          format: 'json',
        })

        // Expect to find relevant memories
        expect(searchResult).toBeDefined()
        expect(searchResult.memories).toBeDefined()
        expect(searchResult.memories.length).toBeGreaterThan(0)

        logger.info('Starting semantic search for thriller preferences')
        // Search for facts about specific genre preferences
        const genreSearchResult = await mongoRagClient.search(
          'Does the user like thriller movies?',
          {
            user_id: 'test-user-123',
          }
        )
        logger.info(
          `Found ${genreSearchResult.memories.length} memories about thriller preferences`
        )

        // Log the genre-specific search result
        writeLog({
          content: genreSearchResult,
          label: 'genre-specific-search-result',
          format: 'json',
        })

        // Expect to find relevant facts about genre preferences
        expect(genreSearchResult).toBeDefined()
        expect(genreSearchResult.memories).toBeDefined()
        expect(genreSearchResult.memories.length).toBeGreaterThan(0)

        // Log summary of test results
        const summary = {
          conversation_length: conversation.length,
          facts_extracted: factsResult.results.length,
          search_results: searchResult.memories.length,
          genre_search_results: genreSearchResult.memories.length,
        }

        logger.info('Test summary', summary)
        writeLog({
          content: summary,
          label: 'CONVERSATION_FACTS_TEST',
          format: 'json',
        })
      } finally {
        // Clean up - ensure this happens even if test fails
        logger.info('Cleaning up test data')
        // await mongoRagClient.deleteAll({ user_id: 'test-user-123' })
        logger.info('Test cleanup complete')
      }
    },
    {
      timeout: 30000, // Increase timeout for API calls
    }
  )
})
