<div align="center">

[![npm latest package][npm-latest-image]][npm-url]
[![Build Status][ci-image]][ci-url]
[![License][license-image]][license-url]
[![npm downloads][npm-downloads-image]][npm-url]
[![Follow on Twitter][twitter-image]][twitter-url]

# MongoDB RAG

A type-safe MongoDB implementation for Retrieval Augmented Generation with vector search

</div>

## Overview

MongoDB RAG is a semantic memory system for AI applications that provides persistent storage and retrieval of context-aware knowledge using MongoDB Atlas. It enables AI systems to remember past interactions, learn from them, and provide more personalized and context-aware responses over time.

This package implements a RAG (Retrieval Augmented Generation) system that:

- Stores text content with vector embeddings in MongoDB
- Enables semantic search based on meaning, not just keywords
- Organizes memories by user, agent, and session contexts
- Allows rich metadata and categorization
- Supports automatic expiration for temporary memories
- Prevents duplicate content to maintain database efficiency

Check out the [Changelog](./CHANGELOG.md) to see what changed in the last releases.

## Features

- **Vector Storage**: Store and retrieve text with vector embeddings using MongoDB Atlas
- **Semantic Search**: Find semantically similar content using MongoDB's vector search
- **Flexible Organization**: Categorize memories with user, agent, session IDs and custom metadata
- **Rich Filtering**: Filter memories by various attributes including custom metadata
- **Memory Expiration**: Set automatic expiration dates for temporary memories
- **Type Safety**: Fully typed API with TypeScript
- **Batch Operations**: Support for batch updates and deletes
- **Pagination**: Handle large datasets with efficient pagination
- **Duplicate Prevention**: Automatically checks and prevents storing duplicate content
- **Automatic Collections**: Creates required collections and indexes if they don't exist

## Installation

### Prerequisites

- MongoDB Atlas account with vector search capabilities
- Node.js v16+ or Bun
- Google Gemini API key for embeddings and fact extraction

### Install with npm:

```bash
npm install mongo-rag
```

### Install with bun:

```bash
bun add mongo-rag
```

### Environment Setup

Set up the required environment variables:

```bash
# MongoDB connection string
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/database

# Gemini API key for embeddings and fact extraction
GEMINI_API_KEY=your_gemini_api_key
```

## Quick Start

### 1. Connect to MongoDB

```typescript
import mongoose from 'mongoose'

// Connect to MongoDB
await mongoose.connect(process.env.MONGO_URI)
```

### 2. Initialize the Client

```typescript
import { MongoRagClient } from 'mongo-rag'

const client = new MongoRagClient({
  gemini_api_key: process.env.GEMINI_API_KEY,
})
```

### 3. Add Memories

```typescript
// Add a simple memory string
const stringMemory = await client.add('User prefers vegetarian food', {
  user_id: 'user123',
  categories: ['preferences', 'food'],
})

// Add from chat messages
const messages = [
  { role: 'user', content: 'I like science fiction books' },
  { role: 'assistant', content: "I'll recommend sci-fi titles then!" },
]

const messageMemory = await client.add(messages, {
  user_id: 'user123',
  agent_id: 'books_agent',
  categories: ['preferences', 'books'],
  metadata: { genre: 'science fiction' },
})

// Add with expiration date
const temporaryMemory = await client.add(
  'User is currently looking for a birthday gift',
  {
    user_id: 'user123',
    // Memory expires in 7 days
    expiration_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
  }
)
```

### 4. Search for Memories

```typescript
// Semantic search
const searchResults = await client.search('What books does the user like?', {
  user_id: 'user123',
})

// With filters
const filteredResults = await client.search('What food preferences?', {
  user_id: 'user123',
  categories: ['preferences'],
})

// Advanced filtering
const advancedResults = await client.search('User interests', {
  filters: {
    AND: [
      { user_id: 'user123' },
      {
        created_at: {
          gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
        },
      },
    ],
  },
})
```

### 5. Retrieve All Memories

```typescript
// Get all memories for a user
const allMemories = await client.getAll({ user_id: 'user123' })

// Get memories by category
const foodMemories = await client.getAll({
  user_id: 'user123',
  categories: ['food'],
})

// With pagination
const paginatedMemories = await client.getAll({
  user_id: 'user123',
  page: 1,
  page_size: 10,
})
```

## API Overview

The MongoRagClient provides these primary methods:

- `add(content, options)`: Add a new memory
- `search(query, options)`: Find semantically similar memories
- `getAll(options)`: Retrieve memories with filters and pagination
- `get(id)`: Get a specific memory by ID
- `update(id, content, options)`: Update an existing memory
- `delete(id)`: Delete a specific memory
- `deleteAll(options)`: Delete memories matching criteria
- `batchUpdate(updates)`: Update multiple memories at once
- `batchDelete(deletes)`: Delete multiple memories at once

See the [API Documentation](./docs/api.md) for complete details.

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/mguleryuz/mongo-rag.git
cd mongo-rag

# Install dependencies
bun i
```

### Testing

```bash
# Set environment variables for testing
export MONGO_URI=mongodb+srv://...
export GEMINI_API_KEY=your_gemini_api_key
export NODE_ENV=test

# Run tests
bun test
```

### Watching TS Problems:

```bash
bun watch
```

## How to make a release

**For the Maintainer**: Add `NPM_TOKEN` to the GitHub Secrets.

1. PR with changes
2. Merge PR into main
3. Checkout main
4. `git pull`
5. `bun release: '' | alpha | beta` optionally add `-- --release-as minor | major | 0.0.1`
6. Make sure everything looks good (e.g. in CHANGELOG.md)
7. Lastly run `bun release:pub`
8. Done

## License

This package is licensed - see the [LICENSE](./LICENSE) file for details.

[ci-image]: https://badgen.net/github/checks/mguleryuz/mongo-rag/main?label=ci
[ci-url]: https://github.com/mguleryuz/mongo-rag/actions/workflows/ci.yaml
[npm-url]: https://npmjs.org/package/mongo-rag
[twitter-url]: https://twitter.com/mgguleryuz
[twitter-image]: https://img.shields.io/twitter/follow/mgguleryuz.svg?label=follow+Mguleryuz
[license-image]: https://img.shields.io/badge/License-LGPL%20v3-blue
[license-url]: ./LICENSE
[npm-latest-image]: https://img.shields.io/npm/v/mongo-rag/latest.svg
[npm-downloads-image]: https://img.shields.io/npm/dm/mongo-rag.svg
