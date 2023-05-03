import express from "express"
import cors from "cors"
import bodyParser from "body-parser"
import dotenv from "dotenv"
import { OpenAIEmbeddings } from "langchain/embeddings/openai"
import { VectorDBQAChain } from "langchain/chains"
import { PineconeStore } from "langchain/vectorstores/pinecone"
import { PineconeClient } from "@pinecone-database/pinecone"
import { OpenAI } from "langchain/llms/openai"
import { VectorOperationsApi } from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch"

dotenv.config()
const client = new PineconeClient()

const app = express()

app.use(bodyParser.json())
app.use(cors())

// let qaChain: VectorDBQAChain
let openAI: OpenAI
let embeddings: OpenAIEmbeddings
let pineconeIndex: VectorOperationsApi

const initializeData = async () => {
  try {
    console.log("Initializing data")
    // const fileContents = fs.readFileSync(
    //   "src/rules-docs/magic-the-gathering.txt",
    //   "utf8"
    // )
    // const docs = fileContents.split("\r\r").filter(Boolean)
    // const docs = await textSplitter.splitText(fileContents)
    // docs = await mySplitter.createDocuments([fileContents])

    openAI = new OpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY,
    })

    await client.init({
      apiKey: process.env.PINECONE_API_KEY,
      environment: process.env.PINECONE_ENVIRONMENT,
    })
    pineconeIndex = client.Index(process.env.PINECONE_INDEX)

    embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    })

    // Do this in the route handler and pass namespace to dbConfig
    // const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    //   pineconeIndex,
    // })

    // const docSearch = await MemoryVectorStore.fromTexts(
    //   docs,
    //   docs.map((d, i) => ({ id: i, game: "magic-the-gathering" })),
    //   embeddings
    // )

    // qaChain = VectorDBQAChain.fromLLM(openAI, vectorStore, {
    //   returnSourceDocuments: true,
    // })
    console.log("Initialization complete")
  } catch (e) {
    console.log("Error initializing data")
    console.log(e)
  }
}

initializeData()

app.get("/", async (req, res) => {
  res.json({ status: 200 }).status(200)
})

app.get("/supported-games", async (req, res) => {
  let games: string[] = []
  const stats = await pineconeIndex.describeIndexStats({
    describeIndexStatsRequest: {},
  })
  if (stats?.namespaces) {
    games = Object.keys(stats.namespaces)
  }
  res.json({ games }).status(200)
})

app.post("/query", async (req, res) => {
  const { query, game } = req.body
  console.log({ query, game })
  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
    namespace: game,
  })

  // const docSearch = await MemoryVectorStore.fromTexts(
  //   docs,
  //   docs.map((d, i) => ({ id: i, game: "magic-the-gathering" })),
  //   embeddings
  // )

  const qaChain = VectorDBQAChain.fromLLM(openAI, vectorStore, {
    returnSourceDocuments: true,
  })
  const result = await qaChain.call({ query })
  const { text, sourceDocuments } = result
  res
    .json({
      answer: text,
      sources: sourceDocuments.map((doc) => doc.pageContent),
    })
    .status(200)
})

const port = process.env.PORT || 4000
app.listen(port, () => {
  console.log(`Server running on port ${port}`)
})
